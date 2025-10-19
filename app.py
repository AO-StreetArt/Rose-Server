from __future__ import annotations

import base64
import binascii
import logging
from logging.handlers import RotatingFileHandler
from http import HTTPStatus
from pathlib import Path
from typing import Optional
from boto3.session import Session
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from auth import Auth0TokenVerifier, AuthError, ensure_authorized, requires_auth
from bedrock import BedrockAgentClient
from chat_service import handle_chat_request
from sagemaker import SageMakerRuntimeClient
from config import Settings
from s3_storage import S3ObjectNotFound, S3StorageClient, S3StorageError

logger = logging.getLogger(__name__)


def _sanitize_subpath(untrusted: str) -> Optional[Path]:
    """Validate and sanitize a relative path using secure_filename on each component."""
    if not untrusted:
        return None

    try:
        prospective = Path(untrusted)
    except (TypeError, ValueError):
        return None

    if prospective.is_absolute():
        return None

    sanitized_parts = []
    for part in prospective.parts:
        if part in {"", ".", ".."}:
            return None
        secure_part = secure_filename(part)
        if not secure_part:
            return None
        sanitized_parts.append(secure_part)

    return Path(*sanitized_parts)

def create_app(settings: Optional[Settings] = None) -> Flask:
    settings = settings or Settings()
    settings.validate()

    app = Flask(__name__, static_folder=None)
    app.config.update(settings.as_flask_config())

    # Register Auth0 verifier and shared AWS clients for later use.
    app.extensions["auth0_verifier"] = Auth0TokenVerifier(
        domain=app.config["AUTH0_DOMAIN"],
        audience=app.config["AUTH0_AUDIENCE"],
        client_id=app.config.get("AUTH0_CLIENT_ID"),
    )
    aws_session = Session()
    app.extensions["aws_session"] = aws_session
    app.extensions["bedrock_client"] = BedrockAgentClient(
        region=app.config["BEDROCK_REGION"],
        agent_id=app.config["BEDROCK_AGENT_ID"],
        agent_alias_id=app.config["BEDROCK_AGENT_ALIAS_ID"],
        session=aws_session,
    )
    boto_s3_client = aws_session.client(
        "s3",
        region_name=app.config.get("S3_REGION_NAME"),
    )
    app.extensions["s3_client"] = boto_s3_client
    app.extensions["s3_storage"] = S3StorageClient(
        boto_s3_client,
        bucket=app.config["S3_BUCKET_NAME"],
        prefix=app.config.get("S3_OBJECT_PREFIX", ""),
        public_prefix=app.config.get("S3_PUBLIC_URL_PREFIX"),
        region=app.config.get("S3_REGION_NAME"),
    )
    app.extensions["sagemaker_client"] = SageMakerRuntimeClient(
        region=app.config["SAGEMAKER_REGION"],
        depth_endpoint=app.config["SAGEMAKER_DEPTH_ENDPOINT"],
        segmentation_endpoint=app.config["SAGEMAKER_SEGMENTATION_ENDPOINT"],
        session=aws_session,
    )

    configure_logging(app)
    register_routes(app)
    register_error_handlers(app)

    return app


def configure_logging(app: Flask) -> None:
    """Attach a rotating file handler so server logs are written to disk."""
    log_path_setting = app.config.get("SERVER_LOG_PATH")
    if log_path_setting:
        log_path = Path(log_path_setting)
        if not log_path.is_absolute():
            log_path = Path(app.instance_path) / log_path
    else:
        log_path = Path(app.instance_path) / "logs" / "rose-server.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", None) == str(log_path):
            return

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10_000_000,
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    root_logger.addHandler(file_handler)
    if root_logger.level == logging.NOTSET or root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)
    if app.logger.level > logging.DEBUG:
        app.logger.setLevel(logging.DEBUG)

def register_routes(app: Flask) -> None:
    @app.route("/ping/", methods=["GET"])
    def ping():
        return jsonify({"status": "ok"})

    @app.route("/health/", methods=["GET"])
    def health():
        if app.config.get("REQUIRE_AUTH_FOR_HEALTH", False):
            try:
                ensure_authorized()
            except AuthError:
                logger.exception("Health check authorization failed")
                return jsonify({"error": "authentication failed"}), HTTPStatus.UNAUTHORIZED
        return jsonify({"status": "healthy"})

    @app.route("/api/chat", methods=["POST"])
    @requires_auth()
    def chat():
        return handle_chat_request(app)

    @app.route("/api/depth-estimation", methods=["POST"])
    @requires_auth()
    def depth_estimation():
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("imageBase64") or payload.get("image_base64")
        if not image_b64:
            return jsonify({"error": "imageBase64 is required"}), HTTPStatus.BAD_REQUEST

        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except (binascii.Error, ValueError):
            return jsonify({"error": "imageBase64 must be valid base64"}), HTTPStatus.BAD_REQUEST

        estimator = payload.get("estimator") or "dpt"
        output_format = payload.get("outputFormat") or payload.get("output_format") or "png"

        client: SageMakerRuntimeClient = app.extensions["sagemaker_client"]
        try:
            result = client.invoke_depth_estimator(
                image_bytes=image_bytes,
                estimator=estimator,
                output_format=output_format,
            )
        except RuntimeError:
            logger.exception("Depth estimation failed")
            return jsonify({"error": "depth estimation unavailable"}), HTTPStatus.BAD_GATEWAY

        return jsonify(result)

    @app.route("/api/segmentation", methods=["POST"])
    @requires_auth()
    def segmentation():
        payload = request.get_json(silent=True) or {}
        image_b64 = payload.get("imageBase64") or payload.get("image_base64")
        if not image_b64:
            return jsonify({"error": "imageBase64 is required"}), HTTPStatus.BAD_REQUEST

        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except (binascii.Error, ValueError):
            return jsonify({"error": "imageBase64 must be valid base64"}), HTTPStatus.BAD_REQUEST

        content_type = payload.get("contentType") or payload.get("content_type") or "application/x-image"
        accept = payload.get("accept") or "application/json;verbose"

        client: SageMakerRuntimeClient = app.extensions["sagemaker_client"]
        try:
            result = client.invoke_segmentation(
                image_bytes=image_bytes,
                content_type=content_type,
                accept=accept,
            )
        except RuntimeError:
            logger.exception("Segmentation inference failed")
            return jsonify({"error": "segmentation service unavailable"}), HTTPStatus.BAD_GATEWAY

        return jsonify(result)

    @app.route("/api/upload", methods=["POST"])
    @requires_auth()
    def upload_file():
        uploaded_file = request.files.get("file")
        if not uploaded_file or uploaded_file.filename == "":
            return jsonify({"error": "file is required"}), HTTPStatus.BAD_REQUEST

        safe_name = secure_filename(uploaded_file.filename or "") or "file"
        storage: S3StorageClient = app.extensions["s3_storage"]
        try:
            stored_object = storage.upload_fileobj(
                uploaded_file,
                filename=safe_name,
                content_type=uploaded_file.mimetype,
            )
        except S3StorageError:
            logger.exception("Failed to upload file to S3")
            return jsonify({"error": "failed to upload file"}), HTTPStatus.BAD_GATEWAY

        return jsonify({"url": stored_object.url})

    @app.route("/api/files/<path:object_key>", methods=["GET"])
    @requires_auth()
    def get_file(object_key: str):
        storage: S3StorageClient = app.extensions["s3_storage"]
        sanitized_key = storage.sanitize_key(object_key)
        if not sanitized_key:
            return jsonify({"error": "invalid object key"}), HTTPStatus.BAD_REQUEST

        try:
            s3_response = storage.get_object(sanitized_key)
        except S3ObjectNotFound:
            return jsonify({"error": "file not found"}), HTTPStatus.NOT_FOUND
        except S3StorageError:
            logger.exception("Failed to retrieve S3 object %s", sanitized_key)
            return jsonify({"error": "failed to retrieve file"}), HTTPStatus.BAD_GATEWAY

        body = s3_response.get("Body")
        if body is None:
            return jsonify({"error": "file unavailable"}), HTTPStatus.BAD_GATEWAY

        def stream_body():
            try:
                while True:
                    chunk = body.read(8192)
                    if not chunk:
                        break
                    yield chunk
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()

        content_type = s3_response.get("ContentType") or "application/octet-stream"
        content_length = s3_response.get("ContentLength")
        response = Response(stream_body(), mimetype=content_type)
        if content_length is not None:
            response.headers["Content-Length"] = str(content_length)
        response.headers["Content-Disposition"] = f'inline; filename="{Path(sanitized_key).name}"'
        return response

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_react(path: str):
        build_dir = Path(app.config["REACT_BUILD_PATH"]).resolve()

        if path:
            sanitized = _sanitize_subpath(path)
            if sanitized is None:
                return jsonify({"error": "invalid path"}), HTTPStatus.BAD_REQUEST

            candidate = (build_dir / sanitized).resolve()
            try:
                relative_path = candidate.relative_to(build_dir)
            except (ValueError, OSError):
                return jsonify({"error": "invalid path"}), HTTPStatus.BAD_REQUEST

            if candidate.is_file():
                return send_from_directory(build_dir, relative_path.as_posix())

        if app.config.get("REQUIRE_AUTH_FOR_UI", False):
            try:
                ensure_authorized()
            except AuthError:
                logger.exception("UI authorization failed")
                return jsonify({"error": "authentication failed"}), HTTPStatus.UNAUTHORIZED

        index_path = build_dir / "index.html"
        if not index_path.exists():
            return jsonify({"error": "React build missing index.html"}), HTTPStatus.NOT_FOUND

        return send_from_directory(build_dir, "index.html")


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPStatus.UNAUTHORIZED)
    def unauthorized(error):
        description = getattr(error, "description", "Unauthorized")
        return jsonify({"error": description}), HTTPStatus.UNAUTHORIZED

    @app.errorhandler(HTTPStatus.BAD_REQUEST)
    def bad_request(error):
        description = getattr(error, "description", "Invalid request")
        return jsonify({"error": description}), HTTPStatus.BAD_REQUEST


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    flask_app = create_app()
    flask_app.run(host="0.0.0.0", port=5000)
