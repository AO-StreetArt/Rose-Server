from __future__ import annotations

import base64
import binascii
import logging
from http import HTTPStatus
from pathlib import Path
from typing import Optional
from uuid import uuid4

from botocore.exceptions import BotoCoreError, ClientError
from boto3.session import Session
from flask import Flask, jsonify, request, send_from_directory, g
from werkzeug.utils import secure_filename

from auth import Auth0TokenVerifier, AuthError, ensure_authorized, requires_auth
from bedrock import BedrockAgentClient
from sagemaker import SageMakerRuntimeClient
from config import Settings

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
    app.extensions["s3_client"] = aws_session.client(
        "s3",
        region_name=app.config.get("S3_REGION_NAME"),
    )
    app.extensions["sagemaker_client"] = SageMakerRuntimeClient(
        region=app.config["SAGEMAKER_REGION"],
        depth_endpoint=app.config["SAGEMAKER_DEPTH_ENDPOINT"],
        segmentation_endpoint=app.config["SAGEMAKER_SEGMENTATION_ENDPOINT"],
        session=aws_session,
    )

    register_routes(app)
    register_error_handlers(app)

    return app


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
        payload = request.get_json(silent=True) or {}
        message = payload.get("message")
        if not message:
            return jsonify({"error": "message is required"}), HTTPStatus.BAD_REQUEST

        session_id = payload.get("sessionId")
        enable_trace = bool(payload.get("enableTrace"))

        client: BedrockAgentClient = app.extensions["bedrock_client"]
        g.sagemaker_client = app.extensions["sagemaker_client"]
        try:
            result = client.invoke(
                user_input=message,
                session_id=session_id,
                enable_trace=enable_trace,
            )
        except RuntimeError:
            logger.exception("Bedrock invocation error")
            return jsonify({"error": "chat service unavailable"}), HTTPStatus.BAD_GATEWAY

        return jsonify({"sessionId": result["sessionId"], "messages": result["messages"], "trace": result.get("trace")})

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
        suffix = Path(safe_name).suffix

        prefix = app.config.get("S3_OBJECT_PREFIX", "")
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"

        object_key = f"{prefix}{uuid4().hex}{suffix}"
        bucket_name = app.config["S3_BUCKET_NAME"]
        s3_client = app.extensions["s3_client"]

        extra_args = {}
        if uploaded_file.mimetype:
            extra_args["ContentType"] = uploaded_file.mimetype

        try:
            if extra_args:
                s3_client.upload_fileobj(uploaded_file, bucket_name, object_key, ExtraArgs=extra_args)
            else:
                s3_client.upload_fileobj(uploaded_file, bucket_name, object_key)
        except (BotoCoreError, ClientError) as exc:
            logger.exception("Failed to upload file to S3")
            return jsonify({"error": "failed to upload file"}), HTTPStatus.BAD_GATEWAY

        public_prefix = app.config.get("S3_PUBLIC_URL_PREFIX")
        if public_prefix:
            public_prefix = public_prefix.rstrip("/")
            file_url = f"{public_prefix}/{object_key}"
        else:
            region = app.config.get("S3_REGION_NAME")
            if region and region != "us-east-1":
                file_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_key}"
            else:
                file_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"

        return jsonify({"url": file_url})

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
