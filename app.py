from __future__ import annotations

import logging
from http import HTTPStatus
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory

from auth import Auth0TokenVerifier, AuthError, ensure_authorized, requires_auth
from bedrock import BedrockAgentClient
from config import Settings

logger = logging.getLogger(__name__)


def create_app(settings: Optional[Settings] = None) -> Flask:
    settings = settings or Settings()
    settings.validate()

    app = Flask(__name__, static_folder=None)
    app.config.update(settings.as_flask_config())

    # Register Auth0 verifier and Bedrock client for later use.
    app.extensions["auth0_verifier"] = Auth0TokenVerifier(
        domain=app.config["AUTH0_DOMAIN"],
        audience=app.config["AUTH0_AUDIENCE"],
        client_id=app.config.get("AUTH0_CLIENT_ID"),
    )
    app.extensions["bedrock_client"] = BedrockAgentClient(
        region=app.config["BEDROCK_REGION"],
        agent_id=app.config["BEDROCK_AGENT_ID"],
        agent_alias_id=app.config["BEDROCK_AGENT_ALIAS_ID"],
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
            except AuthError as exc:
                return jsonify({"error": str(exc)}), HTTPStatus.UNAUTHORIZED
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
        try:
            result = client.invoke(
                user_input=message,
                session_id=session_id,
                enable_trace=enable_trace,
            )
        except RuntimeError as exc:
            logger.exception("Bedrock invocation error")
            return jsonify({"error": str(exc)}), HTTPStatus.BAD_GATEWAY

        return jsonify({"sessionId": result["sessionId"], "messages": result["messages"], "trace": result.get("trace")})

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_react(path: str):
        build_dir = Path(app.config["REACT_BUILD_PATH"])
        candidate = (build_dir / path).resolve()
        try:
            candidate.relative_to(build_dir)
        except ValueError:
            return jsonify({"error": "invalid path"}), HTTPStatus.BAD_REQUEST

        if path and candidate.is_file():
            return send_from_directory(build_dir, path)

        if app.config.get("REQUIRE_AUTH_FOR_UI", False):
            try:
                ensure_authorized()
            except AuthError as exc:
                return jsonify({"error": str(exc)}), HTTPStatus.UNAUTHORIZED

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
