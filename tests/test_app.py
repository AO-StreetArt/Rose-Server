from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest import mock

import pytest

from app import create_app
from config import Settings
from s3_storage import S3ObjectNotFound


@pytest.fixture
def fake_settings(tmp_path):
    react_build = tmp_path / "build"
    react_build.mkdir()
    (react_build / "index.html").write_text("<html></html>", encoding="utf-8")
    return Settings(
        react_build_path=react_build,
        auth0_domain="example.auth0.com",
        auth0_audience="https://api.example.com",
        auth0_client_id="client-id",
        bedrock_agent_id="agent-id",
        bedrock_agent_alias_id="alias-id",
        bedrock_region="us-east-1",
        sagemaker_region="us-east-1",
        sagemaker_depth_endpoint="depth-endpoint",
        sagemaker_segmentation_endpoint="seg-endpoint",
        require_auth_for_health=False,
        require_auth_for_ui=False,
        s3_bucket_name="bucket",
        s3_region_name="us-east-1",
        s3_object_prefix="uploads/",
    )


def test_create_app_wires_shared_aws_session(fake_settings):
    mock_session = mock.Mock()
    mock_session.client.return_value = mock.Mock()

    with mock.patch("app.Session", return_value=mock_session) as session_cls, \
        mock.patch("app.BedrockAgentClient") as bedrock_cls, \
        mock.patch("app.SageMakerRuntimeClient") as sagemaker_cls:

        flask_app = create_app(settings=fake_settings)

    session_cls.assert_called_once_with()
    bedrock_cls.assert_called_once_with(
        region="us-east-1",
        agent_id="agent-id",
        agent_alias_id="alias-id",
        session=mock_session,
    )
    sagemaker_cls.assert_called_once_with(
        region="us-east-1",
        depth_endpoint="depth-endpoint",
        segmentation_endpoint="seg-endpoint",
        session=mock_session,
    )
    mock_session.client.assert_called_once_with("s3", region_name="us-east-1")
    assert flask_app.extensions["aws_session"] is mock_session
    assert flask_app.extensions["bedrock_client"] is bedrock_cls.return_value
    assert flask_app.extensions["sagemaker_client"] is sagemaker_cls.return_value


@pytest.fixture
def app_client(fake_settings, monkeypatch):
    mock_session = mock.Mock()
    monkeypatch.setattr("app.Session", mock.Mock(return_value=mock_session))
    monkeypatch.setattr("app.Auth0TokenVerifier", mock.Mock())

    bedrock_client = mock.Mock()
    monkeypatch.setattr("app.BedrockAgentClient", mock.Mock(return_value=bedrock_client))

    sagemaker_client = mock.Mock()
    monkeypatch.setattr("app.SageMakerRuntimeClient", mock.Mock(return_value=sagemaker_client))

    storage_client = mock.Mock()
    monkeypatch.setattr("app.S3StorageClient", mock.Mock(return_value=storage_client))

    def identity_decorator(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    monkeypatch.setattr("app.requires_auth", identity_decorator)

    flask_app = create_app(settings=fake_settings)
    flask_app.config.update(TESTING=True)

    return flask_app.test_client(), flask_app, storage_client, sagemaker_client


def test_ping_endpoint_returns_ok(app_client):
    client, *_ = app_client
    response = client.get("/ping/")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_health_requires_auth_when_configured(app_client, monkeypatch):
    client, flask_app, *_ = app_client
    flask_app.config["REQUIRE_AUTH_FOR_HEALTH"] = True

    calls = SimpleNamespace(invoked=False)

    def fake_ensure():
        calls.invoked = True

    monkeypatch.setattr("app.ensure_authorized", fake_ensure)

    response = client.get("/health/")

    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy"}
    assert calls.invoked is True


def test_health_returns_unauthorized_on_auth_error(app_client, monkeypatch):
    client, flask_app, *_ = app_client
    flask_app.config["REQUIRE_AUTH_FOR_HEALTH"] = True

    class FakeAuthError(Exception):
        pass

    def fake_ensure():
        raise FakeAuthError("nope")

    monkeypatch.setattr("app.ensure_authorized", fake_ensure)
    monkeypatch.setattr("app.AuthError", FakeAuthError)

    response = client.get("/health/")

    assert response.status_code == 401
    assert response.get_json() == {"error": "authentication failed"}


def test_chat_route_delegates_to_handler(app_client, monkeypatch):
    client, _, *_ = app_client
    monkeypatch.setattr("app.handle_chat_request", lambda app: ("ok", 200))

    response = client.post("/api/chat", json={"message": "hi"})

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "ok"


def test_depth_estimation_validates_input(app_client):
    client, *_ = app_client

    response = client.post("/api/depth-estimation", json={})

    assert response.status_code == 400
    assert response.get_json() == {"error": "imageBase64 is required"}


def test_depth_estimation_invokes_sagemaker(app_client):
    client, flask_app, _, sagemaker_client = app_client
    payload = base64.b64encode(b"image-bytes").decode("ascii")
    sagemaker_client.invoke_depth_estimator.return_value = {"depth": "ok"}

    response = client.post(
        "/api/depth-estimation",
        json={"imageBase64": payload, "estimator": "midas", "outputFormat": "json"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"depth": "ok"}
    sagemaker_client.invoke_depth_estimator.assert_called_once()
    args, kwargs = sagemaker_client.invoke_depth_estimator.call_args
    assert kwargs["estimator"] == "midas"
    assert kwargs["output_format"] == "json"
    assert kwargs["image_bytes"] == b"image-bytes"


def test_segmentation_handles_invalid_base64(app_client):
    client, *_ = app_client

    response = client.post("/api/segmentation", json={"imageBase64": "not-base64"})

    assert response.status_code == 400
    assert response.get_json() == {"error": "imageBase64 must be valid base64"}


def test_upload_file_success(app_client):
    client, flask_app, storage_client, _ = app_client
    storage_client.upload_fileobj.return_value = SimpleNamespace(url="https://example.com/file.png")
    data = {"file": (io.BytesIO(b"content"), "photo.png")}

    response = client.post("/api/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert response.get_json() == {"url": "https://example.com/file.png"}
    storage_client.upload_fileobj.assert_called_once()


def test_upload_file_missing_payload(app_client):
    client, *_ = app_client

    response = client.post("/api/upload", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json() == {"error": "file is required"}


def test_get_file_handles_not_found(app_client):
    client, flask_app, storage_client, _ = app_client
    storage_client.sanitize_key.return_value = "safe/key.txt"
    storage_client.get_object.side_effect = S3ObjectNotFound("missing")

    response = client.get("/api/files/foo.txt")

    assert response.status_code == 404
    assert response.get_json() == {"error": "file not found"}


def test_get_file_streams_content(app_client):
    client, flask_app, storage_client, _ = app_client
    storage_client.sanitize_key.return_value = "safe/path.txt"

    class FakeBody:
        def __init__(self):
            self.closed = False
            self._chunks = [b"part1", b"part2"]

        def read(self, size):
            if self._chunks:
                return self._chunks.pop(0)
            return b""

        def close(self):
            self.closed = True

    fake_body = FakeBody()
    storage_client.get_object.return_value = {
        "Body": fake_body,
        "ContentType": "text/plain",
        "ContentLength": 10,
    }

    response = client.get("/api/files/doc.txt")

    assert response.status_code == 200
    assert response.mimetype == "text/plain"
    assert response.headers["Content-Disposition"] == 'inline; filename="path.txt"'
    assert response.data == b"part1part2"
    assert fake_body.closed is True


def test_get_file_invalid_key(app_client):
    client, _, storage_client, _ = app_client
    storage_client.sanitize_key.return_value = None

    response = client.get("/api/files/../../secret")

    assert response.status_code == 400
    assert response.get_json() == {"error": "invalid object key"}


def test_serve_react_invalid_path(app_client):
    client, *_ = app_client

    response = client.get("/../secrets.txt")

    assert response.status_code == 400
    assert response.get_json() == {"error": "invalid path"}


def test_serve_react_requires_auth(app_client, monkeypatch):
    client, flask_app, *_ = app_client
    flask_app.config["REQUIRE_AUTH_FOR_UI"] = True

    calls = SimpleNamespace(count=0)

    def fake_ensure():
        calls.count += 1

    monkeypatch.setattr("app.ensure_authorized", fake_ensure)

    response = client.get("/")

    assert response.status_code == 200
    assert response.data  # index.html exists
    assert calls.count == 1
