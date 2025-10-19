from __future__ import annotations

import io
from types import SimpleNamespace
from urllib.error import URLError

import pytest

import image_utils
from bedrock_action_helper import (
    _as_list,
    _base_key_name,
    _derive_url_key,
    _get_param,
    _humanize_key,
    _infer_function_from_path,
    _normalize_parameters,
)
from image_utils import (
    extract_mask_bytes,
    fetch_public_image_bytes,
    fetch_s3_image_bytes,
    is_s3_url,
    parse_s3_url,
    _ensure_extension,
    _prepare_image_reference,
)
from s3_storage import S3StorageError


def test_normalize_parameters_coerces_types():
    parameters = [
        {"name": "count", "value": "3", "type": "integer"},
        {"name": "ratio", "value": "2.5", "type": "number"},
        {"name": "active", "value": "TRUE", "type": "boolean"},
        {"name": "tags", "value": '["a","b"]', "type": "array"},
        {"name": "notes", "value": "plain text"},
        {"name": None, "value": "ignored"},
    ]

    normalized = _normalize_parameters(parameters)

    assert normalized["count"] == 3
    assert normalized["ratio"] == 2.5
    assert normalized["active"] is True
    assert normalized["tags"] == ["a", "b"]
    assert normalized["notes"] == "plain text"
    assert None not in normalized


def test_get_param_case_insensitive_lookup():
    parameters = {"UserName": "alice"}
    assert _get_param(parameters, "username") == "alice"
    assert _get_param(parameters, "missing") is None


def test_as_list_parses_and_wraps_values():
    assert _as_list(["a", "b"]) == ["a", "b"]
    assert _as_list('["x", 1]') == ["x", 1]
    assert _as_list("single") == ["single"]
    assert _as_list(None) is None


def test_key_helpers_and_humanizer():
    assert _base_key_name("mask_base64") == "mask"
    assert _derive_url_key("mask_base64") == "mask_url"
    assert _humanize_key("mask_base64", default_label="artifact-label") == "Mask"
    assert _humanize_key("_base64", default_label="artifact-label") == "Artifact Label"


def test_infer_function_from_path_handles_variations():
    assert _infer_function_from_path("/foo-bar/baz/", "Default") == "FooBarBaz"
    assert _infer_function_from_path("", "MyAction") == "MyAction"
    assert _infer_function_from_path(None, "") == "Function"


def test_fetch_public_image_bytes_success(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    def fake_urlopen(request, timeout: int):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        assert timeout == 30
        return FakeResponse(b"imagedata")

    monkeypatch.setattr(image_utils, "urlopen", fake_urlopen)

    data = fetch_public_image_bytes("https://example.com/image.png")

    assert data == b"imagedata"
    assert captured["url"] == "https://example.com/image.png"
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["user-agent"] == "rose-server/1.0"


def test_fetch_public_image_bytes_raises_for_errors(monkeypatch):
    def fake_urlopen(*args, **kwargs):
        raise URLError("boom")

    monkeypatch.setattr(image_utils, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError):
        fetch_public_image_bytes("https://example.com/image.png")


def test_fetch_public_image_bytes_rejects_other_schemes():
    with pytest.raises(ValueError):
        fetch_public_image_bytes("ftp://example.com/image.png")


def test_fetch_public_image_bytes_rejects_empty_response(monkeypatch):
    class FakeResponse:
        def read(self) -> bytes:
            return b""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(image_utils, "urlopen", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError):
        fetch_public_image_bytes("https://example.com/image.png")


def test_is_s3_url_and_parse_s3_url():
    assert is_s3_url("s3://bucket/key")
    assert not is_s3_url("https://bucket/key")
    bucket, key = parse_s3_url("s3://bucket/path/to/object.png")
    assert bucket == "bucket"
    assert key == "path/to/object.png"
    with pytest.raises(ValueError):
        parse_s3_url("s3://missing-key")


def test_fetch_s3_image_bytes_success(monkeypatch):
    class DummyStorage:
        def fetch_object_bytes(self, *, bucket: str, key: str) -> bytes:
            assert bucket == "bucket"
            assert key == "path/to/object.png"
            return b"content"

    data = fetch_s3_image_bytes("s3://bucket/path/to/object.png", DummyStorage())

    assert data == b"content"


def test_fetch_s3_image_bytes_wraps_errors(monkeypatch):
    class DummyStorage:
        def fetch_object_bytes(self, *, bucket: str, key: str) -> bytes:
            raise S3StorageError("boom")

    with pytest.raises(RuntimeError):
        fetch_s3_image_bytes("s3://bucket/path/to/object.png", DummyStorage())


def test_prepare_image_reference_returns_existing_uri():
    storage = SimpleNamespace(bucket="ignored")
    attachment = {
        "source": {
            "s3Location": {"uri": "s3://bucket/key"},
        }
    }

    assert _prepare_image_reference(attachment, storage) == "s3://bucket/key"


def test_prepare_image_reference_uploads_bytes(monkeypatch):
    class RecordingStorage:
        def __init__(self) -> None:
            self.bucket = "bucket"
            self.calls = []

        def upload_fileobj(self, buffer: io.BytesIO, *, filename: str, content_type: str):
            self.calls.append(
                {
                    "filename": filename,
                    "content_type": content_type,
                    "data": buffer.getvalue(),
                }
            )
            return SimpleNamespace(key="uploads/123")

    storage = RecordingStorage()
    attachment = {
        "data": b"\x00\x01",
        "contentType": "image/jpeg",
        "name": "photo",
    }

    uri = _prepare_image_reference(attachment, storage)

    assert uri == "s3://bucket/uploads/123"
    assert len(storage.calls) == 1
    call = storage.calls[0]
    assert call["data"] == b"\x00\x01"
    assert call["content_type"] == "image/jpeg"
    assert call["filename"].startswith("photo")
    assert call["filename"].endswith((".jpg", ".jpe"))


def test_prepare_image_reference_requires_bytes():
    storage = SimpleNamespace(bucket="bucket", upload_fileobj=lambda *args, **kwargs: None)
    with pytest.raises(ValueError):
        _prepare_image_reference({"data": "not-bytes"}, storage)


def test_prepare_image_reference_without_data_returns_none():
    storage = SimpleNamespace(bucket="bucket", upload_fileobj=lambda *args, **kwargs: None)
    assert _prepare_image_reference({}, storage) is None


def test_ensure_extension_preserves_or_adds_extensions(monkeypatch):
    assert _ensure_extension("image.png", "image/png") == "image.png"
    result = _ensure_extension("photo", "image/jpeg")
    assert result.endswith((".jpg", ".jpe"))
    assert _ensure_extension("mask", "application/x-custom") == "mask.png"


def test_extract_mask_bytes_builds_image(monkeypatch):
    created = {}

    class FakeImage:
        def __init__(self, size):
            self.size = size
            self.data = None

        def putdata(self, data):
            self.data = data

        def save(self, buffer, format):
            buffer.write(b"fakepng")

    class FakeImageModule:
        @staticmethod
        def new(mode, size):
            assert mode == "L"
            created["image"] = FakeImage(size)
            return created["image"]

    monkeypatch.setattr(image_utils, "Image", FakeImageModule)

    mask_bytes = extract_mask_bytes({"predictions": [[0, 255], ["10", 20]]})

    assert mask_bytes == b"fakepng"
    assert created["image"].size == (2, 2)
    assert created["image"].data == [0, 255, 10, 20]


def test_extract_mask_bytes_returns_none_without_image(monkeypatch):
    monkeypatch.setattr(image_utils, "Image", None)
    assert extract_mask_bytes({"predictions": [[0]]}) is None


def test_extract_mask_bytes_returns_none_for_invalid_predictions(monkeypatch):
    class FakeImageModule:
        @staticmethod
        def new(*args, **kwargs):
            raise AssertionError("Image.new should not be called for invalid predictions")

    monkeypatch.setattr(image_utils, "Image", FakeImageModule)

    assert extract_mask_bytes({"predictions": [{"unexpected": "value"}, []]}) is None
