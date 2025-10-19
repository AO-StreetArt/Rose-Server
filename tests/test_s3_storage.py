from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
from botocore.exceptions import BotoCoreError, ClientError

import s3_storage
from s3_storage import (
    S3ObjectNotFound,
    S3StorageClient,
    S3StorageError,
)
def make_client_error(code: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": "boom"}},
        operation_name="TestOperation",
    )


def test_upload_fileobj_includes_content_type(monkeypatch):
    client = mock.Mock()
    storage = S3StorageClient(
        client,
        bucket="bucket",
        public_prefix="https://cdn.example.com/assets",
    )
    fake_uuid = SimpleNamespace(hex="deadbeef")
    monkeypatch.setattr(s3_storage, "uuid4", lambda: fake_uuid)

    fileobj = object()
    result = storage.upload_fileobj(fileobj, filename="demo.png", content_type="image/png")

    client.upload_fileobj.assert_called_once_with(
        fileobj,
        "bucket",
        "deadbeef.png",
        ExtraArgs={"ContentType": "image/png"},
    )
    assert result.key == "deadbeef.png"
    assert result.url == "https://cdn.example.com/assets/deadbeef.png"


def test_upload_fileobj_raises_storage_error(monkeypatch):
    client = mock.Mock()
    client.upload_fileobj.side_effect = make_client_error("500")
    storage = S3StorageClient(client, bucket="bucket")
    monkeypatch.setattr(s3_storage, "uuid4", lambda: SimpleNamespace(hex="deadbeef"))

    with pytest.raises(S3StorageError):
        storage.upload_fileobj(object(), filename="file.txt")


def test_upload_artifact_bytes_success(monkeypatch):
    client = mock.Mock()
    storage = S3StorageClient(
        client,
        bucket="bucket",
        prefix="generated",
        public_prefix="https://cdn.example.com/assets",
    )
    monkeypatch.setattr(s3_storage, "uuid4", lambda: SimpleNamespace(hex="cafebabe"))

    result = storage.upload_artifact_bytes(
        b"data",
        artifact_label="DepthMask",
        key_hint="mask_base64",
        content_type="image/jpeg",
    )

    client.put_object.assert_called_once_with(
        Bucket="bucket",
        Key="generated/depthmask-mask-cafebabe.png",
        Body=b"data",
        ContentType="image/jpeg",
    )
    assert result.key == "generated/depthmask-mask-cafebabe.png"
    assert result.url == "https://cdn.example.com/assets/generated/depthmask-mask-cafebabe.png"


def test_upload_artifact_bytes_validates_data(monkeypatch):
    storage = S3StorageClient(mock.Mock(), bucket="bucket")

    with pytest.raises(ValueError):
        storage.upload_artifact_bytes(
            b"",
            artifact_label="DepthMask",
            key_hint="mask_base64",
        )


def test_upload_artifact_bytes_wraps_errors(monkeypatch):
    client = mock.Mock()
    client.put_object.side_effect = BotoCoreError()
    storage = S3StorageClient(client, bucket="bucket")
    monkeypatch.setattr(s3_storage, "uuid4", lambda: SimpleNamespace(hex="deadbeef"))

    with pytest.raises(S3StorageError):
        storage.upload_artifact_bytes(
            b"data",
            artifact_label="DepthMask",
            key_hint="mask_base64",
        )


def test_get_object_success():
    client = mock.Mock()
    client.get_object.return_value = {"Body": "payload"}
    storage = S3StorageClient(client, bucket="bucket")

    result = storage.get_object("key")

    client.get_object.assert_called_once_with(Bucket="bucket", Key="key")
    assert result == {"Body": "payload"}


def test_get_object_handles_missing_key():
    client = mock.Mock()
    client.get_object.side_effect = make_client_error("NoSuchKey")
    storage = S3StorageClient(client, bucket="bucket")

    with pytest.raises(S3ObjectNotFound):
        storage.get_object("missing")


def test_get_object_wraps_other_errors():
    client = mock.Mock()
    client.get_object.side_effect = BotoCoreError()
    storage = S3StorageClient(client, bucket="bucket")

    with pytest.raises(S3StorageError):
        storage.get_object("key")


def test_fetch_object_bytes_success(monkeypatch):
    class FakeBody:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self.closed = False

        def read(self) -> bytes:
            return self._payload

        def close(self) -> None:
            self.closed = True

    fake_body = FakeBody(b"hello")
    client = mock.Mock()
    client.get_object.return_value = {"Body": fake_body}
    storage = S3StorageClient(client, bucket="bucket")

    data = storage.fetch_object_bytes(key="path/to/object.dat")

    client.get_object.assert_called_once_with(Bucket="bucket", Key="path/to/object.dat")
    assert data == b"hello"
    assert fake_body.closed is True


def test_fetch_object_bytes_requires_body():
    client = mock.Mock()
    client.get_object.return_value = {}
    storage = S3StorageClient(client, bucket="bucket")

    with pytest.raises(S3StorageError):
        storage.fetch_object_bytes(key="path")


def test_fetch_object_bytes_requires_non_empty_data():
    class FakeBody:
        def read(self) -> bytes:
            return b""

        def close(self) -> None:
            pass

    client = mock.Mock()
    client.get_object.return_value = {"Body": FakeBody()}
    storage = S3StorageClient(client, bucket="bucket")

    with pytest.raises(S3StorageError):
        storage.fetch_object_bytes(key="path")


def test_fetch_object_bytes_not_found():
    client = mock.Mock()
    client.get_object.side_effect = make_client_error("404")
    storage = S3StorageClient(client, bucket="bucket")

    with pytest.raises(S3ObjectNotFound):
        storage.fetch_object_bytes(key="missing")


def test_build_public_url_variants():
    client = mock.Mock()
    storage_default = S3StorageClient(client, bucket="bucket")
    storage_regional = S3StorageClient(client, bucket="bucket", region="eu-west-1")
    storage_public = S3StorageClient(
        client,
        bucket="bucket",
        public_prefix="https://cdn.example.com/assets",
    )

    assert (
        storage_default.build_public_url("file.png")
        == "https://bucket.s3.amazonaws.com/file.png"
    )
    assert (
        storage_regional.build_public_url("file.png")
        == "https://bucket.s3.eu-west-1.amazonaws.com/file.png"
    )
    assert (
        storage_public.build_public_url("file.png")
        == "https://cdn.example.com/assets/file.png"
    )


def test_sanitize_key_filters_unsafe_segments():
    client = mock.Mock()
    storage = S3StorageClient(client, bucket="bucket")

    assert storage.sanitize_key("folder/file.png") == "folder/file.png"
    assert storage.sanitize_key("../evil") is None
    assert storage.sanitize_key("folder//file") is None
    assert storage.sanitize_key("") is None
