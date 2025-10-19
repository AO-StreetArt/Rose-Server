from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from botocore.exceptions import BotoCoreError, ClientError
from werkzeug.utils import secure_filename
from uuid import uuid4

logger = logging.getLogger(__name__)


class S3StorageError(RuntimeError):
    """Base class for storage failures."""


class S3ObjectNotFound(S3StorageError):
    """Raised when the requested S3 object could not be found."""


@dataclass(frozen=True)
class StoredObject:
    key: str
    url: str


class S3StorageClient:
    """Helper that wraps boto3 S3 interactions and shared app configuration."""

    def __init__(
        self,
        client: Any,
        *,
        bucket: str,
        prefix: str = "",
        public_prefix: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket name is required.")

        self._client = client
        self._bucket = bucket
        self._prefix = (prefix or "").strip().strip("/")
        self._public_prefix = public_prefix.rstrip("/") if public_prefix else None
        self._region = region

    @property
    def bucket(self) -> str:
        return self._bucket

    def upload_fileobj(
        self,
        fileobj: Any,
        *,
        filename: str,
        content_type: Optional[str] = None,
    ) -> StoredObject:
        key = self._generate_upload_key(filename)
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        try:
            if extra_args:
                self._client.upload_fileobj(fileobj, self._bucket, key, ExtraArgs=extra_args)
            else:
                self._client.upload_fileobj(fileobj, self._bucket, key)
        except (BotoCoreError, ClientError) as exc:
            raise S3StorageError("Failed to upload file to S3.") from exc

        return StoredObject(key=key, url=self.build_public_url(key))

    def upload_artifact_bytes(
        self,
        data: bytes,
        *,
        artifact_label: str,
        key_hint: str,
        content_type: str = "image/png",
    ) -> StoredObject:
        if not data:
            raise ValueError("Artifact data is empty.")

        key = self._generate_artifact_key(artifact_label=artifact_label, key_hint=key_hint)
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        except (BotoCoreError, ClientError) as exc:
            raise S3StorageError("Failed to upload artifact to S3.") from exc

        return StoredObject(key=key, url=self.build_public_url(key))

    def get_object(self, key: str) -> Any:
        try:
            return self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"NoSuchKey", "404"}:
                raise S3ObjectNotFound(f"S3 object not found: {key}") from exc
            raise S3StorageError(f"Failed to retrieve S3 object: {key}") from exc
        except BotoCoreError as exc:
            raise S3StorageError(f"Failed to retrieve S3 object: {key}") from exc

    def fetch_object_bytes(self, *, key: str, bucket: Optional[str] = None) -> bytes:
        target_bucket = bucket or self._bucket
        try:
            response = self._client.get_object(Bucket=target_bucket, Key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"NoSuchKey", "404"}:
                raise S3ObjectNotFound(f"S3 object not found: {target_bucket}/{key}") from exc
            raise S3StorageError(
                f"Failed to download S3 object: {target_bucket}/{key}"
            ) from exc
        except BotoCoreError as exc:
            raise S3StorageError(
                f"Failed to download S3 object: {target_bucket}/{key}"
            ) from exc

        body = response.get("Body")
        if not body:
            raise S3StorageError("S3 object response did not include a body.")
        try:
            data = body.read()
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if not data:
            raise S3StorageError("S3 object body was empty.")
        return data

    def build_public_url(self, key: str) -> str:
        if self._public_prefix:
            return f"{self._public_prefix}/{key}"

        if self._region and self._region != "us-east-1":
            return f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{key}"
        return f"https://{self._bucket}.s3.amazonaws.com/{key}"

    def sanitize_key(self, untrusted: str) -> Optional[str]:
        if not untrusted:
            return None

        parts = []
        for part in untrusted.split("/"):
            if part in {"", ".", ".."}:
                return None
            safe_part = secure_filename(part)
            if not safe_part:
                return None
            parts.append(safe_part)

        return "/".join(parts) if parts else None

    def _generate_upload_key(self, filename: str) -> str:
        safe_name = secure_filename(filename or "") or "file"
        suffix = Path(safe_name).suffix
        key = f"{uuid4().hex}{suffix}"
        return self._apply_prefix(key)

    def _generate_artifact_key(self, *, artifact_label: str, key_hint: str) -> str:
        base_name = self._slugify(f"{artifact_label}-{self._base_key_name(key_hint)}")
        unique = uuid4().hex
        filename = f"{base_name}-{unique}.png"
        return self._apply_prefix(filename)

    def _apply_prefix(self, key: str) -> str:
        if not self._prefix:
            return key
        return f"{self._prefix}/{key}"

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9-]+", "-", value.lower())
        slug = re.sub(r"-{2,}", "-", slug)
        return slug.strip("-") or "image"

    @staticmethod
    def _base_key_name(key: str) -> str:
        lowered = key.lower()
        suffix = "_base64"
        if lowered.endswith(suffix):
            return key[: -len(suffix)]
        return key

