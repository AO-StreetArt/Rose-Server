from __future__ import annotations

import io
import logging
import mimetypes
import os
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:  # Pillow may be optional in some environments
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Image = None

from s3_storage import S3StorageClient, S3StorageError

logger = logging.getLogger(__name__)


def fetch_public_image_bytes(image_url: str) -> bytes:
    parsed = urlparse(image_url)
    scheme = (parsed.scheme or "").lower()

    if scheme in {"http", "https"}:
        request = Request(image_url, headers={"User-Agent": "rose-server/1.0"})
        try:
            with urlopen(request, timeout=30) as response:
                data = response.read()
        except (HTTPError, URLError) as exc:
            raise RuntimeError(f"Failed to download image from {image_url}") from exc
        if not data:
            raise RuntimeError("Downloaded image was empty.")
        return data

    raise ValueError(f"Unsupported ImageURL scheme: {scheme or 'unknown'}")


def is_s3_url(image_url: str) -> bool:
    return (urlparse(image_url).scheme or "").lower() == "s3"


def parse_s3_url(image_url: str) -> Tuple[str, str]:
    parsed = urlparse(image_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError("ImageURL must include bucket and key for S3 resources.")
    return bucket, key


def fetch_s3_image_bytes(image_url: str, storage: S3StorageClient) -> bytes:
    bucket, key = parse_s3_url(image_url)
    try:
        return storage.fetch_object_bytes(bucket=bucket, key=key)
    except S3StorageError as exc:
        raise RuntimeError(f"Failed to download image from s3://{bucket}/{key}") from exc


def _prepare_image_reference(
    image_attachment: Optional[Dict[str, Any]],
    storage: S3StorageClient,
) -> Optional[str]:
    if not image_attachment:
        return None

    source = image_attachment.get("source")
    if isinstance(source, dict):
        s3_location = source.get("s3Location")
        if isinstance(s3_location, dict):
            uri = s3_location.get("uri")
            if uri:
                return uri

    data = image_attachment.get("data")
    if data is None:
        return None
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("image data must be bytes.")

    content_type = (image_attachment.get("contentType") or "image/png").strip() or "image/png"
    filename_hint = (image_attachment.get("name") or "input-image").strip() or "input-image"
    filename = _ensure_extension(filename_hint, content_type)

    buffer = io.BytesIO(bytes(data))
    stored_object = storage.upload_fileobj(buffer, filename=filename, content_type=content_type)
    return f"s3://{storage.bucket}/{stored_object.key}"


def _ensure_extension(filename: str, content_type: str) -> str:
    name, ext = os.path.splitext(filename)
    if ext:
        return filename

    guessed = mimetypes.guess_extension(content_type.lower())
    if guessed:
        return f"{filename}{guessed}"
    if content_type.lower() == "image/jpeg":
        return f"{filename}.jpg"
    return f"{filename}.png"


def extract_mask_bytes(result: Dict[str, Any]) -> Optional[bytes]:
    if Image is None:
        logger.warning("Pillow is not available; cannot generate segmentation mask image.")
        return None

    predictions = result.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        return None

    processed_rows = []
    min_width: Optional[int] = None

    for row in predictions:
        if not isinstance(row, list) or not row:
            continue
        if min_width is None or len(row) < min_width:
            min_width = len(row)

    if min_width is None or min_width <= 0:
        return None

    for row in predictions:
        if not isinstance(row, list):
            continue
        values = []
        for value in row[:min_width]:
            numeric: float
            if isinstance(value, (int, float)):
                numeric = float(value)
            else:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = 0.0
            clamped = max(0, min(255, int(round(numeric))))
            values.append(clamped)
        if values:
            processed_rows.append(values)

    if not processed_rows:
        return None

    height = len(processed_rows)
    width = min_width
    image = Image.new("L", (width, height))
    flat_data = [pixel for row in processed_rows for pixel in row]
    image.putdata(flat_data)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
