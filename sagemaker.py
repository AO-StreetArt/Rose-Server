from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, Optional

from botocore.exceptions import BotoCoreError, ClientError
from boto3.session import Session

logger = logging.getLogger(__name__)


class SageMakerRuntimeClient:
    """Convenience wrapper around the SageMaker runtime client."""

    def __init__(
        self,
        *,
        region: str,
        depth_endpoint: Optional[str] = None,
        segmentation_endpoint: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> None:
        self.depth_endpoint = depth_endpoint
        self.segmentation_endpoint = segmentation_endpoint
        self._session = session or Session()
        self._client = self._session.client("sagemaker-runtime", region_name=region)

    def invoke_depth_estimator(
        self,
        *,
        image_bytes: bytes,
        estimator: str = "dpt",
        output_format: str = "png",
    ) -> Dict[str, Any]:
        """Invoke the configured depth estimation endpoint."""
        if not self.depth_endpoint:
            raise RuntimeError("Depth estimation endpoint is not configured.")
        if not image_bytes:
            raise ValueError("image_bytes must be provided.")

        payload = {
            "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
            "estimator": estimator,
            "output_format": output_format,
        }

        try:
            response = self._client.invoke_endpoint(
                EndpointName=self.depth_endpoint,
                ContentType="application/json",
                Body=json.dumps(payload).encode("utf-8"),
            )
        except (BotoCoreError, ClientError) as exc:
            logger.exception("Failed to invoke SageMaker depth estimation endpoint.")
            raise RuntimeError("Failed to invoke SageMaker depth estimation endpoint.") from exc

        return _parse_json_body(response)

    def invoke_segmentation(
        self,
        *,
        image_bytes: bytes,
        content_type: str = "application/x-image",
        accept: str = "application/json;verbose",
    ) -> Dict[str, Any]:
        """Invoke the configured image segmentation endpoint."""
        if not self.segmentation_endpoint:
            raise RuntimeError("Segmentation endpoint is not configured.")
        if not image_bytes:
            raise ValueError("image_bytes must be provided.")

        try:
            response = self._client.invoke_endpoint(
                EndpointName=self.segmentation_endpoint,
                ContentType=content_type,
                Body=image_bytes,
                Accept=accept,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.exception("Failed to invoke SageMaker segmentation endpoint.")
            raise RuntimeError("Failed to invoke SageMaker segmentation endpoint.") from exc

        return _parse_json_body(response, default={})


def _parse_json_body(
    response: Dict[str, Any],
    *,
    default: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read and deserialize a JSON StreamingBody response."""
    body = response.get("Body")
    if not body:
        if default is not None:
            return default
        raise RuntimeError("SageMaker response did not include a body.")

    try:
        payload = body.read()
    except AttributeError as exc:
        raise RuntimeError("Unexpected body type in SageMaker response.") from exc
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()

    if not payload:
        if default is not None:
            return default
        raise RuntimeError("SageMaker response body was empty.")

    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.exception("Unable to decode SageMaker JSON response.")
        raise RuntimeError("Failed to decode response from SageMaker endpoint.") from exc
