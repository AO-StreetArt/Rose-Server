from __future__ import annotations

import base64
import binascii
import json
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, g

from bedrock import BedrockAgentClient
from bedrock_action_helper import (
    _as_list,
    _base_key_name,
    _derive_url_key,
    _get_param,
    _humanize_key,
    _infer_function_from_path,
    _normalize_parameters,
)
from sagemaker import SageMakerRuntimeClient
from image_utils import (
    extract_mask_bytes,
    fetch_public_image_bytes,
    fetch_s3_image_bytes,
    is_s3_url,
    _prepare_image_reference,
)
from s3_storage import S3StorageClient, S3StorageError

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from flask.typing import ResponseReturnValue

logger = logging.getLogger(__name__)


def handle_chat_request(app: Flask) -> "ResponseReturnValue":
    payload = request.get_json(silent=True) or {}
    message = payload.get("message")
    if not message:
        return jsonify({"error": "message is required"}), HTTPStatus.BAD_REQUEST

    session_id: Optional[str] = payload.get("sessionId")
    enable_trace = bool(payload.get("enableTrace"))

    client: BedrockAgentClient = app.extensions["bedrock_client"]
    sagemaker_client: SageMakerRuntimeClient = app.extensions["sagemaker_client"]
    storage: Optional[S3StorageClient] = app.extensions.get("s3_storage")
    g.sagemaker_client = sagemaker_client
    if not storage:
        logger.error("S3 storage client is not configured on the application.")
        return jsonify({"error": "file storage unavailable"}), HTTPStatus.INTERNAL_SERVER_ERROR
    try:
        image_attachment = _extract_image_attachment(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST
    try:
        image_reference_uri = _prepare_image_reference(image_attachment, storage)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST
    except S3StorageError:
        logger.exception("Failed to persist image attachment for chat request.")
        return jsonify({"error": "image processing unavailable"}), HTTPStatus.BAD_GATEWAY

    aggregated_messages: List[Dict[str, Any]] = []
    trace_payload: Optional[Dict[str, Any]] = None
    session_state: Optional[Dict[str, Any]] = None
    user_input: Optional[str] = message
    if image_reference_uri:
        user_input = _append_image_reference_to_input(message, image_reference_uri)
    current_session_id = session_id
    artifact_summaries: List[Dict[str, str]] = []

    while True:
        try:
            result = client.invoke(
                user_input=user_input,
                session_id=current_session_id,
                enable_trace=enable_trace,
                session_state=session_state,
            )
        except RuntimeError:
            logger.exception("Bedrock invocation error")
            return jsonify({"error": "chat service unavailable"}), HTTPStatus.BAD_GATEWAY

        session_state = None
        current_session_id = result.get("sessionId", current_session_id)

        messages = result.get("messages") or []
        if messages:
            aggregated_messages = messages

        if enable_trace and result.get("trace"):
            trace_payload = result["trace"]

        return_control_events = result.get("returnControl") or []
        if not return_control_events:
            break

        try:
            session_state, invocation_artifacts = _build_session_state_from_return_control(
                return_control_events,
                sagemaker_client,
                storage,
            )
        except RuntimeError:
            logger.exception("Return control invocation processing failed")
            return (
                jsonify({"error": "unable to process return control request"}),
                HTTPStatus.BAD_GATEWAY,
            )
        artifact_summaries.extend(invocation_artifacts)
        user_input = None

    if artifact_summaries:
        aggregated_messages = list(aggregated_messages)
        aggregated_messages.append(
            {
                "role": "assistant",
                "content": _format_artifact_summary(artifact_summaries),
            }
        )

    response_body: Dict[str, Any] = {
        "sessionId": current_session_id,
        "messages": aggregated_messages,
        "trace": trace_payload,
    }
    if artifact_summaries:
        response_body["artifacts"] = artifact_summaries
    return jsonify(response_body)


def _build_session_state_from_return_control(
    events: List[Dict[str, Any]],
    sagemaker_client: SageMakerRuntimeClient,
    storage: S3StorageClient,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    if not events:
        raise RuntimeError("Return control events required.")

    invocation_id: Optional[str] = None
    invocation_results: List[Dict[str, Any]] = []
    artifact_summaries: List[Dict[str, str]] = []

    for event in events:
        event_invocation_id = event.get("invocationId")
        if not event_invocation_id:
            raise RuntimeError("Return control event missing invocationId.")
        invocation_id = invocation_id or event_invocation_id

        invocation_inputs = event.get("invocationInputs") or []
        for invocation_input in invocation_inputs:
            function_input = invocation_input.get("functionInvocationInput")
            api_input = invocation_input.get("apiInvocationInput")
            if function_input:
                handler = _handle_function_invocation
                handler_input = function_input
            elif api_input:
                handler = _handle_api_invocation
                handler_input = api_input
            else:
                logger.debug("Skipping invocation input without function or api data: %s", invocation_input)
                continue

            result_entry, function_artifacts = handler(
                handler_input,
                sagemaker_client,
                storage,
            )
            if result_entry:
                invocation_results.append(result_entry)
            artifact_summaries.extend(function_artifacts)

    if not invocation_results or not invocation_id:
        raise RuntimeError("No invocation results were produced.")

    session_state = {
        "invocationId": invocation_id,
        "returnControlInvocationResults": invocation_results,
    }
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Prepared session state for return control: %s", session_state)

    return session_state, artifact_summaries


def _handle_function_invocation(
    function_input: Dict[str, Any],
    sagemaker_client: SageMakerRuntimeClient,
    storage: S3StorageClient,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    action_group = function_input.get("actionGroup") or ""
    function_name = function_input.get("function") or ""
    parameters = _normalize_parameters(function_input.get("parameters") or [])

    try:
        if action_group == "DepthMask":
            response_body, artifacts = _invoke_depth_mask(
                parameters,
                sagemaker_client,
                storage,
            )
        elif action_group == "SegmentationMask":
            response_body, artifacts = _invoke_segmentation_mask(
                parameters,
                sagemaker_client,
                storage,
            )
        else:
            raise RuntimeError(f"Unsupported action group: {action_group}")

        return {
            "functionResult": {
                "actionGroup": action_group,
                "function": function_name,
                "confirmationState": "CONFIRM",
                "responseBody": response_body,
            }
        }, artifacts
    except Exception:  # noqa: BLE001 - bubble a failure state to the agent
        logger.exception("Return control handler for action group %s failed", action_group)
        failure_message = (
            f"Unable to execute {action_group or 'the requested'} action at this time."
        )
        return {
            "functionResult": {
                "actionGroup": action_group,
                "function": function_name,
                "confirmationState": "CONFIRM",
                "responseBody": _build_failure_body(failure_message),
                "responseState": "FAILURE",
            }
        }, []


def _handle_api_invocation(
    api_input: Dict[str, Any],
    sagemaker_client: SageMakerRuntimeClient,
    storage: S3StorageClient,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    action_group = api_input.get("actionGroup") or ""
    api_path = api_input.get("apiPath")
    http_method = api_input.get("httpMethod") or "POST"
    inferred_function = api_input.get("function") or _infer_function_from_path(api_path, action_group)

    request_body = api_input.get("requestBody") or {}
    content = request_body.get("content") or {}
    body_schema = content.get("application/json") or {}
    properties = body_schema.get("properties") or []

    parameters: List[Dict[str, Any]] = []
    for prop in properties:
        parameters.append(
            {
                "name": prop.get("name"),
                "type": prop.get("type"),
                "value": prop.get("value"),
            }
        )

    function_input = {
        "actionGroup": action_group,
        "function": inferred_function,
        "parameters": parameters,
    }
    result_entry, artifacts = _handle_function_invocation(
        function_input,
        sagemaker_client,
        storage,
    )

    if not result_entry:
        return None, artifacts

    function_result = result_entry.get("functionResult", {})
    response_body = function_result.get("responseBody") or {}

    api_result = {
        "apiResult": {
            "actionGroup": action_group,
            "httpMethod": http_method,
            "apiPath": api_path or "/",
            "httpStatusCode": 200,
            "responseBody": response_body,
        }
    }

    return api_result, artifacts


def _invoke_depth_mask(
    parameters: Dict[str, Any],
    sagemaker_client: SageMakerRuntimeClient,
    storage: S3StorageClient,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    image_url = (
        _get_param(parameters, "ImageURL")
        or _get_param(parameters, "imageUrl")
        or _get_param(parameters, "imageURL")
    )
    if not image_url:
        raise ValueError("ImageURL parameter is required for DepthMask.")

    image_url_str = str(image_url)
    if is_s3_url(image_url_str):
        image_bytes = fetch_s3_image_bytes(image_url_str, storage)
    else:
        image_bytes = fetch_public_image_bytes(image_url_str)
    estimator = _get_param(parameters, "estimator") or "dpt"
    output_format = _get_param(parameters, "outputFormat") or "png"

    result = sagemaker_client.invoke_depth_estimator(
        image_bytes=image_bytes,
        estimator=str(estimator),
        output_format=str(output_format),
    )
    logger.debug("Depth endpoint response summary: %s", _summarize_payload(result))
    artifacts, url_map = _store_generated_images(
        result,
        storage,
        artifact_label="depth-map",
    )
    depth_url = _select_artifact_url(url_map, artifacts)
    if depth_url is None:
        logger.error(
            "Unable to derive depth map URL. Artifacts=%s url_map=%s result_summary=%s",
            artifacts,
            url_map,
            _summarize_payload(result),
        )
        raise RuntimeError("Depth map URL was not generated.")
    response_payload: Dict[str, Any] = {"depthMapUrl": depth_url}
    return _format_response_body(response_payload), artifacts


def _invoke_segmentation_mask(
    parameters: Dict[str, Any],
    sagemaker_client: SageMakerRuntimeClient,
    storage: S3StorageClient,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    image_url = (
        _get_param(parameters, "ImageURL")
        or _get_param(parameters, "imageUrl")
        or _get_param(parameters, "imageURL")
    )
    if not image_url:
        raise ValueError("ImageURL parameter is required for SegmentationMask.")

    image_url_str = str(image_url)
    if is_s3_url(image_url_str):
        image_bytes = fetch_s3_image_bytes(image_url_str, storage)
    else:
        image_bytes = fetch_public_image_bytes(image_url_str)
    content_type = _get_param(parameters, "contentType") or "application/x-image"
    accept = _get_param(parameters, "accept") or "application/json;verbose"

    result = sagemaker_client.invoke_segmentation(
        image_bytes=image_bytes,
        content_type=str(content_type),
        accept=str(accept),
    )
    logger.debug("Segmentation endpoint response summary: %s", _summarize_payload(result))
    detected_objects = _as_list(_get_param(parameters, "DetectedObjects"))
    if detected_objects is not None:
        result = dict(result)
        result["detectedObjects"] = detected_objects
    if isinstance(result, dict):
        mask_bytes = extract_mask_bytes(result)
    else:
        mask_bytes = None
    artifacts, url_map = _store_generated_images_from_bytes(
        result=result if isinstance(result, dict) else None,
        mask_bytes=mask_bytes,
        storage=storage,
        artifact_label="segmentation-mask",
    )
    mask_url = _select_artifact_url(url_map, artifacts)
    if mask_url is None:
        logger.error(
            "Unable to derive segmentation mask URL. Artifacts=%s url_map=%s result_summary=%s",
            artifacts,
            url_map,
            _summarize_payload(result),
        )
        raise RuntimeError("Segmentation mask URL was not generated.")
    response_payload: Dict[str, Any] = {"maskUrl": mask_url}
    if detected_objects:
        response_payload["detectedObjects"] = detected_objects
    return _format_response_body(response_payload), artifacts


def _format_response_body(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        json_body = json.dumps(payload)
    except (TypeError, ValueError):
        json_body = "{}"
    return {"application/json": {"body": json_body}}


def _store_generated_images(
    result: Dict[str, Any],
    storage: S3StorageClient,
    *,
    artifact_label: str,
) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    artifacts: List[Dict[str, str]] = []
    url_map: Dict[str, str] = {}
    for key in list(result.keys()):
        value = result[key]
        if not isinstance(value, str):
            continue
        if not key.lower().endswith("_base64"):
            continue
        try:
            image_bytes = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            logger.warning("Skipping invalid base64 payload for key %s", key)
            continue

        try:
            stored = storage.upload_artifact_bytes(
                image_bytes,
                artifact_label=artifact_label,
                key_hint=key,
            )
        except S3StorageError:
            logger.exception("Failed to upload generated artifact for key %s", key)
            continue

        del result[key]
        url_key = _derive_url_key(key)
        result[url_key] = stored.url
        base_key = _base_key_name(key)
        url_map[base_key] = stored.url

        artifacts.append(
            {
                "label": _humanize_key(key, default_label=artifact_label),
                "url": stored.url,
            }
        )

    return artifacts, url_map


def _store_generated_images_from_bytes(
    *,
    result: Optional[Dict[str, Any]],
    mask_bytes: Optional[bytes],
    storage: S3StorageClient,
    artifact_label: str,
) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    artifacts: List[Dict[str, str]] = []
    url_map: Dict[str, str] = {}

    if mask_bytes:
        try:
            stored_mask = storage.upload_artifact_bytes(
                mask_bytes,
                artifact_label=artifact_label,
                key_hint=artifact_label,
            )
        except S3StorageError:  # noqa: BLE001
            logger.exception("Failed to upload generated %s bytes", artifact_label)
        else:
            artifacts.append(
                {
                    "label": artifact_label.replace("-", " ").title(),
                    "url": stored_mask.url,
                }
            )
            url_map[artifact_label] = stored_mask.url

    if isinstance(result, dict):
        extra_artifacts, extra_map = _store_generated_images(
            result,
            storage,
            artifact_label=artifact_label,
        )
        artifacts.extend(extra_artifacts)
        url_map.update(extra_map)

    return artifacts, url_map


def _select_artifact_url(url_map: Dict[str, str], artifacts: List[Dict[str, str]]) -> Optional[str]:
    for preferred in ("mask", "segmentation_mask", "segmentation", "depth_map", "depth"):
        if preferred in url_map:
            return url_map[preferred]

    for artifact in artifacts:
        url = artifact.get("url")
        if url:
            return url
    return None


def _extract_url_from_result(result: Dict[str, Any]) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    for key, value in result.items():
        if not isinstance(value, str):
            continue
        if key.lower().endswith("url"):
            return value
    return None


def _summarize_payload(payload: Dict[str, Any], *, max_items: int = 10) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}

    summary: Dict[str, Any] = {}
    for index, (key, value) in enumerate(payload.items()):
        if index >= max_items:
            summary["..."] = f"+{len(payload) - max_items} more"
            break
        summary[key] = _summarize_value(value)
    return summary


def _summarize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            "type": "dict",
            "size": len(value),
            "preview": _summarize_payload(value, max_items=5),
        }
    if isinstance(value, list):
        preview = [_summarize_value(v) for v in value[:5]]
        if len(value) > 5:
            preview.append(f"... +{len(value) - 5} more")
        return {"type": "list", "size": len(value), "preview": preview}
    if isinstance(value, str):
        truncated = value[:60] + ("..." if len(value) > 60 else "")
        return {"type": "str", "length": len(value), "preview": truncated}
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "length": len(value)}
    return {"type": type(value).__name__, "value": value}


def _format_artifact_summary(artifacts: List[Dict[str, str]]) -> str:
    lines = ["Generated resources:"]
    for artifact in artifacts:
        url = artifact.get("url")
        if not url:
            continue
        label = artifact.get("label") or "Resource"
        lines.append(f"- {label}: {url}")

    if len(lines) == 1:
        return "Generated resources are available."
    return "\n".join(lines)


def _build_failure_body(message: str) -> Dict[str, Any]:
    safe_message = message if message else "Invocation failed."
    return {"text/plain": {"body": safe_message}}


def _append_image_reference_to_input(message: str, image_uri: str) -> str:
    """Embed the uploaded image reference into the agent input text."""
    trimmed = (message or "").rstrip()
    prefix = f"{trimmed}\n\n" if trimmed else ""
    return f"{prefix}ImageURL: {image_uri}"


def _extract_image_attachment(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    image_payload = payload.get("image")
    base64_data: Optional[str] = None
    media_type: Optional[str] = None
    s3_url: Optional[str] = None

    if image_payload is None:
        base64_data = payload.get("imageBase64") or payload.get("image_base64")
        media_type = payload.get("imageMediaType") or payload.get("imageMimeType")
        s3_url = payload.get("imageS3Url") or payload.get("image_s3_url")
    elif isinstance(image_payload, str):
        base64_data = image_payload
    elif isinstance(image_payload, dict):
        base64_data = image_payload.get("data") or image_payload.get("base64")
        media_type = (
            image_payload.get("mediaType")
            or image_payload.get("mimeType")
            or image_payload.get("format")
        )
        s3_url = image_payload.get("s3Url") or image_payload.get("s3URI") or image_payload.get("s3_uri")
    else:
        raise ValueError("image must be a string or object.")

    if base64_data is not None:
        trimmed = base64_data.strip()
        if trimmed.lower().startswith("s3://"):
            s3_url = trimmed
            base64_data = None
    if s3_url:
        s3_url = s3_url.strip()
        if not s3_url.lower().startswith("s3://"):
            raise ValueError("image S3 URL must start with s3://.")
        if not media_type:
            media_type = "image/png"
        media_type = media_type.strip() if media_type else "image/png"
        if not media_type:
            media_type = "image/png"
        if "/" not in media_type:
            media_type = media_type.lstrip(".")
            media_type = f"image/{media_type or 'png'}"

        return {
            "name": "input-image",
            "contentType": media_type,
            "source": {
                "s3Location": {
                    "uri": s3_url,
                }
            },
        }

    if base64_data is None:
        return None
    if not isinstance(base64_data, str):
        raise ValueError("image data must be provided as a base64-encoded string.")

    try:
        image_bytes = base64.b64decode(base64_data, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("image must be valid base64 data.")

    if not image_bytes:
        raise ValueError("image data cannot be empty.")

    if not media_type:
        media_type = "image/png"
    media_type = media_type.strip()
    if not media_type:
        media_type = "image/png"
    if "/" not in media_type:
        media_type = media_type.lstrip(".")
        media_type = f"image/{media_type or 'png'}"

    return {
        "name": "input-image",
        "contentType": media_type,
        "data": image_bytes,
    }
