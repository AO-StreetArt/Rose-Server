"""Utility helpers for working with Bedrock action inputs and outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def _normalize_parameters(parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for parameter in parameters:
        name = parameter.get("name")
        if not name:
            continue
        value = parameter.get("value")
        param_type = (parameter.get("type") or "").lower()
        normalized[name] = _coerce_parameter_value(value, param_type)
    return normalized


def _coerce_parameter_value(value: Any, param_type: str) -> Any:
    if value is None:
        return None
    if param_type == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if param_type == "integer":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if param_type == "boolean":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() == "true"
    if param_type == "array":
        if isinstance(value, list):
            return value
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value
    return value


def _get_param(parameters: Dict[str, Any], name: str) -> Any:
    for key, value in parameters.items():
        if key.lower() == name.lower():
            return value
    return None


def _as_list(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                inner = stripped[1:-1].strip()
                if inner:
                    return [item.strip() for item in inner.split(",")]
                return []
        return [value]
    return [value]


def _base_key_name(key: str) -> str:
    lowered = key.lower()
    suffix = "_base64"
    if lowered.endswith(suffix):
        return key[: -len(suffix)]
    return key


def _derive_url_key(key: str) -> str:
    base = _base_key_name(key)
    return f"{base}_url"


def _humanize_key(key: str, *, default_label: str) -> str:
    base = _base_key_name(key).replace("_", " ").strip()
    if not base:
        base = default_label.replace("-", " ")
    return base.title()


def _infer_function_from_path(api_path: Optional[str], action_group: str) -> str:
    if api_path:
        cleaned = api_path.strip("/")
        if cleaned:
            parts = re.split(r"[-_/]+", cleaned)
            candidate = "".join(part.capitalize() for part in parts if part)
            if candidate:
                return candidate
    return action_group or "Function"

