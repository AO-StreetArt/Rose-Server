from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from boto3.session import Session
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class BedrockAgentClient:
    """Proxy client for AWS Bedrock Agent invocations."""

    def __init__(
        self,
        region: str,
        agent_id: str,
        agent_alias_id: str,
        *,
        session: Optional[Session] = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self._session = session or Session()
        self._client = self._session.client("bedrock-agent-runtime", region_name=region)

    @property
    def session(self) -> Session:
        return self._session

    def invoke(
        self,
        *,
        user_input: Optional[str] = None,
        session_id: Optional[str] = None,
        enable_trace: bool = False,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        has_text = True
        if isinstance(user_input, str):
            has_text = bool(user_input.strip())
        elif user_input is None:
            has_text = False

        if not has_text and session_state is None:
            raise ValueError("Either user_input or session_state must be provided.")

        request_session_id = session_id or str(uuid4())
        request_payload: Dict[str, Any] = {
            "agentId": self.agent_id,
            "agentAliasId": self.agent_alias_id,
            "sessionId": request_session_id,
        }
        if enable_trace:
            request_payload["enableTrace"] = True
        if has_text and user_input is not None:
            request_payload["inputText"] = user_input
        if session_state:
            request_payload["sessionState"] = session_state

        logger.info("Invoking Bedrock agent with payload: %s", _mask_sensitive_fields(request_payload))

        try:
            response = self._client.invoke_agent(
                **request_payload,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.exception("Bedrock agent invocation failed.")
            raise RuntimeError("Failed to invoke Bedrock agent") from exc

        completion_events: List[Dict[str, Any]] = list(response.get("completion") or [])
        completion_text = _extract_completion_text(completion_events)
        return_control_events = _extract_return_control_events(completion_events)
        trace_events = _extract_trace_events(completion_events)
        payload: Dict[str, Any] = {
            "sessionId": response.get("sessionId", request_session_id),
            "messages": [{"role": "assistant", "content": completion_text}] if completion_text else [],
        }
        if enable_trace:
            payload["trace"] = {"events": trace_events} if trace_events else None
        if return_control_events:
            payload["returnControl"] = return_control_events

        return payload


def _extract_completion_text(events: List[Dict[str, Any]]) -> str:
    """Aggregate completion text from the streaming Bedrock response."""
    chunks = []
    for event in events:
        chunk = event.get("chunk")
        if not chunk:
            continue
        data = chunk.get("bytes")
        if not data:
            continue
        if isinstance(data, (bytes, bytearray)):
            chunks.append(data.decode("utf-8"))
        else:
            chunks.append(str(data))
    return "".join(chunks)


def _extract_trace_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    for event in events:
        trace = event.get("trace")
        if trace:
            traces.append(trace)
    return traces


def _extract_return_control_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return_control_events: List[Dict[str, Any]] = []
    for event in events:
        payload = event.get("returnControl")
        if payload:
            return_control_events.append(payload)
    return return_control_events


def _mask_sensitive_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(payload)
    if "inputText" in sanitized:
        sanitized["inputText"] = "***redacted***"
    if "inputAttachments" in sanitized:
        sanitized["inputAttachments"] = "***redacted***"
    return sanitized
