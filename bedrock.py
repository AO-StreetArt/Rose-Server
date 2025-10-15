from __future__ import annotations

import logging
from typing import Dict, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class BedrockAgentClient:
    """Proxy client for AWS Bedrock Agent invocations."""

    def __init__(self, region: str, agent_id: str, agent_alias_id: str) -> None:
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self._client = boto3.client("bedrock-agent-runtime", region_name=region)

    def invoke(
        self,
        *,
        user_input: str,
        session_id: Optional[str] = None,
        enable_trace: bool = False,
    ) -> Dict[str, object]:
        if not user_input:
            raise ValueError("user_input must be provided.")

        request_session_id = session_id or str(uuid4())
        try:
            response = self._client.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=request_session_id,
                enableTrace=enable_trace,
                inputText=user_input,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.exception("Bedrock agent invocation failed.")
            raise RuntimeError("Failed to invoke Bedrock agent") from exc

        completion_text = _extract_completion_text(response)
        payload = {
            "sessionId": request_session_id,
            "messages": [{"role": "assistant", "content": completion_text}],
        }

        if enable_trace:
            payload["trace"] = _extract_trace(response)

        return payload


def _extract_completion_text(response: Dict[str, object]) -> str:
    """Aggregate completion text from the streaming Bedrock response."""
    completion_events = response.get("completion") or []
    chunks = []
    for event in completion_events:
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


def _extract_trace(response: Dict[str, object]) -> Optional[Dict[str, object]]:
    traces = []
    for event in response.get("completion") or []:
        trace = event.get("trace")
        if trace:
            traces.append(trace)
    return {"events": traces} if traces else None
