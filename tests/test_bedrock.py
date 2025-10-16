from __future__ import annotations

from unittest import mock
from unittest.mock import ANY

import pytest
from botocore.exceptions import ClientError

from bedrock import BedrockAgentClient


def make_session():
    session = mock.Mock()
    client = mock.Mock()
    session.client.return_value = client
    return session, client


def test_invoke_success_aggregates_completion():
    session, client = make_session()
    client.invoke_agent.return_value = {
        "completion": [
            {"chunk": {"bytes": b"hello "}},
            {"chunk": {"bytes": b"world"}},
        ]
    }

    bedrock_client = BedrockAgentClient(
        region="us-east-1",
        agent_id="agent",
        agent_alias_id="alias",
        session=session,
    )

    result = bedrock_client.invoke(user_input="hi")

    session.client.assert_called_once_with("bedrock-agent-runtime", region_name="us-east-1")
    client.invoke_agent.assert_called_once_with(
        agentId="agent",
        agentAliasId="alias",
        sessionId=ANY,
        enableTrace=False,
        inputText="hi",
    )
    assert result["messages"][0]["content"] == "hello world"
    assert "sessionId" in result


def test_invoke_failure_raises_runtime_error():
    session, client = make_session()
    client.invoke_agent.side_effect = ClientError(
        error_response={"Error": {"Code": "BadRequestException", "Message": "nope"}},
        operation_name="InvokeAgent",
    )

    bedrock_client = BedrockAgentClient(
        region="us-east-1",
        agent_id="agent",
        agent_alias_id="alias",
        session=session,
    )

    with pytest.raises(RuntimeError):
        bedrock_client.invoke(user_input="hi")


def test_invoke_requires_user_input():
    session, _ = make_session()
    bedrock_client = BedrockAgentClient(
        region="us-east-1",
        agent_id="agent",
        agent_alias_id="alias",
        session=session,
    )

    with pytest.raises(ValueError):
        bedrock_client.invoke(user_input="")
