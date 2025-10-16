from __future__ import annotations

from unittest import mock

import pytest

from app import create_app
from config import Settings


@pytest.fixture
def fake_settings(tmp_path):
    react_build = tmp_path / "build"
    react_build.mkdir()
    return Settings(
        react_build_path=react_build,
        auth0_domain="example.auth0.com",
        auth0_audience="https://api.example.com",
        auth0_client_id="client-id",
        bedrock_agent_id="agent-id",
        bedrock_agent_alias_id="alias-id",
        bedrock_region="us-east-1",
        sagemaker_region="us-east-1",
        sagemaker_depth_endpoint="depth-endpoint",
        sagemaker_segmentation_endpoint="seg-endpoint",
        require_auth_for_health=False,
        require_auth_for_ui=False,
        s3_bucket_name="bucket",
        s3_region_name="us-east-1",
        s3_object_prefix="uploads/",
    )


def test_create_app_wires_shared_aws_session(fake_settings):
    mock_session = mock.Mock()
    mock_session.client.return_value = mock.Mock()

    with mock.patch("app.Session", return_value=mock_session) as session_cls, \
        mock.patch("app.BedrockAgentClient") as bedrock_cls, \
        mock.patch("app.SageMakerRuntimeClient") as sagemaker_cls:

        flask_app = create_app(settings=fake_settings)

    session_cls.assert_called_once_with()
    bedrock_cls.assert_called_once_with(
        region="us-east-1",
        agent_id="agent-id",
        agent_alias_id="alias-id",
        session=mock_session,
    )
    sagemaker_cls.assert_called_once_with(
        region="us-east-1",
        depth_endpoint="depth-endpoint",
        segmentation_endpoint="seg-endpoint",
        session=mock_session,
    )
    mock_session.client.assert_called_once_with("s3", region_name="us-east-1")
    assert flask_app.extensions["aws_session"] is mock_session
    assert flask_app.extensions["bedrock_client"] is bedrock_cls.return_value
    assert flask_app.extensions["sagemaker_client"] is sagemaker_cls.return_value
