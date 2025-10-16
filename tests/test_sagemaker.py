from __future__ import annotations

import base64
import json
from unittest import mock
from unittest.mock import ANY

import pytest

from sagemaker import SageMakerRuntimeClient


def make_session():
    session = mock.Mock()
    client = mock.Mock()
    session.client.return_value = client
    return session, client


def make_streaming_body(payload: dict):
    body = mock.Mock()
    data = json.dumps(payload).encode("utf-8")
    body.read.return_value = data
    return body, data


def test_invoke_depth_estimator_decodes_response():
    session, client = make_session()
    body, _ = make_streaming_body({"foo": "bar"})
    client.invoke_endpoint.return_value = {"Body": body}

    runtime_client = SageMakerRuntimeClient(
        region="us-east-1",
        depth_endpoint="depth-endpoint",
        segmentation_endpoint="seg-endpoint",
        session=session,
    )

    result = runtime_client.invoke_depth_estimator(
        image_bytes=b"\x00\x01",
        estimator="zoedepth",
        output_format="array",
    )

    session.client.assert_called_once_with("sagemaker-runtime", region_name="us-east-1")
    client.invoke_endpoint.assert_called_once_with(
        EndpointName="depth-endpoint",
        ContentType="application/json",
        Body=ANY,
    )

    sent_body = client.invoke_endpoint.call_args.kwargs["Body"]
    payload = json.loads(sent_body.decode("utf-8"))
    assert base64.b64decode(payload["image_base64"]) == b"\x00\x01"
    assert payload["estimator"] == "zoedepth"
    assert payload["output_format"] == "array"

    body.read.assert_called_once()
    body.close.assert_called_once()
    assert result == {"foo": "bar"}


def test_invoke_segmentation_returns_default_when_body_missing():
    session, client = make_session()
    client.invoke_endpoint.return_value = {"Body": None}

    runtime_client = SageMakerRuntimeClient(
        region="us-east-1",
        depth_endpoint="depth-endpoint",
        segmentation_endpoint="seg-endpoint",
        session=session,
    )

    result = runtime_client.invoke_segmentation(image_bytes=b"abc")

    client.invoke_endpoint.assert_called_once_with(
        EndpointName="seg-endpoint",
        ContentType="application/x-image",
        Body=b"abc",
        Accept="application/json;verbose",
    )
    assert result == {}


def test_invoke_depth_requires_endpoint():
    session, _ = make_session()
    runtime_client = SageMakerRuntimeClient(region="us-east-1", session=session)

    with pytest.raises(RuntimeError):
        runtime_client.invoke_depth_estimator(image_bytes=b"x")
