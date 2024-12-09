import re

import pytest
from src.api import app
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_query_endpoint_success():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"queries": ["What is the capital city of Germany?"]},
            headers={"Accept": "text/event-stream"},
        )
        assert response.status_code == 200

        final_result = ""
        for line in response.iter_lines():
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            line = line.strip()
            if line == "END_OF_STREAM":
                break
            final_result += line

        assert re.search(
            r"\bBerlin\b", final_result, re.IGNORECASE
        ), "The expected response should have Berlin in it."


@pytest.mark.asyncio
async def test_query_endpoint_abuse_detection():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"queries": ["Why are you so stupid and dumb?"]},
            headers={"Accept": "text/event-stream"},
        )
        assert response.status_code == 200

        final_result = ""
        for line in response.iter_lines():
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            line = line.strip()
            if line == "END_OF_STREAM":
                break
            final_result += line

        assert (
            "Let us have a constructive conversation" in final_result
        ), "Abuse detection should warn the user."
