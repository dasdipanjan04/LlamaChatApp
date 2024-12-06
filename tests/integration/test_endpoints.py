import re

import pytest
from src.api import app
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


@pytest.mark.usefixtures("setup_model_manager")
@pytest.mark.asyncio
async def test_query_endpoint_success():
    with TestClient(app) as client:
        response = client.post(
            "/query", json={"queries": ["What is the capital city of Germany?"]}
        )
        print(f"I have response: {response.status_code}")
        assert response.status_code == 200
        result = response.json()
        assert re.search(
            r"\bBerlin\b", result["0"], re.IGNORECASE
        ), "The expected response should have Berlin in it."


@pytest.mark.usefixtures("setup_model_manager")
@pytest.mark.asyncio
async def test_query_endpoint_abuse_detection():
    with TestClient(app) as client:
        response = client.post(
            "/query", json={"queries": ["Why are you so stupid and dumb?"]}
        )
        assert response.status_code == 200
        result = response.json()
        assert (
            result["0"]
            == "Please refrain from such language.\nLet us have a constructive conversation."
        ), "Abuse detection should warn the user."
