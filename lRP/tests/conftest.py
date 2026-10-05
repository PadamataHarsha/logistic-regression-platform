import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def csrf_token(client: TestClient) -> str:
    response = client.get("/login")
    marker = 'name="csrf_token" value="'
    start = response.text.index(marker) + len(marker)
    return response.text[start : response.text.index('"', start)]


def sign_in(client: TestClient, account_key: str) -> None:
    from app.config import ACCOUNTS

    account = ACCOUNTS[account_key]
    response = client.post(
        "/login",
        data={
            "csrf_token": csrf_token(client),
            "username": account["username"],
            "password": account["password"],
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard"
    return response