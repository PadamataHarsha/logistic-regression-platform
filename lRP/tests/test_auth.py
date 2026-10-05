import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import csrf_token, sign_in


@pytest.mark.parametrize(
    ("account_key", "role"),
    [("admin", "Admin"), ("analyst", "Analyst"), ("viewer", "Viewer")],
)
def test_demo_accounts_can_sign_in(client, account_key, role):
    sign_in(client, account_key)

    response = client.get("/profile")
    assert response.status_code == 200
    assert role in response.text


def test_session_cookie_is_http_only_and_same_site(client):
    response = sign_in(client, "admin")
    cookie = response.headers["set-cookie"].lower()

    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "max-age" not in cookie


def test_remember_me_sets_a_thirty_day_cookie(client):
    from app.config import ACCOUNTS

    account = ACCOUNTS["admin"]
    response = client.post(
        "/login",
        data={
            "csrf_token": csrf_token(client),
            "username": account["username"],
            "password": account["password"],
            "remember_me": "yes",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "max-age=2592000" in response.headers["set-cookie"].lower()


def test_production_requires_non_demo_credentials_and_secure_cookie():
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "APP_ENV": "production",
            "SESSION_SECRET_KEY": "validation-secret-value-long-enough-123456",
            "ADMIN_USERNAME": "production_admin",
            "ADMIN_PASSWORD": "AdminDemo123!",
            "ANALYST_USERNAME": "production_analyst",
            "ANALYST_PASSWORD": "AnalystDemo123!",
            "VIEWER_USERNAME": "production_viewer",
            "VIEWER_PASSWORD": "ViewerDemo123!",
        }
    )
    rejected = subprocess.run(
        [sys.executable, "-c", "from main import app"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "Replace demo passwords" in rejected.stderr

    environment.update(
        {
            "ADMIN_PASSWORD": "Production-Admin-Secret-2026!",
            "ANALYST_PASSWORD": "Production-Analyst-Secret-2026!",
            "VIEWER_PASSWORD": "Production-Viewer-Secret-2026!",
        }
    )
    accepted = subprocess.run(
        [
            sys.executable,
            "-c",
            "from main import app; middleware = next(item for item in app.user_middleware "
            "if item.cls.__name__ == 'SessionMiddleware'); "
            "assert middleware.kwargs['https_only'] is True; print('secure-cookie-ok')",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert "secure-cookie-ok" in accepted.stdout


def test_login_html_does_not_contain_demo_passwords(client):
    response = client.get("/login")

    assert response.status_code == 200
    assert "AdminDemo123!" not in response.text
    assert "AnalystDemo123!" not in response.text
    assert "ViewerDemo123!" not in response.text


def test_admin_pages_render(client):
    sign_in(client, "admin")

    for path in ("/dashboard", "/prediction", "/evaluation", "/performance", "/analytics", "/insights", "/about", "/profile"):
        response = client.get(path)
        assert response.status_code == 200, path


def test_viewer_can_open_allowed_pages(client):
    sign_in(client, "viewer")

    for path in ("/dashboard", "/prediction", "/about", "/profile"):
        response = client.get(path)
        assert response.status_code == 200, path

    for path in ("/evaluation", "/performance", "/analytics", "/insights"):
        response = client.get(path)
        assert response.status_code == 403, path


def test_invalid_password_is_rejected(client):
    from app.config import ACCOUNTS

    response = client.post(
        "/login",
        data={
            "csrf_token": csrf_token(client),
            "username": ACCOUNTS["admin"]["username"],
            "password": "not-the-password",
        },
        follow_redirects=False,
    )

    assert response.status_code == 401
    assert "was not recognized" in response.text


def test_protected_page_redirects_anonymous_user(client):
    response = client.get("/dashboard", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_viewer_is_denied_analyst_page(client):
    sign_in(client, "viewer")

    response = client.get("/performance")

    assert response.status_code == 403
    assert "not available to your role" in response.text


def test_logout_clears_the_session(client):
    sign_in(client, "analyst")
    response = client.post(
        "/logout",
        data={"csrf_token": csrf_token(client)},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    protected = client.get("/dashboard", follow_redirects=False)
    assert protected.status_code == 303