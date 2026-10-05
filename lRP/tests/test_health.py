from pathlib import Path

from main import app, templates
from tests.conftest import sign_in


def test_public_health_reports_model_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["application"] == "Logistic Regression Intelligence Platform"
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_saved_pipeline_and_feature_order_are_preserved(client):
    service = client.app.state.model_service
    expected_features = [
        "Daily Time Spent on Site",
        "Age",
        "Area Income",
        "Daily Internet Usage",
        "Male",
    ]

    assert set(service.pipeline.named_steps) == {"scaler", "logistic_regression"}
    assert service.metadata["feature_order"] == expected_features
    assert service.metadata["target_name"] == "Clicked on Ad"
    assert service.metadata["dataset_records"] == 1000


def test_api_health_is_public(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_static_assets_load_without_login(client):
    css = client.get("/static/styles.css")
    javascript = client.get("/static/app.js")

    assert css.status_code == 200
    assert "--gradient-blue" in css.text
    assert javascript.status_code == 200
    assert "data-password-toggle" in javascript.text

    login = client.get("/login")
    assert 'href="/static/styles.css"' in login.text
    assert 'src="/static/app.js"' in login.text


def test_root_redirects_and_login_redirects_authenticated_user(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"

    sign_in(client, "admin")
    response = client.get("/login", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/dashboard"


def test_every_template_compiles(client):
    template_directory = Path(__file__).resolve().parents[1] / "templates"
    for template_path in template_directory.glob("*.html"):
        templates.env.get_template(template_path.name)


def test_unknown_route_uses_custom_404_page(client):
    response = client.get("/page-that-does-not-exist")

    assert response.status_code == 404
    assert "That page is not in this workspace" in response.text


def test_analytics_page_requires_analyst_role(client):
    sign_in(client, "analyst")
    response = client.get("/analytics")

    assert response.status_code == 200
    assert "Confusion matrix" in response.text


def test_evaluation_page_requires_analyst_role(client):
    sign_in(client, "analyst")
    response = client.get("/evaluation")

    assert response.status_code == 200
    assert "Confusion matrix" in response.text


def test_missing_model_degrades_health_without_crashing(client, monkeypatch):
    monkeypatch.setattr(app.state, "model_service", None)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["model_loaded"] is False

    sign_in(client, "admin")
    prediction = client.get("/prediction")
    assert prediction.status_code == 503
    assert "python train_model.py" in prediction.text