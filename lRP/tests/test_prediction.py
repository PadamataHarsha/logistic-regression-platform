from tests.conftest import csrf_token, sign_in


def test_valid_prediction_returns_class_probabilities(client):
    sign_in(client, "analyst")
    token = csrf_token(client)

    response = client.post(
        "/prediction",
        data={
            "csrf_token": token,
            "features": ["68.95", "35", "61833.9", "256.09", "0"],
        },
    )

    assert response.status_code == 200
    assert "Probability: unlikely to click" in response.text
    assert "Probability: likely to click" in response.text
    assert "CONFIDENCE" in response.text
    assert "Educational estimate" in response.text


def test_invalid_prediction_is_reported_and_values_are_preserved(client):
    sign_in(client, "analyst")

    response = client.post(
        "/prediction",
        data={
            "csrf_token": csrf_token(client),
            "features": ["68.95", "999", "61833.9", "256.09", "0"],
        },
    )

    assert response.status_code == 422
    assert "based on the training dataset" in response.text
    assert 'value="999"' in response.text


def test_missing_prediction_values_do_not_crash(client):
    sign_in(client, "analyst")

    response = client.post(
        "/prediction",
        data={"csrf_token": csrf_token(client), "features": ["68.95", "35"]},
    )

    assert response.status_code == 422
    assert "Enter all 5 values" in response.text


def test_json_prediction_returns_real_model_result(client):
    sign_in(client, "analyst")

    response = client.post(
        "/api/predict",
        headers={"X-CSRF-Token": csrf_token(client)},
        json={"features": [68.95, 35, 61833.9, 256.09, 0]},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["predicted_class"] in (0, 1)
    assert result["label"] in ("Likely to Click", "Unlikely to Click")
    assert 0.0 <= result["probability_class_0"] <= 1.0
    assert 0.0 <= result["probability_class_1"] <= 1.0
    assert abs(result["probability_class_0"] + result["probability_class_1"] - 1) < 1e-9


def test_json_prediction_requires_csrf_token(client):
    sign_in(client, "analyst")

    response = client.post(
        "/api/predict",
        json={"features": [68.95, 35, 61833.9, 256.09, 0]},
    )

    assert response.status_code == 403