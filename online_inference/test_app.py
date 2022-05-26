import json

import pytest
from fastapi.testclient import TestClient

from app import app, load_model
from ml_project.enities import InputData, PredictResponse


@pytest.fixture(scope="session", autouse=True)
def get_model():
    load_model()


@pytest.fixture()
def get_test_request_data():
    data = [
        InputData(
            id=8,
            age=100,
            sex=1,
            cp=3,
            trestbps=210,
            chol=600,
            fbs=1,
            restecg=2,
            thalach=210,
            exang=1,
            oldpeak=7,
            slope=2,
            ca=3,
            thal=2,
        ),
        InputData(
            id=88,
            age=18,
            sex=0,
            cp=0,
            trestbps=90,
            chol=110,
            fbs=0,
            restecg=0,
            thalach=65,
            exang=0,
            oldpeak=0,
            slope=0,
            ca=0,
            thal=0,
        ),
    ]
    return data


def test_can_get_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert 200 == response.status_code


def test_can_get_status_endpoint():
    with TestClient(app) as client:
        expected_status = 200
        response = client.get("/health")
        assert expected_status == response.status_code


def test_predict_endpoint_works_correctly(get_test_request_data):
    expected_status = 200
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            data=json.dumps([item.__dict__ for item in get_test_request_data]),
        )
        responce_content = response.json()
        assert expected_status == response.status_code
        assert len(responce_content) == len(get_test_request_data)
        for idx, item in enumerate(get_test_request_data):
            assert responce_content[idx]["id"] == item.__getattribute__("id")
            assert (
                responce_content[idx]["target"] >= 0
                and responce_content[idx]["target"] <= 1
            )


def test_predict_endpoint_corrupted_data_type(get_test_request_data):
    with TestClient(app) as client:
        corrupted_data = get_test_request_data[0]
        corrupted_data.sex = "MALE"
        response = client.post("/predict", data=json.dumps([corrupted_data.__dict__]))
        expected_text = "value is not a valid integer"
        expected_status = 422
        assert expected_status == response.status_code
        assert expected_text == response.json()["detail"][0]["msg"]


def test_predict_endpoint_extreme_data_value(get_test_request_data):
    with TestClient(app) as client:
        corrupted_data = get_test_request_data[0]
        corrupted_data.age = -1
        response = client.post("/predict", data=json.dumps([corrupted_data.__dict__]))
        expected_status = 400
        expected_text = "value -1 in 'age' is out of [18, 100] interval"
        assert expected_status == response.status_code
        assert expected_text == response.json()["detail"]
