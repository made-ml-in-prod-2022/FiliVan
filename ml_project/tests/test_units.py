import os
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


import pytest


from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig

import ml_project.enities
from ml_project.data import (
    download_data_from_gdrive,
    read_data,
    split_train_val_data,
)
from ml_project.features import make_features
from ml_project.features.build_features import (
    extract_target,
    build_transformer,
)
from ml_project.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
    create_pipeline,
)


TRAIN_DATA_SIZE = 100
CONFIG_FILES = [f.split(".")[0] for f in os.listdir("configs") if "yaml" in f]


@pytest.mark.parametrize("config_name", CONFIG_FILES)
def test_config_initialization(config_name: str) -> None:
    if config_name == "train_config":
        ml_project.enities.register_train_configs()
    with initialize(config_path="../configs"):
        params = compose(config_name=config_name)
        assert isinstance(params, DictConfig)


def test_download_data_from_gdrive_read_data(train_config):
    path = download_data_from_gdrive(train_config.downloading_params)
    data = read_data(path)
    assert isinstance(data, pd.DataFrame)


def test_split_train_val_data(synthetic_train_data, train_config):
    train_df, val_df = split_train_val_data(
        synthetic_train_data, train_config.splitting_params
    )
    assert len(val_df) / len(synthetic_train_data) == pytest.approx(
        train_config.splitting_params.val_size, 0.01
    )


def test_extract_target(synthetic_train_data, train_config):
    train_target = extract_target(synthetic_train_data, train_config.feature_params)
    assert set(train_target.values) == {0, 1}


def test_build_transformer(train_config):
    transformer = build_transformer(
        train_config.feature_params, train_config.transform_params
    )
    assert isinstance(transformer, ColumnTransformer)


def test_train_predict_synthetinc(train_config, synthetic_train_data):
    model = instantiate(train_config.model)
    train_df, val_df = split_train_val_data(
        synthetic_train_data, train_config.splitting_params
    )
    train_target = extract_target(train_df, train_config.feature_params)
    val_target = extract_target(val_df, train_config.feature_params)
    train_df = train_df.drop(labels=train_config.feature_params.target_col, axis=1)
    val_df = val_df.drop(labels=train_config.feature_params.target_col, axis=1)
    transformer = build_transformer(
        train_config.feature_params, train_config.transform_params
    )
    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    model = train_model(train_features, train_target, model)
    inference_pipeline = create_pipeline(model, transformer)
    predicts = predict_model(inference_pipeline, val_df)
    metrics = evaluate_model(
        predicts,
        val_target,
    )
    assert "accuracy" in metrics
    assert "precision" in metrics
    os.makedirs(os.path.dirname(train_config.metric_path), exist_ok=True)
    with open(train_config.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    path_to_model = serialize_model(inference_pipeline, train_config.output_model_path)
    assert os.path.exists(train_config.metric_path)
    assert os.path.exists(path_to_model)


@pytest.fixture()
def train_config():
    ml_project.enities.register_train_configs()
    with initialize(config_path="../configs"):
        train_params = compose(config_name="train_config")
    return train_params


@pytest.fixture()
def synthetic_train_data():
    synthetic_data_raw = {
        "age": [29, 77],
        "sex": [1, 0],
        "cp": [0, 1, 2, 3],
        "trestbps": [94, 200],
        "chol": [126, 564],
        "fbs": [1, 0],
        "restecg": [2, 0, 1],
        "thalach": [71, 202],
        "exang": [0, 1],
        "oldpeak": [0.0, 6.2],
        "slope": [1, 0, 2],
        "ca": [1, 2, 0, 3],
        "thal": [0, 2, 1],
        "condition": [0, 1],
    }
    synthetic_data = {}
    for column, values in synthetic_data_raw.items():
        synthetic_data[column] = np.random.choice(values, TRAIN_DATA_SIZE)
    synthetic_data = pd.DataFrame(synthetic_data)
    return synthetic_data


@pytest.fixture()
def synthetic_predict_data(synthetic_train_data):
    return synthetic_train_data.iloc[:50, :-1]
