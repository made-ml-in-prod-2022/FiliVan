import json
import logging
import os

import hydra
from hydra.utils import instantiate, get_original_cwd

import mlflow

import ml_project.enities
from ml_project.data import download_data_from_gdrive, read_data, split_train_val_data
from ml_project.features import make_features
from ml_project.features.build_features import extract_target, build_transformer
from ml_project.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
    create_pipeline,
)

LOG_FILEPATH = "log/training.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

os.makedirs(os.path.dirname(LOG_FILEPATH), exist_ok=True)
fh = logging.FileHandler(LOG_FILEPATH)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)


# Register config by structured config schema
enities.register_train_configs()


@hydra.main(config_path="../configs", config_name="train_config")
def run_train_pipeline(params: enities.TrainingParams):
    mlflow_params = params.mlflow_params
    if mlflow_params:
        logger.debug(f"start mlflow with params: {mlflow_params}")
        mlflow_uri = (
            "file:////"
            + os.path.join(get_original_cwd(), mlflow_params.mlflow_uri)
            if "://" not in mlflow_params.mlflow_uri
            else mlflow_params.mlflow_uri
        )
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_params.mlflow_experiment)
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_params", params.model)
            mlflow.log_param("transform_params", params.transform_params)
            # mlflow.log_artifact("../../configs/train_config.yaml")
            mlflow.log_artifact(
                os.path.join(get_original_cwd(), "configs/train_config.yaml")
            )
            model_path, model, metrics = train_pipeline(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(model_path)
    else:
        return train_pipeline(params)


def train_pipeline(params):
    model = instantiate(params.model)
    downloading_params = params.downloading_params
    if downloading_params:
        logger.debug(
            f"start download dataset to {os.path.join(get_original_cwd(), downloading_params.output_folder)}"
        )

        path = download_data_from_gdrive(
            downloading_params, prefix=get_original_cwd()
        )
        logger.debug(f"Done. path to dataset: {path}")

    logger.debug(f"start train pipeline with params {params}")
    data = read_data(path)
    logger.info(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(data, params.splitting_params)

    train_target = extract_target(train_df, params.feature_params)
    val_target = extract_target(val_df, params.feature_params)
    train_df = train_df.drop(labels=params.feature_params.target_col, axis=1)
    val_df = val_df.drop(labels=params.feature_params.target_col, axis=1)

    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    transformer = build_transformer(
        params.feature_params, params.transform_params
    )
    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    logger.info(f"train_features.shape is {train_features.shape}")
    model = train_model(train_features, train_target, model)

    inference_pipeline = create_pipeline(model, transformer)
    predicts = predict_model(inference_pipeline, val_df)
    metrics = evaluate_model(predicts, val_target)
    os.makedirs(
        os.path.dirname(os.path.join(get_original_cwd(), params.metric_path)),
        exist_ok=True,
    )
    with open(
        os.path.join(get_original_cwd(), params.metric_path), "w"
    ) as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_model(
        inference_pipeline,
        os.path.join(get_original_cwd(), params.output_model_path),
    )
    # predict_config_path = os.path.join(
    #     get_original_cwd(), "configs/predict_config.yaml"
    # )
    # predict_config = OmegaConf.load(predict_config_path)
    # predict_config.model_path = os.path.join(
    #     os.getcwd(), path_to_model
    # ).replace(get_original_cwd(), "../..")
    # OmegaConf.save(config=predict_config, f=predict_config_path)
    return path_to_model, model, metrics


if __name__ == "__main__":
    run_train_pipeline()
