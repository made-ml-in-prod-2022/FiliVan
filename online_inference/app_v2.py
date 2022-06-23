import os
from pyexpat import features
import sys
import time
from datetime import datetime
from typing import List, Optional, Union, Dict
import logging

import gdown
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml_project.enities import InputData, PredictResponse
from ml_project.data.make_dataset import read_data
from ml_project.inference import download_model, check_data_valid
from ml_project.models.model_fit_predict import predict_model


TIME_TO_SLEEP = 20
TIME_TO_FAIL = 90

SOURCE = "gdrive"
DEFAULT_DOWNLOAD_MODEL_PATH = "https://drive.google.com/file/d/1oGfWKZVbueiiMZ1L2f_BqDv0oiIE3uH8/view?usp=sharing"
DEFAULT_DOWNLOAD_DATA_PATH = (
    "https://drive.google.com/file/d/1avYStm_N9QqfelsJsMZyYwsO8LBs1GKl"
)

DATA_PATH = "data/predict/"
MODEL_PATH = "models/"
LOG_FILEPATH = "log/online_inference.log"


SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]

model: Optional[SklearnClassifierModel] = None


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

app = FastAPI()
start_time = datetime.now()


@app.on_event("startup")
def load_model():
    time.sleep(TIME_TO_SLEEP)
    model_path = os.getenv("PATH_TO_MODEL")
    logger.info(f"Downloading model from: {model_path}")
    global model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model = download_model(model_path, source=SOURCE)


@app.get("/")
def main():
    return (
        "Check the '/docs' endpoint to see how to get predictions on your data"
    )


@app.get("/health")
def status() -> bool:
    global model
    global start_time

    elapsed_time = datetime.now() - start_time
    if elapsed_time.seconds > TIME_TO_FAIL:
        raise Exception("app is dead")
    return model is not None


@app.api_route(
    "/predict", response_model=List[PredictResponse], methods=["POST"]
)
def predict(request: List[InputData]):
    for data in request:
        logger.info(f"Checking dataframe for model consistency")
        is_valid, error_message = check_data_valid(data)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
    df = pd.DataFrame(example.__dict__ for example in request)
    ids = df["id"].values
    prediction = predict_model(model=model, features=df.drop("id", axis=1))
    return [
        PredictResponse(id=f_id, target=prediction)
        for f_id, prediction in zip(ids, prediction)
    ]


if __name__ == "__main__":
    uvicorn.run("app_v2:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
