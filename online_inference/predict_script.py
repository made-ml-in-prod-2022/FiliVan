import os
import requests
import json
import logging
import click
import pandas as pd

LOG_FILEPATH = "log/predict_script.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

os.makedirs(os.path.dirname(LOG_FILEPATH), exist_ok=True)
fh = logging.FileHandler(LOG_FILEPATH)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

PATH_TO_DATA = "./data/predict/features.csv"
LOCALHOST = "127.0.0.1"
PORT = 8000
ENDPOINT = "predict"


@click.command(name="predict")
@click.option(
    "--in",
    "-i",
    "data_path",
    default=PATH_TO_DATA,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help=f"Path to .csv files with data to prediction (default={PATH_TO_DATA})",
)
@click.option(
    "--host",
    "-h",
    "host",
    default=LOCALHOST,
    type=str,
    help=f"Host (default={LOCALHOST})",
)
@click.option(
    "--port", "-p", "port", default=PORT, type=int, help=f"Port (default={PORT})"
)
def make_predict_request(data_path: str, host: str, port: int):
    logger.info("Reading data")
    data = pd.read_csv(data_path)
    data["id"] = range(len(data))
    request_data = data.to_dict(orient="records")
    logger.info(f"Request data samples:\n {request_data[:5]}")
    logger.info("Sending post request")
    response = requests.post(
        f"http://{host}:{port}/{ENDPOINT}", json.dumps(request_data)
    )
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response data samples:\n {response.json()}")
    if response.status_code == 200:
        print(f"Response data samples:\n {response.json()}")


if __name__ == "__main__":
    make_predict_request()
