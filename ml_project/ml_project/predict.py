import logging
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.make_dataset import (
    read_data,
)

from models.model_fit_predict import predict_model, load_model

LOG_FILEPATH = "log/predict.log"

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


@hydra.main(config_path="../configs", config_name="predict_config")
def run_predict(params: DictConfig):
    logger.debug(f"read data from: {params.data_path}")
    data = read_data(os.path.join(get_original_cwd(), params.data_path))
    logger.debug(f"load model from: {params.model_path}")
    model = load_model(os.path.join(get_original_cwd(), params.model_path))
    logger.debug(f"load model from: {params.model_path}")
    predictions = predict_model(model, data)
    data["condition"] = predictions
    logger.info(f"predictions save at: {params.output_path}")
    data.to_csv(os.path.join(get_original_cwd(), params.output_path))


if __name__ == "__main__":
    run_predict()
