import os
from typing import Tuple

import pandas as pd
import gdown
from sklearn.model_selection import train_test_split


from enities import SplittingParams, DownloadParams


def download_data_from_gdrive(params: DownloadParams, prefix: str = "") -> str:
    output_path = os.path.join(prefix, params.output_folder)
    os.makedirs(output_path, exist_ok=True)
    path = gdown.download_folder(
        id=params.gdrive_id, output=output_path, quiet=True
    )[0]
    return path


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :rtype: object
    """
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
