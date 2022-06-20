import os
import pickle
from typing import Tuple

import click
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

FEATURES_PATH = "data.csv"

FEATURES_PATH_TRAIN = "features_train.csv"
FEATURES_PATH_VAL = "features_val.csv"

TARGETS_PATH_TRAIN = "target_train.csv"
TARGETS_PATH_VAL = "target_val.csv"
SCALER_PATH = "scaler.pkl"
VAL_PART = 0.2
LOOKBACK = 30
PRICE_COLUMN = "Close"


def split_data(
    stock: pd.Series, lookback: int, test_size: float
) -> Tuple[np.array, np.array, np.array, np.array]:
    data_raw = stock.values
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback + 1):
        data.append(data_raw[index : index + lookback])

    data = np.array(data)
    test_set_size = int(np.round(test_size * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return x_train, y_train, x_test, y_test


@click.command()
@click.option("--in_dir")
@click.option("--out_dir")
def main(in_dir: str, out_dir: str) -> None:
    data = pd.read_csv(os.path.join(in_dir, FEATURES_PATH))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data["Close_norm"] = scaler.fit_transform(
        data["Close"].values.reshape(-1, 1)
    )
    X_train, y_train, X_val, y_val = split_data(
        data[["Close_norm"]],
        LOOKBACK,
        test_size=VAL_PART,
    )
    pd.DataFrame(X_train[:, :, 0]).to_csv(
        os.path.join(in_dir, FEATURES_PATH_TRAIN), index=False
    )
    pd.DataFrame(y_train[:, 0]).to_csv(
        os.path.join(in_dir, TARGETS_PATH_TRAIN), index=False
    )
    pd.DataFrame(X_val[:, :, 0]).to_csv(
        os.path.join(in_dir, FEATURES_PATH_VAL), index=False
    )
    pd.DataFrame(y_val[:, 0]).to_csv(
        os.path.join(in_dir, TARGETS_PATH_VAL), index=False
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, SCALER_PATH), "wb") as fout:
        pickle.dump(scaler, fout)


if __name__ == "__main__":
    main()
