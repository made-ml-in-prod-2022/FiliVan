import os
import pickle

import click
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error


FEATURES_PATH_VALID = "features_val.csv"
TARGETS_PATH_VALID = "target_val.csv"
METRICS_FILEPATH = "metrics.txt"
SCALER_PATH = "scaler.pkl"
MODEL_PATH = "model.pth"
LOOKBACK = 30
PRICE_COLUMN = "Close"
USE_PRED = True


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


@click.command()
@click.option("--model_dir")
@click.option("--data_dir")
def main(model_dir: str, data_dir: str) -> None:
    features = pd.read_csv(os.path.join(data_dir, FEATURES_PATH_VALID)).values
    targets = pd.read_csv(os.path.join(data_dir, TARGETS_PATH_VALID)).values

    scaler_path = os.path.join(model_dir, SCALER_PATH)
    with open(scaler_path, "rb") as fin:
        scaler = pickle.load(fin)

    model_path = os.path.join(model_dir, MODEL_PATH)
    model = torch.load(model_path)
    model.eval()
    x_test = torch.from_numpy(np.expand_dims(features, -1)).type(torch.Tensor)
    y_test = torch.from_numpy(targets).type(torch.Tensor)

    y_pred = model(x_test)
    # invert predictions
    y_test = scaler.inverse_transform(y_test.detach().numpy())
    y_pred = scaler.inverse_transform(y_pred.detach().numpy())

    # calculate root mean squared error
    testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    print("Test Score: %.2f RMSE" % (testScore))

    with open(os.path.join(model_dir, METRICS_FILEPATH), "w") as fout:
        fout.write(f"RMSE: {testScore}")


if __name__ == "__main__":
    main()
