import os

import click
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler


FEATURES_PATH_TRAIN = "features_train.csv"
TARGETS_PATH_TRAIN = "target_train.csv"
SCALER_PATH = "scaler.pkl"
MODEL_PATH = "model.pth"
MODEL_PARAMS = {
    "input_dim": 1,
    "hidden_dim": 32,
    "num_layers": 2,
    "output_dim": 1,
}
NUM_EPOCH = 50


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim
        ).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


@click.command()
@click.option("--in_dir")
@click.option("--out_dir")
def main(in_dir: str, out_dir: str) -> None:
    features = pd.read_csv(os.path.join(in_dir, FEATURES_PATH_TRAIN)).values
    targets = pd.read_csv(os.path.join(in_dir, TARGETS_PATH_TRAIN)).values

    x_train = torch.from_numpy(np.expand_dims(features, -1)).type(torch.Tensor)
    y_train = torch.from_numpy(targets).type(torch.Tensor)
    model = GRU(*MODEL_PARAMS.values())
    criterion = torch.nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(NUM_EPOCH):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model, os.path.join(out_dir, MODEL_PATH))


if __name__ == "__main__":
    main()
