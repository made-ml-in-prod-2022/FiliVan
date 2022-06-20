import os
import pickle

import click
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

INFERENCE_DATAPATH = "data.csv"
MODEL_PATH = "model.pth"
SCALER_PATH = "scaler.pkl"
MODEL_PATH = "model.pth"
PREDS_PATH = "predictions.csv"
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


def get_tomorow_close_prediction(
    data: pd.DataFrame, model: GRU, scaler: MinMaxScaler, lookback: int = 30
) -> float:
    last_data = data[PRICE_COLUMN][-lookback + 1 :].values.reshape(-1, 1)
    norm_last_data = scaler.transform(last_data)
    norm_last_data = torch.from_numpy(np.expand_dims(norm_last_data, axis=0)).type(
        torch.Tensor
    )
    pred_norm = model(norm_last_data).detach().numpy()
    pred = scaler.inverse_transform(pred_norm)
    return pred[0][0]


def get_signals(
    price: np.array, pred_price: np.array = None, treshhold: float = 10
) -> np.array:
    price = pd.Series(price)
    pred_price = pred_price if pred_price is not None else price
    day_change = price - pred_price.shift(-1)
    # Смотрим в будущее, если цена завтра > сегодня + treshhold, то сегодня покупаем
    signals = (day_change < -treshhold).astype(int)
    return signals


def get_today_signal(today_price: float, pred_price: float) -> int:
    return get_signals(np.array([today_price, pred_price])).values[0]


@click.command()
@click.option("--in_dir")
@click.option("--model_dir")
@click.option("--pred_dir")
def main(in_dir: str, model_dir: str, pred_dir: str) -> None:
    path = os.path.join(in_dir, INFERENCE_DATAPATH)
    data = pd.read_csv(path)

    scaler_path = os.path.join(model_dir, SCALER_PATH)
    with open(scaler_path, "rb") as fin:
        scaler = pickle.load(fin)

    model_path = os.path.join(model_dir, MODEL_PATH)
    model = torch.load(model_path)
    model.eval()

    if USE_PRED:
        today_price = get_tomorow_close_prediction(
            data.iloc[:-1, :], model, scaler, LOOKBACK
        )
    else:
        today_price = data.iloc[-1, :][PRICE_COLUMN]
    pred_price = get_tomorow_close_prediction(data, model, scaler, LOOKBACK)
    today_signal = get_today_signal(today_price, pred_price)
    os.makedirs(pred_dir, exist_ok=True)
    pred = pd.DataFrame(
        [[data.iloc[-1, :]["Date"], today_signal]], columns=["Date", "signal"]
    )
    pred_path = os.path.join(pred_dir, PREDS_PATH)
    pred.to_csv(pred_path, index=False)


if __name__ == "__main__":
    main()
