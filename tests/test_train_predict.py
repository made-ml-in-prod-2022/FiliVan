import os


def test_pred_train():
    exit_status_train = os.system("python ml_project/train.py")
    exit_status_predict = os.system("python ml_project/predict.py")
    assert exit_status_train == 0
    assert exit_status_predict == 0
