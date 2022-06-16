ML CI/CD project
==============================

ML CI/CD project

Доступные модели: RandomForestClasifier (default), LogisticRegression (mщdel=logreg)

Как запустить обучение:
```
# Запустится загрузка данных и обучение RandomForestClasifier:
python ml_project/train.py
# Для использования LogisticRegression:
python ml_project/train.py mpdel=logreg
# Посмотреть текущие параметры:
python ml_project/train.py --cfg job
# Любой параметр можно изменить, например изменение стратегии split:
python ml_project/train.py splitting_params.val_size = 0.3
# или изменения параметров предобработки:
python ml_project/train.py transform_params.normilize_numerical=False
```
Как запустить применение последней обученной модели:
```
python ml_project/predict.py
# Так же можно изменить путь к модели или данными
python ml_project/predict.py model_path=new_model/model.pkl data_path=new_data_path/features.csv
```
Запуск тестов:
```
pytest
# Только unit:
pytest tests/test_units.py
# Только прогон train/predict:
pytest tests/test_train_predict.py
```


Project Organization
------------

    ├── configs            <- train and predict config yaml files.
    ├── data
    │   ├── predict        <- Data for predict by last model.
    │   └── raw            <- Data for train.
    │
    ├── models             <- Trained and serialized models, model validation metrics
    │
    ├── notebooks          <- EDA jupyter notebooks
    │
    ├── ml_project         <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download and split data
    │   │   └── make_dataset.py
    │   │
    │   ├── enities        <- Dataclasses and hydra config compose initialization function/
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── model_fit_predict.py
    │   │
    │   ├── train.py       <- Scripts to train model
    │   ├── predict.py     <- Scripts to predict with last model
    │   └── inference.py   <- Scripts to load model and validate data
    │
    ├── tests              <- Test ml_project package 
    │   ├── test_train_predict.py
    │   └── test_units.py
    |
    ├── README.md          <- The top-level README.
    │    
    └─── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
--------

