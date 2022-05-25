from dataclasses import dataclass


@dataclass()
class MlflowParams:
    mlflow_uri: str
    mlflow_experiment: str
