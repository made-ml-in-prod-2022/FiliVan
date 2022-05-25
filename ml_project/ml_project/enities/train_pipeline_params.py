from typing import Any, Optional

from dataclasses import dataclass

from .download_params import DownloadParams
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .model_params import RFParams, LogregParams
from .general_params import GeneralParams
from .transform_params import TransformParams
from .mlflow_params import MlflowParams
from hydra.core.config_store import ConfigStore


@dataclass()
class TrainingParams:
    model: Any
    general: GeneralParams
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    transform_params: Optional[TransformParams] = None
    downloading_params: Optional[DownloadParams] = None
    mlflow_params: Optional[MlflowParams] = None


def register_train_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=TrainingParams)
    cs.store(
        group="model",
        name="rf",
        node=RFParams,
    )
    cs.store(
        group="model",
        name="logreg",
        node=LogregParams,
    )
