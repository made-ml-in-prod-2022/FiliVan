from dataclasses import dataclass
from typing import Optional


@dataclass
class RFParams:
    max_depth: Optional[int] = None
    _target_: str = "sklearn.ensemble.RandomForestClassifier"
    n_estimators: int = 100
    random_state: int = 42


@dataclass
class LogregParams:
    _target_: str = "sklearn.linear_model.LogisticRegression"
    penalty: str = "l1"
    solver: str = "liblinear"
    C: float = 1.0
    random_state: int = 42
    max_iter: int = 42
