from dataclasses import dataclass
from typing import Optional


@dataclass()
class TransformParams:
    ohe_categorical: Optional[bool]
    normilize_numerical: Optional[bool]
