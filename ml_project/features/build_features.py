import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer


from ml_project.enities.feature_params import FeatureParams
from ml_project.enities.transform_params import TransformParams


def build_categorical_pipeline(transform_params: TransformParams) -> Pipeline:

    categorical_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
            )
        ]
    )
    if transform_params.ohe_categorical:
        categorical_pipeline.steps.append(["ohe", OneHotEncoder()])
    return categorical_pipeline


def build_numerical_pipeline(transform_params: TransformParams) -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ]
    )
    if transform_params.normilize_numerical:
        num_pipeline.steps.append(["normilize", Normalizer()])
    return num_pipeline


def make_features(
    transformer: ColumnTransformer, df: pd.DataFrame
) -> pd.DataFrame:
    return transformer.transform(df)


def build_transformer(
    feature_params: FeatureParams, transform_params: TransformParams
) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(transform_params),
                list(feature_params.categorical_features),
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(transform_params),
                list(feature_params.numerical_features),
            ),
        ]
    )
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
