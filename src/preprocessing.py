"""
Preprocessing Pipeline for BotPulse.

Provides a robust scikit-learn preprocessing pipeline for the 27 common features:
- Median imputation for missing values (e.g. STANDARD_DEVIATION for single-request sessions)
- Robust scaling to handle heavy-tailed web traffic distributions (bytes, duration, barrage)
- Strictly fitted on training data to prevent preprocessing leakage
"""

import os
from typing import Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import joblib

from src.feature_mapping import COMMON_FEATURES, validate_feature_dict


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DEFAULT_PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")


def build_preprocessor() -> Pipeline:
    """Creates an unfitted scikit-learn preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
        ]
    )


def fit_and_save_preprocessor(
    X_train: pd.DataFrame,
    save_path: Optional[str] = None
) -> Pipeline:
    """
    Fits preprocessing pipeline strictly on X_train and saves it to disk.
    Prevents any test-data leakage.
    """
    if save_path is None:
        save_path = DEFAULT_PREPROCESSOR_PATH

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    pipe = build_preprocessor()
    pipe.fit(X_train[COMMON_FEATURES])

    joblib.dump(pipe, save_path)
    print(f"[Preprocessor] Fitted pipeline on {len(X_train)} samples and saved to: {save_path}")
    return pipe


def load_preprocessor(model_path: Optional[str] = None) -> Pipeline:
    """Loads a previously fitted preprocessing pipeline from disk."""
    if model_path is None:
        model_path = DEFAULT_PREPROCESSOR_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Fitted preprocessor not found at {model_path}. "
            f"Please run model training first via 'python train.py'."
        )
    return joblib.load(model_path)


def transform_features(
    data: Union[pd.DataFrame, Dict[str, Any]],
    preprocessor: Optional[Pipeline] = None
) -> np.ndarray:
    """
    Transforms DataFrame or dictionary of feature values using the fitted pipeline.
    Validates features against the common schema.
    """
    if preprocessor is None:
        preprocessor = load_preprocessor()

    if isinstance(data, dict):
        validated = validate_feature_dict(data)
        df = pd.DataFrame([validated])[COMMON_FEATURES]
    elif isinstance(data, pd.DataFrame):
        missing = set(COMMON_FEATURES) - set(data.columns)
        if missing:
            raise ValueError(f"Input DataFrame missing required features: {missing}")
        df = data[COMMON_FEATURES].copy()
    else:
        raise TypeError(f"Expected DataFrame or dict, got {type(data)}")

    return preprocessor.transform(df)
