"""Tests for data loading, combining, and train/test splits."""

import os
import pandas as pd
from src.data_loader import (
    build_combined_dataset,
    create_train_test_splits,
    get_feature_matrices,
    PROCESSED_DIR,
    COMBINED_DATASET_CSV,
)
from src.feature_mapping import COMMON_FEATURES, TARGET_COLUMN


def test_combined_dataset_structure():
    assert os.path.exists(COMBINED_DATASET_CSV)
    df = pd.read_csv(COMBINED_DATASET_CSV)

    assert len(df) > 60000
    assert "DATASET_SOURCE" in df.columns
    assert "M4D" in df["DATASET_SOURCE"].values
    assert "Web_Robot_Detection" in df["DATASET_SOURCE"].values
    assert TARGET_COLUMN in df.columns

    # Verify PENALTY is NOT in common features
    assert "PENALTY" not in COMMON_FEATURES


def test_splits_source_contribution():
    train_df, test_df = create_train_test_splits()

    m4d_train = (train_df["DATASET_SOURCE"] == "M4D").sum()
    wr_train = (train_df["DATASET_SOURCE"] == "Web_Robot_Detection").sum()

    m4d_test = (test_df["DATASET_SOURCE"] == "M4D").sum()
    wr_test = (test_df["DATASET_SOURCE"] == "Web_Robot_Detection").sum()

    # Mandatory requirement: both datasets MUST contribute training observations
    assert m4d_train > 0, "M4D must contribute to training set"
    assert wr_train > 0, "Web Robot Detection must contribute to training set"

    assert m4d_test > 0, "M4D must contribute to held-out test set"
    assert wr_test > 0, "Web Robot Detection must contribute to held-out test set"


def test_feature_matrices_no_leakage():
    X_train, y_train, X_test, y_test, s_train, s_test = get_feature_matrices()
    assert list(X_train.columns) == COMMON_FEATURES
    assert list(X_test.columns) == COMMON_FEATURES
    assert "ID" not in X_train.columns
    assert "PENALTY" not in X_train.columns
    assert "DATASET_SOURCE" not in X_train.columns
