"""
Data Loader and Combined Dataset Pipeline for BotPulse.

Loads:
- Dataset A: M4D Web Bot Detection Dataset
- Dataset B: Web Robot Detection Dataset (simple_features.csv)

Enforces:
- Common feature schema selection
- Target label harmonization (0 = Human, 1 = Bot)
- Leakage prevention (strictly drops PENALTY, ID, PPI, UNASSIGNED from X)
- Group-aware and stratified splitting
- Source auditing without source memorization
"""

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.feature_mapping import COMMON_FEATURES, TARGET_COLUMN
from src.feature_extraction import process_m4d_dataset


# Default project directories relative to this file
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DEFAULT_RAW_WEB_ROBOT = os.path.join(PROJECT_ROOT, "data", "raw", "web_robot_detection", "simple_features.csv")
FALLBACK_RAW_WEB_ROBOT = r"C:\Users\lenovo\Downloads\web_bot_detection_dataset\simple_features.csv"

DEFAULT_RAW_M4D_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "m4d")
FALLBACK_RAW_M4D_DIR = r"C:\Users\lenovo\Downloads\web_bot_detection_dataset"

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
M4D_PROCESSED_CSV = os.path.join(PROCESSED_DIR, "m4d_extracted_features.csv")
COMBINED_DATASET_CSV = os.path.join(PROCESSED_DIR, "combined_dataset.csv")
TRAIN_SPLIT_CSV = os.path.join(PROCESSED_DIR, "train_split.csv")
TEST_SPLIT_CSV = os.path.join(PROCESSED_DIR, "test_split.csv")


def locate_file(preferred: str, fallback: str, description: str) -> str:
    """Locates a file, trying preferred project location first, then fallback."""
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"Could not locate {description}.\n"
        f"Checked preferred: {preferred}\n"
        f"Checked fallback: {fallback}\n"
        f"Please ensure the dataset file is placed in data/raw/."
    )


def load_web_robot_detection(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and standardizes Web Robot Detection (simple_features.csv).
    Excludes PENALTY, PPI, UNASSIGNED, preserves auditing metadata.
    """
    if csv_path is None:
        csv_path = locate_file(DEFAULT_RAW_WEB_ROBOT, FALLBACK_RAW_WEB_ROBOT, "simple_features.csv")

    print(f"[Data Loader] Loading Web Robot Detection from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate target column presence
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {csv_path}")

    # Check for target leakage column PENALTY
    if "PENALTY" in df.columns:
        print("[Data Loader] Identified 'PENALTY' in raw data -> excluding from predictive features to prevent target leakage.")

    # Retain common features + metadata
    keep_cols = [c for c in COMMON_FEATURES if c in df.columns]
    missing_feats = set(COMMON_FEATURES) - set(keep_cols)
    if missing_feats:
        raise ValueError(f"Web Robot Detection CSV missing required schema features: {missing_feats}")

    df_clean = df[keep_cols + [TARGET_COLUMN]].copy()
    df_clean["ID"] = df["ID"].astype(str) if "ID" in df.columns else [f"WR_{i}" for i in range(len(df))]
    df_clean["DATASET_SOURCE"] = "Web_Robot_Detection"
    df_clean["ORIGINAL_LABEL"] = df[TARGET_COLUMN].map({0: "human", 1: "robot"})
    df_clean["BASE_GROUP"] = df_clean["ID"]

    print(f"[Data Loader] Web Robot Detection loaded: {len(df_clean)} records. Classes: {dict(df_clean[TARGET_COLUMN].value_counts())}")
    return df_clean


def load_m4d_dataset(m4d_dir: Optional[str] = None, force_reextract: bool = False) -> pd.DataFrame:
    """
    Loads M4D dataset features. Uses cached processed CSV if available,
    otherwise runs the feature extractor on raw Apache logs.
    """
    if not force_reextract and os.path.exists(M4D_PROCESSED_CSV):
        print(f"[Data Loader] Loading pre-extracted M4D features from: {M4D_PROCESSED_CSV}")
        df_m4d = pd.read_csv(M4D_PROCESSED_CSV)
    else:
        if m4d_dir is None:
            m4d_dir = locate_file(DEFAULT_RAW_M4D_DIR, FALLBACK_RAW_M4D_DIR, "M4D dataset directory")
        df_m4d = process_m4d_dataset(m4d_dir, output_csv_path=M4D_PROCESSED_CSV)

    # Standardize metadata
    df_m4d["DATASET_SOURCE"] = "M4D"
    df_m4d["ID"] = df_m4d["ID"].astype(str)
    # Group base ID for M4D to prevent related session leakage (e.g. user_0 and user_1)
    df_m4d["BASE_GROUP"] = df_m4d["ID"].apply(lambda x: x.split("_")[0])

    print(f"[Data Loader] M4D Dataset loaded: {len(df_m4d)} records. Classes: {dict(df_m4d[TARGET_COLUMN].value_counts())}")
    return df_m4d


def build_combined_dataset(
    web_robot_path: Optional[str] = None,
    m4d_dir: Optional[str] = None,
    force_rebuild: bool = False
) -> pd.DataFrame:
    """
    Combines observations from both M4D and Web Robot Detection into one unified DataFrame.
    Saves to data/processed/combined_dataset.csv.
    """
    if not force_rebuild and os.path.exists(COMBINED_DATASET_CSV):
        print(f"[Data Loader] Loading existing combined dataset from: {COMBINED_DATASET_CSV}")
        return pd.read_csv(COMBINED_DATASET_CSV)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_wr = load_web_robot_detection(web_robot_path)
    df_m4d = load_m4d_dataset(m4d_dir)

    # Ensure all common features are present
    common_cols = COMMON_FEATURES + [TARGET_COLUMN, "ID", "DATASET_SOURCE", "ORIGINAL_LABEL", "BASE_GROUP"]

    df_combined = pd.concat([df_wr[common_cols], df_m4d[common_cols]], ignore_index=True)
    df_combined.to_csv(COMBINED_DATASET_CSV, index=False)
    print(f"[Data Loader] Successfully saved combined dataset: {COMBINED_DATASET_CSV}")
    print(f"  Total records: {len(df_combined)}")
    print(f"  Source breakdown: {dict(df_combined['DATASET_SOURCE'].value_counts())}")
    print(f"  Target distribution: {dict(df_combined[TARGET_COLUMN].value_counts())}")
    return df_combined


def create_train_test_splits(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    force_resplit: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates leakage-preventing train and test splits:
    - Stratified by both target label and dataset source
    - Group-aware for sessions sharing user/base identities
    Saves splits to data/processed/train_split.csv and test_split.csv.
    """
    if not force_resplit and os.path.exists(TRAIN_SPLIT_CSV) and os.path.exists(TEST_SPLIT_CSV):
        print(f"[Data Loader] Loading existing train and test splits from: {PROCESSED_DIR}")
        train_df = pd.read_csv(TRAIN_SPLIT_CSV)
        test_df = pd.read_csv(TEST_SPLIT_CSV)
        return train_df, test_df

    if df is None:
        df = build_combined_dataset()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Split M4D and Web Robot Detection separately with stratification and grouping
    # M4D has group structure (BASE_GROUP):
    df_m4d = df[df["DATASET_SOURCE"] == "M4D"].copy()
    df_wr = df[df["DATASET_SOURCE"] == "Web_Robot_Detection"].copy()

    # M4D Group-aware split
    # Aggregate M4D by BASE_GROUP for stratification
    group_summary = df_m4d.groupby("BASE_GROUP").agg({
        TARGET_COLUMN: lambda x: int(x.mode()[0])
    }).reset_index()

    train_groups, test_groups = train_test_split(
        group_summary["BASE_GROUP"],
        test_size=test_size,
        stratify=group_summary[TARGET_COLUMN],
        random_state=random_state
    )
    m4d_train = df_m4d[df_m4d["BASE_GROUP"].isin(train_groups)]
    m4d_test = df_m4d[df_m4d["BASE_GROUP"].isin(test_groups)]

    # Web Robot Detection stratified split
    wr_train, wr_test = train_test_split(
        df_wr,
        test_size=test_size,
        stratify=df_wr[TARGET_COLUMN],
        random_state=random_state
    )

    train_df = pd.concat([wr_train, m4d_train], ignore_index=True)
    test_df = pd.concat([wr_test, m4d_test], ignore_index=True)

    # Shuffle training set
    train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    train_df.to_csv(TRAIN_SPLIT_CSV, index=False)
    test_df.to_csv(TEST_SPLIT_CSV, index=False)

    print("[Data Loader] Created train/test splits:")
    print(f"  Train set: {len(train_df)} observations (M4D: {sum(train_df['DATASET_SOURCE'] == 'M4D')}, Web_Robot: {sum(train_df['DATASET_SOURCE'] == 'Web_Robot_Detection')})")
    print(f"  Test set:  {len(test_df)} observations (M4D: {sum(test_df['DATASET_SOURCE'] == 'M4D')}, Web_Robot: {sum(test_df['DATASET_SOURCE'] == 'Web_Robot_Detection')})")
    return train_df, test_df


def get_feature_matrices(
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Returns X_train, y_train, X_test, y_test, source_train, source_test.
    X contains ONLY the 27 common features.
    """
    if train_df is None or test_df is None:
        train_df, test_df = create_train_test_splits()

    X_train = train_df[COMMON_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()
    source_train = train_df["DATASET_SOURCE"].copy()

    X_test = test_df[COMMON_FEATURES].copy()
    y_test = test_df[TARGET_COLUMN].copy()
    source_test = test_df["DATASET_SOURCE"].copy()

    return X_train, y_train, X_test, y_test, source_train, source_test
