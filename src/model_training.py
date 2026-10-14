"""
Combined Model Training Pipeline for BotPulse.

Trains and compares:
1. Random Forest Classifier
2. XGBoost Classifier

Protocol:
- Trains on unified observations from BOTH M4D and Web Robot Detection
- Evaluates on leakage-preventing held-out test split
- Selects best model by held-out Macro-F1
- Saves fitted models, feature schemas, training configs, and feature importances
"""

import os
import time
import json
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.feature_mapping import COMMON_FEATURES, TARGET_COLUMN, FEATURE_METADATA, CLASS_NAMES
from src.data_loader import create_train_test_splits, get_feature_matrices
from src.preprocessing import fit_and_save_preprocessor
from src.evaluation import evaluate_model_comprehensive, save_evaluation_report


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def train_random_forest(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, float]:
    """Trains a tuned Random Forest Classifier on scaled features."""
    print("\n[Training] Training Random Forest Classifier...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train_scaled, y_train.values)
    train_time = time.perf_counter() - t0
    print(f"[Training] Random Forest trained in {train_time:.2f} seconds.")
    return rf, train_time


def train_xgboost(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[XGBClassifier, float]:
    """Trains an XGBoost Classifier on scaled features."""
    print("\n[Training] Training XGBoost Classifier...")
    t0 = time.perf_counter()

    # Calculate class weight scale_pos_weight
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos = (n_neg / max(1, n_pos))

    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
    )
    xgb.fit(X_train_scaled, y_train.values)
    train_time = time.perf_counter() - t0
    print(f"[Training] XGBoost trained in {train_time:.2f} seconds.")
    return xgb, train_time


def extract_feature_importances(
    rf_model: RandomForestClassifier,
    xgb_model: XGBClassifier
) -> Dict[str, Dict[str, float]]:
    """Extracts and normalizes feature importances from both tree models."""
    rf_imp = rf_model.feature_importances_
    xgb_imp = xgb_model.feature_importances_

    importances = {}
    for i, feat in enumerate(COMMON_FEATURES):
        importances[feat] = {
            "random_forest": float(rf_imp[i]),
            "xgboost": float(xgb_imp[i]),
            "average": float((rf_imp[i] + xgb_imp[i]) / 2.0),
        }

    # Sort descending by average
    sorted_importances = dict(
        sorted(importances.items(), key=lambda item: item[1]["average"], reverse=True)
    )
    return sorted_importances


def run_full_training_pipeline(
    force_rebuild: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the entire end-to-end model training workflow:
    1. Loads / extracts data from both M4D and Web Robot Detection
    2. Builds combined dataset and leakage-free train/test splits
    3. Fits preprocessor strictly on X_train
    4. Trains Random Forest and XGBoost
    5. Evaluates both on held-out test split (overall and source-wise)
    6. Selects and saves the best model based on Macro-F1
    7. Saves all artifacts to disk
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 70)
    print(" BOTPULSE: COMBINED MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Prepare splits
    train_df, test_df = create_train_test_splits(force_resplit=force_rebuild, random_state=random_state)
    X_train, y_train, X_test, y_test, source_train, source_test = get_feature_matrices(train_df, test_df)

    # Verify observations from both datasets were used in training
    m4d_train_cnt = int(np.sum(source_train == "M4D"))
    wr_train_cnt = int(np.sum(source_train == "Web_Robot_Detection"))
    total_train_cnt = len(X_train)

    m4d_test_cnt = int(np.sum(source_test == "M4D"))
    wr_test_cnt = int(np.sum(source_test == "Web_Robot_Detection"))
    total_test_cnt = len(X_test)

    print("\n--- Training Set Source Contribution Audit ---")
    print(f"Total Training Observations:        {total_train_cnt:,}")
    print(f"  Contributed by M4D:                {m4d_train_cnt:,} ({m4d_train_cnt/total_train_cnt*100:.2f}%)")
    print(f"  Contributed by Web Robot Detection: {wr_train_cnt:,} ({wr_train_cnt/total_train_cnt*100:.2f}%)")
    print(f"Total Held-out Test Observations:    {total_test_cnt:,}")
    print(f"  M4D Test Observations:             {m4d_test_cnt:,}")
    print(f"  Web Robot Detection Test:          {wr_test_cnt:,}")

    if m4d_train_cnt == 0 or wr_train_cnt == 0:
        raise RuntimeError("FATAL: Combined training requirement failed! Both datasets must contribute training observations.")

    # 2. Fit preprocessor strictly on X_train
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    preprocessor = fit_and_save_preprocessor(X_train, save_path=preprocessor_path)

    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # 3. Train Random Forest
    rf_model, rf_train_time = train_random_forest(X_train_scaled, y_train, random_state=random_state)
    rf_eval = evaluate_model_comprehensive(
        rf_model, X_test_scaled, y_test, source_test, "Random Forest", rf_train_time
    )

    # 4. Train XGBoost
    xgb_model, xgb_train_time = train_xgboost(X_train_scaled, y_train, random_state=random_state)
    xgb_eval = evaluate_model_comprehensive(
        xgb_model, X_test_scaled, y_test, source_test, "XGBoost", xgb_train_time
    )

    # 5. Extract Feature Importances
    feat_importances = extract_feature_importances(rf_model, xgb_model)
    imp_path = os.path.join(REPORTS_DIR, "feature_importances.json")
    with open(imp_path, "w", encoding="utf-8") as f:
        json.dump(feat_importances, f, indent=2)

    # 6. Compare & Select Best Model by held-out Macro-F1
    rf_macro_f1 = rf_eval["overall"]["macro_f1"]
    xgb_macro_f1 = xgb_eval["overall"]["macro_f1"]

    print("\n--- Held-Out Test Evaluation Comparison ---")
    print(f"Random Forest Macro-F1: {rf_macro_f1:.4f} | Accuracy: {rf_eval['overall']['accuracy']:.4f} | Latency: {rf_eval['latency']['mean_ms']:.3f} ms")
    print(f"XGBoost       Macro-F1: {xgb_macro_f1:.4f} | Accuracy: {xgb_eval['overall']['accuracy']:.4f} | Latency: {xgb_eval['latency']['mean_ms']:.3f} ms")

    if xgb_macro_f1 >= rf_macro_f1:
        best_name = "XGBoost"
        best_model = xgb_model
        best_eval = xgb_eval
    else:
        best_name = "Random Forest"
        best_model = rf_model
        best_eval = rf_eval

    print(f"\n[Winner] Selected Champion Model: {best_name} (Macro-F1: {best_eval['overall']['macro_f1']:.4f})")

    # 7. Persist Models and Artifacts
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.joblib"))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost.joblib"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))

    # Save training configuration & metadata
    training_config = {
        "best_model_name": best_name,
        "features": COMMON_FEATURES,
        "n_features": len(COMMON_FEATURES),
        "target_column": TARGET_COLUMN,
        "class_mapping": CLASS_NAMES,
        "training_counts": {
            "total": total_train_cnt,
            "m4d": m4d_train_cnt,
            "web_robot_detection": wr_train_cnt,
        },
        "test_counts": {
            "total": total_test_cnt,
            "m4d": m4d_test_cnt,
            "web_robot_detection": wr_test_cnt,
        },
        "evaluation_summary": {
            "random_forest": {
                "accuracy": rf_eval["overall"]["accuracy"],
                "macro_f1": rf_eval["overall"]["macro_f1"],
                "latency_mean_ms": rf_eval["latency"]["mean_ms"],
            },
            "xgboost": {
                "accuracy": xgb_eval["overall"]["accuracy"],
                "macro_f1": xgb_eval["overall"]["macro_f1"],
                "latency_mean_ms": xgb_eval["latency"]["mean_ms"],
            },
        },
    }
    with open(os.path.join(MODELS_DIR, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    with open(os.path.join(MODELS_DIR, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump(FEATURE_METADATA, f, indent=2)

    # Save comprehensive evaluation metrics
    all_eval_reports = {
        "champion_model": best_name,
        "models": {
            "random_forest": rf_eval,
            "xgboost": xgb_eval,
        },
        "audit": {
            "m4d_train_observations": m4d_train_cnt,
            "web_robot_train_observations": wr_train_cnt,
            "m4d_test_observations": m4d_test_cnt,
            "web_robot_test_observations": wr_test_cnt,
        },
    }
    save_evaluation_report(all_eval_reports, os.path.join(REPORTS_DIR, "evaluation_metrics.json"))

    print("\n[Training Pipeline] Completed successfully! All model and report artifacts saved.")
    return all_eval_reports


if __name__ == "__main__":
    run_full_training_pipeline()
