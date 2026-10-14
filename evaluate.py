"""
Model Evaluation and Benchmarking CLI for BotPulse.

Loads fitted models and held-out test split:
- Evaluates Random Forest vs XGBoost
- Prints accuracy, macro-F1, precision, recall, and confusion matrix
- Prints source-wise held-out breakdown for M4D vs Web Robot Detection
- Benchmarks single-query latency
"""

import sys
import os
import json
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.feature_mapping import COMMON_FEATURES, TARGET_COLUMN
from src.data_loader import create_train_test_splits, get_feature_matrices, MODELS_DIR, REPORTS_DIR
from src.preprocessing import load_preprocessor
from src.evaluation import evaluate_model_comprehensive


def print_table(title: str, headers: list, rows: list):
    """Utility to print a clean terminal ASCII table."""
    print(f"\n--- {title} ---")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    header_line = " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(f"{str(val):<{widths[i]}}" for i, val in enumerate(row)))


def main():
    print("=" * 70)
    print(" BOTPULSE: MODEL EVALUATION & SOURCE-WISE BENCHMARK")
    print("=" * 70)

    # 1. Load test split
    _, test_df = create_train_test_splits()
    X_test = test_df[COMMON_FEATURES]
    y_test = test_df[TARGET_COLUMN]
    source_test = test_df["DATASET_SOURCE"]

    # 2. Load preprocessor
    preprocessor = load_preprocessor()
    X_test_scaled = preprocessor.transform(X_test)

    # 3. Load models
    rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    xgb_path = os.path.join(MODELS_DIR, "xgboost.joblib")

    if not os.path.exists(rf_path) or not os.path.exists(xgb_path):
        print("[ERROR] Trained models not found. Please run 'python train.py' first.")
        sys.exit(1)

    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)

    rf_eval = evaluate_model_comprehensive(rf_model, X_test_scaled, y_test, source_test, "Random Forest")
    xgb_eval = evaluate_model_comprehensive(xgb_model, X_test_scaled, y_test, source_test, "XGBoost")

    # Overall Comparison Table
    headers = ["Model", "Accuracy", "Macro-F1", "Precision", "Recall", "Weighted-F1", "Latency (ms)"]
    rows = [
        [
            "Random Forest",
            f"{rf_eval['overall']['accuracy']:.4f}",
            f"{rf_eval['overall']['macro_f1']:.4f}",
            f"{rf_eval['overall']['precision_macro']:.4f}",
            f"{rf_eval['overall']['recall_macro']:.4f}",
            f"{rf_eval['overall']['weighted_f1']:.4f}",
            f"{rf_eval['latency']['mean_ms']:.3f}",
        ],
        [
            "XGBoost",
            f"{xgb_eval['overall']['accuracy']:.4f}",
            f"{xgb_eval['overall']['macro_f1']:.4f}",
            f"{xgb_eval['overall']['precision_macro']:.4f}",
            f"{xgb_eval['overall']['recall_macro']:.4f}",
            f"{xgb_eval['overall']['weighted_f1']:.4f}",
            f"{xgb_eval['latency']['mean_ms']:.3f}",
        ],
    ]
    print_table("Held-Out Test Set Performance Benchmark", headers, rows)

    # Source-wise Breakdown Table
    sw_headers = ["Model", "Test Subset Source", "Samples", "Accuracy", "Macro-F1", "Robot Precision", "Robot Recall"]
    sw_rows = []
    for m_name, m_eval in [("Random Forest", rf_eval), ("XGBoost", xgb_eval)]:
        for src, s_data in m_eval["source_wise"].items():
            sw_rows.append([
                m_name,
                src,
                f"{s_data['sample_count']:,}",
                f"{s_data['accuracy']:.4f}",
                f"{s_data['macro_f1']:.4f}",
                f"{s_data['per_class']['robot']['precision']:.4f}",
                f"{s_data['per_class']['robot']['recall']:.4f}",
            ])
    print_table("Source-Wise Held-Out Generalization Audit", sw_headers, sw_rows)

    # Confusion Matrices
    print("\n--- Confusion Matrices ---")
    for m_name, m_eval in [("Random Forest", rf_eval), ("XGBoost", xgb_eval)]:
        cm = m_eval["overall"]["confusion_matrix"]
        print(f"\n{m_name}:")
        print(f"  Actual Human -> Predicted Human (TN): {cm[0][0]:<6} | Predicted Bot (FP): {cm[0][1]:<6}")
        print(f"  Actual Bot   -> Predicted Human (FN): {cm[1][0]:<6} | Predicted Bot (TP): {cm[1][1]:<6}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
