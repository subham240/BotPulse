"""
Model Evaluation and Benchmarking Module for BotPulse.

Computes comprehensive metrics on held-out test data:
- Accuracy, Precision, Recall, Macro-F1, Weighted-F1
- Per-class classification report (Human vs Web Robot)
- Confusion Matrix
- Inference latency benchmarks (mean, median, p95 per query in ms)
- Source-wise evaluation for M4D and Web Robot Detection held-out subsets
"""

import time
import json
import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.feature_mapping import CLASS_NAMES, COMMON_FEATURES


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Computes all primary classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    cm = confusion_matrix(y_true, y_pred).tolist()

    report_dict = classification_report(
        y_true, y_pred,
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "per_class": {
            "human": {
                "precision": float(prec_per_class[0]),
                "recall": float(rec_per_class[0]),
                "f1": float(f1_per_class[0]),
            },
            "robot": {
                "precision": float(prec_per_class[1]),
                "recall": float(rec_per_class[1]),
                "f1": float(f1_per_class[1]),
            },
        },
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            # Positive class probability
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def benchmark_inference_latency(
    model: Any,
    X_sample: np.ndarray,
    n_warmup: int = 10,
    n_runs: int = 100
) -> Dict[str, float]:
    """Measures single-sample inference latency in milliseconds."""
    single_record = X_sample[:1]

    # Warmup
    for _ in range(n_warmup):
        _ = model.predict(single_record)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(single_record)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    lat_arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(lat_arr)),
        "median_ms": float(np.median(lat_arr)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "min_ms": float(np.min(lat_arr)),
        "max_ms": float(np.max(lat_arr)),
    }


def evaluate_model_comprehensive(
    model: Any,
    X_test_scaled: np.ndarray,
    y_test: pd.Series,
    source_test: pd.Series,
    model_name: str,
    training_time: float = 0.0
) -> Dict[str, Any]:
    """
    Full evaluation of a trained model:
    - Overall test split metrics
    - Source-wise metrics (M4D vs Web Robot Detection held-out records)
    - Latency benchmarks
    """
    t0 = time.perf_counter()
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None
    batch_inf_time = time.perf_counter() - t0

    # Overall metrics
    overall_metrics = compute_classification_metrics(y_test.values, y_pred, y_prob)

    # Latency benchmark
    latency_info = benchmark_inference_latency(model, X_test_scaled)
    latency_info["batch_total_seconds"] = float(batch_inf_time)
    latency_info["batch_per_sample_ms"] = float((batch_inf_time / len(y_test)) * 1000.0)

    # Source-wise evaluation
    source_metrics = {}
    for source in ["M4D", "Web_Robot_Detection"]:
        mask = (source_test == source).values
        if np.any(mask):
            y_sub_true = y_test.values[mask]
            y_sub_pred = y_pred[mask]
            y_sub_prob = y_prob[mask] if y_prob is not None else None
            sub_m = compute_classification_metrics(y_sub_true, y_sub_pred, y_sub_prob)
            sub_m["sample_count"] = int(np.sum(mask))
            sub_m["class_distribution"] = {
                "human": int(np.sum(y_sub_true == 0)),
                "robot": int(np.sum(y_sub_true == 1)),
            }
            source_metrics[source] = sub_m

    return {
        "model_name": model_name,
        "training_time_seconds": float(training_time),
        "latency": latency_info,
        "overall": overall_metrics,
        "source_wise": source_metrics,
    }


def save_evaluation_report(report_data: Dict[str, Any], filepath: str) -> None:
    """Saves evaluation report dictionary to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[Evaluation] Report saved to: {filepath}")
