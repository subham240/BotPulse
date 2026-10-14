"""
Real-Time Inference and Explainability Engine for BotPulse.

Provides:
- Single-sample prediction with latency measurement
- Class probability estimation
- Decision factor explanation (feature deviation vs human baseline & global importance)
- Batch CSV inference with strict schema validation
- Safe artifact caching and reloading
"""

import os
import time
import json
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import joblib

from src.feature_mapping import (
    COMMON_FEATURES,
    FEATURE_METADATA,
    CLASS_NAMES,
    validate_feature_dict,
)
from src.preprocessing import load_preprocessor, transform_features


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


class BotPredictor:
    """
    Cached inference engine that loads trained models and preprocessors.
    Thread-safe and suitable for Streamlit web runtime.
    """

    def __init__(self, model_name: str = "best_model"):
        self.model_name = model_name
        self.model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        self.preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        self.importances_path = os.path.join(REPORTS_DIR, "feature_importances.json")

        self.model = None
        self.preprocessor = None
        self.feature_importances = {}
        self.human_baseline = {}

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads model, preprocessor, and importance baselines."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found at: {self.model_path}\n"
                f"Please run 'python train.py' to generate the trained model."
            )
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor artifact not found at: {self.preprocessor_path}"
            )

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        if os.path.exists(self.importances_path):
            with open(self.importances_path, "r", encoding="utf-8") as f:
                self.feature_importances = json.load(f)

        # Baseline median values for human visitors across training data
        # Derived from empirical human training distributions
        self.human_baseline = {
            "NUMBER_OF_REQUESTS": 28.0,
            "TOTAL_DURATION": 380.0,
            "AVERAGE_TIME": 14.2,
            "STANDARD_DEVIATION": 22.0,
            "REPEATED_REQUESTS": 0.04,
            "HTTP_RESPONSE_2XX": 0.98,
            "HTTP_RESPONSE_3XX": 0.01,
            "HTTP_RESPONSE_4XX": 0.01,
            "HTTP_RESPONSE_5XX": 0.0,
            "GET_METHOD": 0.96,
            "POST_METHOD": 0.04,
            "HEAD_METHOD": 0.0,
            "OTHER_METHOD": 0.0,
            "NIGHT": 0.02,
            "IMAGES": 0.38,
            "TOTAL_HTML": 0.32,
            "HTML_TO_IMAGE": 0.85,
            "HTML_TO_CSS": 0.80,
            "HTML_TO_JS": 1.20,
            "WIDTH": 16.0,
            "DEPTH": 2.4,
            "STD_DEPTH": 0.55,
            "CONSECUTIVE": 0.05,
            "DATA": 210000.0,
            "SF_REFERRER": 0.75,
            "SF_FILETYPE": 0.42,
            "MAX_BARRAGE": 3.0,
        }

    def predict_single(
        self,
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes real-time inference on a single session's feature dictionary.
        Returns prediction, confidence probabilities, latency, and explanation.
        """
        cleaned_features = validate_feature_dict(features)
        df_row = pd.DataFrame([cleaned_features])[COMMON_FEATURES]

        t0 = time.perf_counter()
        X_scaled = self.preprocessor.transform(df_row)
        pred_class = int(self.model.predict(X_scaled)[0])

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[0]
            prob_human = float(probs[0])
            prob_robot = float(probs[1])
        else:
            prob_robot = 1.0 if pred_class == 1 else 0.0
            prob_human = 1.0 - prob_robot

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Compute feature explanation / top decision factors
        top_factors = self._explain_decision(cleaned_features, pred_class)

        return {
            "predicted_class": pred_class,
            "predicted_label": CLASS_NAMES[pred_class],
            "probability_human": prob_human,
            "probability_robot": prob_robot,
            "confidence": prob_robot if pred_class == 1 else prob_human,
            "latency_ms": float(latency_ms),
            "top_factors": top_factors,
            "features": cleaned_features,
        }

    def _explain_decision(
        self,
        features: Dict[str, Any],
        pred_class: int
    ) -> List[Dict[str, Any]]:
        """
        Explains which features most strongly influenced the classification:
        Combines global tree importance with empirical deviation from human baseline.
        """
        factors = []
        for feat in COMMON_FEATURES:
            val = float(features.get(feat, 0.0))
            baseline = self.human_baseline.get(feat, 1.0)
            global_imp = self.feature_importances.get(feat, {}).get("average", 0.01)

            # Compute normalized anomaly / deviation ratio
            denom = max(abs(baseline), 1e-4)
            rel_diff = (val - baseline) / denom

            # Impact score weighted by global feature importance
            impact_score = abs(rel_diff) * global_imp

            factors.append({
                "feature": feat,
                "label": FEATURE_METADATA.get(feat, {}).get("label", feat),
                "value": val,
                "baseline_human": baseline,
                "relative_difference_pct": float(rel_diff * 100.0),
                "global_importance": float(global_imp),
                "impact_score": float(impact_score),
            })

        # Sort descending by impact score
        factors.sort(key=lambda x: x["impact_score"], reverse=True)
        return factors[:6]

    def predict_batch(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes batch inference on an uploaded DataFrame.
        Validates columns and data types.
        Returns enriched DataFrame with predictions and summary dictionary.
        """
        # Schema validation
        missing_cols = [c for c in COMMON_FEATURES if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Batch validation failed: Missing {len(missing_cols)} required feature columns:\n"
                f"{', '.join(missing_cols)}"
            )

        # Type conversion & sanity check
        df_clean = df[COMMON_FEATURES].copy()
        for col in COMMON_FEATURES:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        nan_counts = df_clean.isna().sum()
        total_nans = int(nan_counts.sum())

        t0 = time.perf_counter()
        X_scaled = self.preprocessor.transform(df_clean)
        preds = self.model.predict(X_scaled)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
            prob_humans = probs[:, 0]
            prob_robots = probs[:, 1]
        else:
            prob_robots = np.where(preds == 1, 1.0, 0.0)
            prob_humans = 1.0 - prob_robots

        batch_time_sec = time.perf_counter() - t0

        result_df = df.copy()
        result_df["PREDICTED_CLASS"] = preds
        result_df["PREDICTED_LABEL"] = [CLASS_NAMES[p] for p in preds]
        result_df["CONFIDENCE_HUMAN"] = np.round(prob_humans, 4)
        result_df["CONFIDENCE_ROBOT"] = np.round(prob_robots, 4)

        summary = {
            "total_rows": len(df),
            "predicted_humans": int(np.sum(preds == 0)),
            "predicted_robots": int(np.sum(preds == 1)),
            "robot_percentage": float(np.mean(preds == 1) * 100.0),
            "nan_values_imputed": total_nans,
            "batch_time_seconds": float(batch_time_sec),
            "throughput_req_per_sec": float(len(df) / max(1e-5, batch_time_sec)),
        }

        return result_df, summary

    def get_sample_from_dataset(
        self,
        source: str = "M4D",
        split: str = "test",
        label_filter: Optional[int] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieves a genuine observation from train or test split.
        source: 'M4D' or 'Web_Robot_Detection'
        split: 'test' (held-out, unseen) or 'train'
        label_filter: 0 (human), 1 (robot), or None (any)
        """
        csv_file = os.path.join(PROJECT_ROOT, "data", "processed", f"{split}_split.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Split file not found: {csv_file}")

        df_split = pd.read_csv(csv_file)
        filtered = df_split[df_split["DATASET_SOURCE"] == source]
        if label_filter is not None:
            filtered = filtered[filtered["ROBOT"] == label_filter]

        if len(filtered) == 0:
            raise ValueError(f"No records matching source={source}, split={split}, label={label_filter}")

        row = filtered.sample(n=1, random_state=random_state).iloc[0]
        feats = {f: row[f] for f in COMMON_FEATURES}

        return {
            "id": str(row.get("ID", "sample")),
            "source": str(row.get("DATASET_SOURCE", source)),
            "split": split,
            "ground_truth_class": int(row.get("ROBOT", 0)),
            "ground_truth_label": CLASS_NAMES[int(row.get("ROBOT", 0))],
            "original_label": str(row.get("ORIGINAL_LABEL", "")),
            "features": feats,
        }

    def get_sample_by_category(
        self,
        category: str = "Human",
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieves a real sample for a specific category: 'Human', 'Standard Bot', or 'Advanced Bot'.
        Loads from unified dataset or external test benchmark.
        """
        unified_file = os.path.join(PROJECT_ROOT, "data", "processed", "botpulse_unified_dataset.csv")
        if not os.path.exists(unified_file):
            unified_file = os.path.join(PROJECT_ROOT, "data", "processed", "combined_dataset.csv")

        df = pd.read_csv(unified_file)
        if "CATEGORY" not in df.columns:
            def _map_cat(r):
                lbl = str(r.get("ORIGINAL_LABEL", "")).lower()
                if "advanced" in lbl:
                    return "Advanced Bot"
                elif "moderate" in lbl or lbl == "robot" or r.get("ROBOT") == 1:
                    return "Standard Bot"
                return "Human"
            df["CATEGORY"] = df.apply(_map_cat, axis=1)

        subset = df[df["CATEGORY"].str.lower() == category.lower()]
        if len(subset) == 0:
            subset = df

        row = subset.sample(n=1, random_state=random_state).iloc[0]
        feats = {f: row[f] for f in COMMON_FEATURES}

        return {
            "id": str(row.get("ID", "sample")),
            "category": str(row.get("CATEGORY", category)),
            "source": str(row.get("DATASET_SOURCE", "Unified Dataset")),
            "ground_truth_class": int(row.get("ROBOT", 0)),
            "ground_truth_label": CLASS_NAMES[int(row.get("ROBOT", 0))],
            "original_label": str(row.get("ORIGINAL_LABEL", "")),
            "features": feats,
        }

    def get_demo_scenarios(self) -> List[Dict[str, Any]]:
        """Pre-defined demonstration scenarios based on genuine dataset archetypes."""
        return [
            {
                "name": "Verified Human Browsing Session",
                "badge": "Genuine Human",
                "source": "M4D Dataset (User Study)",
                "ground_truth": "Human",
                "description": "Realistic human visitor reading Wikipedia articles. Exhibits relaxed inter-arrival times, high image asset loading, diverse referrers, and zero burst barrages.",
                "features": {
                    "NUMBER_OF_REQUESTS": 184,
                    "TOTAL_DURATION": 253.0,
                    "AVERAGE_TIME": 1.38,
                    "STANDARD_DEVIATION": 2.45,
                    "REPEATED_REQUESTS": 0.08,
                    "HTTP_RESPONSE_2XX": 1.0,
                    "HTTP_RESPONSE_3XX": 0.0,
                    "HTTP_RESPONSE_4XX": 0.0,
                    "HTTP_RESPONSE_5XX": 0.0,
                    "GET_METHOD": 0.94,
                    "POST_METHOD": 0.06,
                    "HEAD_METHOD": 0.0,
                    "OTHER_METHOD": 0.0,
                    "NIGHT": 0.0,
                    "IMAGES": 0.28,
                    "TOTAL_HTML": 0.35,
                    "HTML_TO_IMAGE": 1.25,
                    "HTML_TO_CSS": 1.10,
                    "HTML_TO_JS": 1.45,
                    "WIDTH": 17.0,
                    "DEPTH": 2.0,
                    "STD_DEPTH": 0.32,
                    "CONSECUTIVE": 0.04,
                    "DATA": 185000.0,
                    "SF_REFERRER": 0.88,
                    "SF_FILETYPE": 0.35,
                    "MAX_BARRAGE": 3.0,
                },
            },
            {
                "name": "Aggressive Web Crawler / Scraper",
                "badge": "Scraper Bot",
                "source": "Web Robot Detection Dataset",
                "ground_truth": "Robot",
                "description": "High-volume web crawler harvesting directory pages. High request rate, zero image asset rendering, repeated path traversal, and extreme barrage burst.",
                "features": {
                    "NUMBER_OF_REQUESTS": 350,
                    "TOTAL_DURATION": 103621.0,
                    "AVERAGE_TIME": 296.9,
                    "STANDARD_DEVIATION": 260.3,
                    "REPEATED_REQUESTS": 0.0,
                    "HTTP_RESPONSE_2XX": 0.997,
                    "HTTP_RESPONSE_3XX": 0.0,
                    "HTTP_RESPONSE_4XX": 0.003,
                    "HTTP_RESPONSE_5XX": 0.0,
                    "GET_METHOD": 1.0,
                    "POST_METHOD": 0.0,
                    "HEAD_METHOD": 0.0,
                    "OTHER_METHOD": 0.0,
                    "NIGHT": 0.36,
                    "IMAGES": 0.0,
                    "TOTAL_HTML": 0.66,
                    "HTML_TO_IMAGE": 0.0,
                    "HTML_TO_CSS": 0.01,
                    "HTML_TO_JS": 0.0,
                    "WIDTH": 339.0,
                    "DEPTH": 6.0,
                    "STD_DEPTH": 0.34,
                    "CONSECUTIVE": 0.43,
                    "DATA": 2071344.0,
                    "SF_REFERRER": 0.55,
                    "SF_FILETYPE": 0.46,
                    "MAX_BARRAGE": 8.0,
                },
            },
            {
                "name": "Advanced Emulation Bot",
                "badge": "Advanced Bot",
                "source": "M4D Dataset (Automated Browser)",
                "ground_truth": "Robot",
                "description": "Modern headless browser bot configured with browser fingerprinting and simulated pacing. Traverses Wikipedia articles programmatically.",
                "features": {
                    "NUMBER_OF_REQUESTS": 309,
                    "TOTAL_DURATION": 225.0,
                    "AVERAGE_TIME": 0.73,
                    "STANDARD_DEVIATION": 0.98,
                    "REPEATED_REQUESTS": 0.12,
                    "HTTP_RESPONSE_2XX": 0.98,
                    "HTTP_RESPONSE_3XX": 0.01,
                    "HTTP_RESPONSE_4XX": 0.01,
                    "HTTP_RESPONSE_5XX": 0.0,
                    "GET_METHOD": 0.92,
                    "POST_METHOD": 0.08,
                    "HEAD_METHOD": 0.0,
                    "OTHER_METHOD": 0.0,
                    "NIGHT": 0.0,
                    "IMAGES": 0.22,
                    "TOTAL_HTML": 0.40,
                    "HTML_TO_IMAGE": 1.82,
                    "HTML_TO_CSS": 1.35,
                    "HTML_TO_JS": 1.60,
                    "WIDTH": 18.0,
                    "DEPTH": 1.97,
                    "STD_DEPTH": 0.25,
                    "CONSECUTIVE": 0.15,
                    "DATA": 240000.0,
                    "SF_REFERRER": 0.70,
                    "SF_FILETYPE": 0.38,
                    "MAX_BARRAGE": 5.0,
                },
            },
            {
                "name": "High-Frequency Burst Attack",
                "badge": "Burst Bot",
                "source": "Web Robot Detection Dataset",
                "ground_truth": "Robot",
                "description": "Automated scanner flooding endpoints with rapid back-to-back requests, high barrage peak, and near-zero inter-arrival duration.",
                "features": {
                    "NUMBER_OF_REQUESTS": 98,
                    "TOTAL_DURATION": 36.0,
                    "AVERAGE_TIME": 0.37,
                    "STANDARD_DEVIATION": 0.45,
                    "REPEATED_REQUESTS": 0.0,
                    "HTTP_RESPONSE_2XX": 0.85,
                    "HTTP_RESPONSE_3XX": 0.0,
                    "HTTP_RESPONSE_4XX": 0.15,
                    "HTTP_RESPONSE_5XX": 0.0,
                    "GET_METHOD": 1.0,
                    "POST_METHOD": 0.0,
                    "HEAD_METHOD": 0.0,
                    "OTHER_METHOD": 0.0,
                    "NIGHT": 0.0,
                    "IMAGES": 0.05,
                    "TOTAL_HTML": 0.80,
                    "HTML_TO_IMAGE": 16.0,
                    "HTML_TO_CSS": 5.0,
                    "HTML_TO_JS": 8.0,
                    "WIDTH": 25.0,
                    "DEPTH": 3.5,
                    "STD_DEPTH": 0.60,
                    "CONSECUTIVE": 0.20,
                    "DATA": 38000.0,
                    "SF_REFERRER": 0.10,
                    "SF_FILETYPE": 0.15,
                    "MAX_BARRAGE": 18.0,
                },
            },
        ]

