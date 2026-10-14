"""Tests for real-time inference, batch prediction, and explainability."""

import pandas as pd
from src.prediction import BotPredictor
from src.feature_mapping import COMMON_FEATURES, FEATURE_METADATA


def test_single_prediction_human_scenario():
    predictor = BotPredictor()
    scenarios = predictor.get_demo_scenarios()
    human_scen = next(s for s in scenarios if s["ground_truth"] == "Human")

    res = predictor.predict_single(human_scen["features"])
    assert "predicted_class" in res
    assert "predicted_label" in res
    assert "probability_human" in res
    assert "probability_robot" in res
    assert "latency_ms" in res
    assert res["latency_ms"] < 200.0  # sub-200ms latency


def test_single_prediction_robot_scenario():
    predictor = BotPredictor()
    scenarios = predictor.get_demo_scenarios()
    robot_scen = next(s for s in scenarios if s["ground_truth"] == "Robot")

    res = predictor.predict_single(robot_scen["features"])
    assert res["predicted_class"] == 1
    assert res["probability_robot"] > 0.5


def test_sample_from_dataset_held_out():
    predictor = BotPredictor()

    # M4D sample from held-out test split
    m4d_sample = predictor.get_sample_from_dataset(source="M4D", split="test")
    assert m4d_sample["source"] == "M4D"
    assert m4d_sample["split"] == "test"
    res = predictor.predict_single(m4d_sample["features"])
    assert "predicted_class" in res

    # Web Robot Detection sample from held-out test split
    wr_sample = predictor.get_sample_from_dataset(source="Web_Robot_Detection", split="test")
    assert wr_sample["source"] == "Web_Robot_Detection"
    assert wr_sample["split"] == "test"
    res_wr = predictor.predict_single(wr_sample["features"])
    assert "predicted_class" in res_wr


def test_batch_prediction():
    predictor = BotPredictor()
    # Create sample batch
    rows = []
    scenarios = predictor.get_demo_scenarios()
    for s in scenarios:
        rows.append(s["features"])

    batch_df = pd.DataFrame(rows)
    result_df, summary = predictor.predict_batch(batch_df)

    assert len(result_df) == len(scenarios)
    assert "PREDICTED_CLASS" in result_df.columns
    assert "PREDICTED_LABEL" in result_df.columns
    assert "CONFIDENCE_ROBOT" in result_df.columns
    assert summary["total_rows"] == len(scenarios)
