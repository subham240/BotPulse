"""Unit tests for feature mapping and schema validation."""

import pytest
from src.feature_mapping import (
    COMMON_FEATURES,
    FEATURE_METADATA,
    TARGET_COLUMN,
    CLASS_NAMES,
    EXCLUDED_COLUMNS,
    validate_feature_dict,
)


def test_common_features_count():
    assert len(COMMON_FEATURES) == 27
    assert TARGET_COLUMN == "ROBOT"
    assert "PENALTY" in EXCLUDED_COLUMNS
    assert "ID" in EXCLUDED_COLUMNS
    assert "DATASET_SOURCE" in EXCLUDED_COLUMNS


def test_feature_metadata_completeness():
    for feat in COMMON_FEATURES:
        assert feat in FEATURE_METADATA
        meta = FEATURE_METADATA[feat]
        assert "min" in meta
        assert "max" in meta
        assert "type" in meta
        assert "default" in meta


def test_validate_feature_dict_valid():
    valid_dict = {f: FEATURE_METADATA[f]["default"] for f in COMMON_FEATURES}
    cleaned = validate_feature_dict(valid_dict)
    assert len(cleaned) == 27
    assert isinstance(cleaned["NUMBER_OF_REQUESTS"], int)
    assert isinstance(cleaned["TOTAL_DURATION"], float)


def test_validate_feature_dict_missing_column():
    invalid_dict = {"NUMBER_OF_REQUESTS": 10}
    with pytest.raises(ValueError, match="Missing required features"):
        validate_feature_dict(invalid_dict)


def test_validate_feature_dict_out_of_bounds():
    valid_dict = {f: FEATURE_METADATA[f]["default"] for f in COMMON_FEATURES}
    valid_dict["REPEATED_REQUESTS"] = 5.0  # max is 1.0
    with pytest.raises(ValueError, match="out of valid bounds"):
        validate_feature_dict(valid_dict)
