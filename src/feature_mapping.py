"""
Feature Mapping and Common Schema Specification for BotPulse.

Defines the common feature schema bridging:
1. Dataset A: M4D Web Bot Detection Dataset (raw Apache web server logs)
2. Dataset B: Web Robot Detection Dataset (simple_features.csv)

Ensures label compatibility, excludes target leakage (e.g. PENALTY, ID),
and provides validation metadata for user inputs and batch CSVs.
"""

from typing import Dict, List, Any


# 27 Common Predictive Server-Log Features
COMMON_FEATURES: List[str] = [
    "NUMBER_OF_REQUESTS",
    "TOTAL_DURATION",
    "AVERAGE_TIME",
    "STANDARD_DEVIATION",
    "REPEATED_REQUESTS",
    "HTTP_RESPONSE_2XX",
    "HTTP_RESPONSE_3XX",
    "HTTP_RESPONSE_4XX",
    "HTTP_RESPONSE_5XX",
    "GET_METHOD",
    "POST_METHOD",
    "HEAD_METHOD",
    "OTHER_METHOD",
    "NIGHT",
    "IMAGES",
    "TOTAL_HTML",
    "HTML_TO_IMAGE",
    "HTML_TO_CSS",
    "HTML_TO_JS",
    "WIDTH",
    "DEPTH",
    "STD_DEPTH",
    "CONSECUTIVE",
    "DATA",
    "SF_REFERRER",
    "SF_FILETYPE",
    "MAX_BARRAGE",
]

# Columns to strictly exclude from the model feature matrix
EXCLUDED_COLUMNS: List[str] = [
    "ID",                 # Identifier memorization
    "PENALTY",            # Direct target-leakage heuristic from Zenodo dataset
    "ROBOT",              # Target label
    "DATASET_SOURCE",     # Auditing column (prevents dataset-source memorization)
    "ORIGINAL_LABEL",     # Raw source label string
    "UNASSIGNED",         # Zenodo-specific parsing artifact
    "PPI",                # Zenodo-specific metric
    "M4D_DETAILED_LABEL", # Fine-grained M4D label
]

# Target column name and label encoding
TARGET_COLUMN: str = "ROBOT"
CLASS_NAMES: Dict[int, str] = {
    0: "Human (Non-Robot)",
    1: "Web Robot (Bot)",
}

# Detailed schema definitions with user-friendly descriptions and valid ranges
FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "NUMBER_OF_REQUESTS": {
        "label": "Number of Requests",
        "description": "Total count of HTTP requests in the browsing session.",
        "type": "integer",
        "min": 1,
        "max": 20000,
        "default": 35,
        "unit": "requests",
    },
    "TOTAL_DURATION": {
        "label": "Total Duration",
        "description": "Total session duration between first and last request.",
        "type": "float",
        "min": 0.0,
        "max": 1200000.0,
        "default": 450.0,
        "unit": "seconds",
    },
    "AVERAGE_TIME": {
        "label": "Average Inter-arrival Time",
        "description": "Mean time elapsed between consecutive requests.",
        "type": "float",
        "min": 0.0,
        "max": 2000.0,
        "default": 12.8,
        "unit": "seconds",
    },
    "STANDARD_DEVIATION": {
        "label": "Inter-arrival Std. Dev.",
        "description": "Standard deviation of intervals between requests (variance in pacing).",
        "type": "float",
        "min": 0.0,
        "max": 1000.0,
        "default": 18.5,
        "unit": "seconds",
    },
    "REPEATED_REQUESTS": {
        "label": "Repeated Requests Ratio",
        "description": "Fraction of requests to URLs that were already requested earlier.",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.05,
        "unit": "ratio [0-1]",
    },
    "HTTP_RESPONSE_2XX": {
        "label": "HTTP 2xx Success Ratio",
        "description": "Proportion of requests returning 2xx status codes (e.g. 200 OK).",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.95,
        "unit": "ratio [0-1]",
    },
    "HTTP_RESPONSE_3XX": {
        "label": "HTTP 3xx Redirection Ratio",
        "description": "Proportion of requests returning 3xx redirect status codes.",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.02,
        "unit": "ratio [0-1]",
    },
    "HTTP_RESPONSE_4XX": {
        "label": "HTTP 4xx Client Error Ratio",
        "description": "Proportion of requests returning 4xx error status codes (e.g. 404).",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.03,
        "unit": "ratio [0-1]",
    },
    "HTTP_RESPONSE_5XX": {
        "label": "HTTP 5xx Server Error Ratio",
        "description": "Proportion of requests returning 5xx server error codes.",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.0,
        "unit": "ratio [0-1]",
    },
    "GET_METHOD": {
        "label": "GET Method Ratio",
        "description": "Proportion of HTTP requests using the GET method.",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.98,
        "unit": "ratio [0-1]",
    },
    "POST_METHOD": {
        "label": "POST Method Ratio",
        "description": "Proportion of HTTP requests using the POST method.",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.02,
        "unit": "ratio [0-1]",
    },
    "HEAD_METHOD": {
        "label": "HEAD Method Ratio",
        "description": "Proportion of HTTP requests using the HEAD method.",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.0,
        "unit": "ratio [0-1]",
    },
    "OTHER_METHOD": {
        "label": "Other HTTP Method Ratio",
        "description": "Proportion of requests using other methods (PUT, DELETE, OPTIONS).",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.0,
        "unit": "ratio [0-1]",
    },
    "NIGHT": {
        "label": "Night Traffic Ratio",
        "description": "Proportion of requests occurring during off-hours (00:00 to 06:00).",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.0,
        "unit": "ratio [0-1]",
    },
    "IMAGES": {
        "label": "Images Resource Ratio",
        "description": "Proportion of requests requesting image assets (.png, .jpg, .ico).",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.35,
        "unit": "ratio [0-1]",
    },
    "TOTAL_HTML": {
        "label": "HTML / Page Ratio",
        "description": "Proportion of requests requesting HTML documents or root pages.",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.30,
        "unit": "ratio [0-1]",
    },
    "HTML_TO_IMAGE": {
        "label": "HTML-to-Image Ratio",
        "description": "Ratio of HTML requests to image asset requests.",
        "type": "float",
        "min": 0.0,
        "max": 120.0,
        "default": 1.2,
        "unit": "ratio",
    },
    "HTML_TO_CSS": {
        "label": "HTML-to-CSS Ratio",
        "description": "Ratio of HTML requests to CSS stylesheet requests.",
        "type": "float",
        "min": 0.0,
        "max": 250.0,
        "default": 0.8,
        "unit": "ratio",
    },
    "HTML_TO_JS": {
        "label": "HTML-to-JS Ratio",
        "description": "Ratio of HTML requests to JavaScript script requests.",
        "type": "float",
        "min": 0.0,
        "max": 25.0,
        "default": 1.5,
        "unit": "ratio",
    },
    "WIDTH": {
        "label": "Session Breadth (Width)",
        "description": "Number of distinct, unique URLs visited during the session.",
        "type": "float",
        "min": 1.0,
        "max": 15000.0,
        "default": 18.0,
        "unit": "unique URLs",
    },
    "DEPTH": {
        "label": "Average Path Depth",
        "description": "Mean directory hierarchy depth (slashes in URL paths).",
        "type": "float",
        "min": 0.0,
        "max": 10.0,
        "default": 2.5,
        "unit": "path levels",
    },
    "STD_DEPTH": {
        "label": "Path Depth Std. Dev.",
        "description": "Standard deviation of URL directory path depth across requests.",
        "type": "float",
        "min": 0.0,
        "max": 5.0,
        "default": 0.65,
        "unit": "levels",
    },
    "CONSECUTIVE": {
        "label": "Consecutive Repeat Ratio",
        "description": "Proportion of back-to-back requests targeting the exact same URL.",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.08,
        "unit": "ratio [0-1]",
    },
    "DATA": {
        "label": "Total Data Transferred",
        "description": "Total bytes returned by the web server across all requests.",
        "type": "float",
        "min": 0.0,
        "max": 200000000.0,
        "default": 185000.0,
        "unit": "bytes",
    },
    "SF_REFERRER": {
        "label": "Referrer Present Ratio",
        "description": "Proportion of requests containing a non-empty HTTP Referrer header.",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.72,
        "unit": "ratio [0-1]",
    },
    "SF_FILETYPE": {
        "label": "Filetype Diversity Index",
        "description": "Ratio of distinct file extensions relative to unique visited pages.",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "default": 0.45,
        "unit": "diversity index",
    },
    "MAX_BARRAGE": {
        "label": "Maximum Barrage Burst",
        "description": "Peak number of requests sent within any single 1-second interval.",
        "type": "float",
        "min": 0.0,
        "max": 250.0,
        "default": 4.0,
        "unit": "req/sec",
    },
}


def validate_feature_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and casts a dictionary of user or batch inputs against the common schema.
    Returns cleaned dictionary with float/int types.
    Raises ValueError if required features are missing or cannot be converted.
    """
    cleaned = {}
    missing = []
    invalid = []

    for feat in COMMON_FEATURES:
        if feat not in data:
            missing.append(feat)
            continue
        val = data[feat]
        try:
            val_float = float(val)
            meta = FEATURE_METADATA.get(feat, {})
            min_val = meta.get("min", float("-inf"))
            max_val = meta.get("max", float("inf"))
            if val_float < min_val or val_float > max_val:
                invalid.append(f"{feat}: value {val_float} out of valid bounds [{min_val}, {max_val}]")
            cleaned[feat] = int(val_float) if meta.get("type") == "integer" else val_float
        except (ValueError, TypeError):
            invalid.append(f"{feat}: '{val}' is not a valid numerical value")

    if missing:
        raise ValueError(f"Missing required features: {missing}")
    if invalid:
        raise ValueError(f"Invalid feature values:\n" + "\n".join(invalid))

    return cleaned
