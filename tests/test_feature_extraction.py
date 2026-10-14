"""Unit tests for M4D log parsing and feature extraction."""

from datetime import datetime
from src.feature_extraction import (
    parse_apache_timestamp,
    extract_features_from_requests,
    APACHE_LOG_PATTERN,
)
from src.feature_mapping import COMMON_FEATURES


def test_parse_apache_timestamp():
    ts = parse_apache_timestamp("01/Nov/2019:10:47:35 +0000")
    assert isinstance(ts, datetime)
    assert ts.year == 2019
    assert ts.month == 11
    assert ts.day == 1
    assert ts.hour == 10
    assert ts.minute == 47
    assert ts.second == 35


def test_apache_log_pattern_matching():
    sample_line = '- - [01/Nov/2019:10:47:35 +0000] "GET /css/main.css HTTP/1.1" 200 735 "http://160.40.52.164/" 97hf7ciplt2k54f5j6109nekn0 "Mozilla/5.0 Firefox/70.0"'
    m = APACHE_LOG_PATTERN.match(sample_line)
    assert m is not None
    d = m.groupdict()
    assert d["method"] == "GET"
    assert d["url"] == "/css/main.css"
    assert d["status"] == "200"
    assert d["bytes"] == "735"
    assert d["session"] == "97hf7ciplt2k54f5j6109nekn0"


def test_extract_features_from_requests():
    reqs = [
        {
            "dt": datetime(2019, 11, 1, 10, 0, 0),
            "method": "GET",
            "url": "/index.php",
            "status": "200",
            "bytes": "1500",
            "referrer": "-",
        },
        {
            "dt": datetime(2019, 11, 1, 10, 0, 5),
            "method": "GET",
            "url": "/img/logo.png",
            "status": "200",
            "bytes": "25000",
            "referrer": "http://example.com/",
        },
        {
            "dt": datetime(2019, 11, 1, 10, 0, 10),
            "method": "POST",
            "url": "/api/action",
            "status": "200",
            "bytes": "400",
            "referrer": "http://example.com/",
        },
    ]

    feats = extract_features_from_requests(reqs)
    assert feats is not None
    assert feats["NUMBER_OF_REQUESTS"] == 3
    assert feats["TOTAL_DURATION"] == 10.0
    assert feats["AVERAGE_TIME"] == 5.0
    assert feats["HTTP_RESPONSE_2XX"] == 1.0
    assert feats["POST_METHOD"] == 1 / 3
    assert feats["SF_REFERRER"] == 2 / 3
    for f in COMMON_FEATURES:
        assert f in feats
