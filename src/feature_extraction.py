"""
M4D Web Server Log Feature Extraction Pipeline.

Extracts compatible behavioral and traffic features from raw Apache2 web server logs
for the M4D Web Bot Detection Dataset (Phase 1 & Phase 2).
Produces a structured DataFrame conforming to the BotPulse common feature schema.
"""

import os
import re
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.feature_mapping import COMMON_FEATURES


# Regex for Apache2 combined format with appended PHP session ID
APACHE_LOG_PATTERN = re.compile(
    r'^(?:[^\s\[]+(?:\s+[^\s\[]+)*)?\s*\[(?P<time>[^\]]+)\]\s+'
    r'"(?P<method>\S+)\s+(?P<url>\S+)(?:\s+(?P<proto>[^"]*))?"\s+'
    r'(?P<status>\d+)\s+(?P<bytes>\S+)\s+'
    r'"(?P<referrer>[^"]*)"\s+(?P<session>\S+)\s+'
    r'"(?P<ua>[^"]*)"'
)


def parse_apache_timestamp(time_str: str) -> Optional[datetime]:
    """Parses Apache timestamp formatted like '01/Nov/2019:10:47:35 +0000'."""
    try:
        t_clean = time_str.split()[0]
        return datetime.strptime(t_clean, "%d/%b/%Y:%H:%M:%S")
    except Exception:
        return None


def collect_m4d_annotations(m4d_raw_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Parses all annotation files from Phase 1 and Phase 2.
    Returns dict mapping session_id -> {label: 'human'|'moderate_bot'|'advanced_bot', phase: str}
    """
    annotations: Dict[str, Dict[str, str]] = {}

    # Phase 1
    p1_base = os.path.join(m4d_raw_dir, "phase1", "annotations")
    if os.path.exists(p1_base):
        for sub in ["humans_and_moderate_bots", "humans_and_advanced_bots"]:
            for split in ["train", "test"]:
                p = os.path.join(p1_base, sub, split)
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                sid, lbl = parts[0], parts[1]
                                annotations[sid] = {
                                    "label": lbl,
                                    "phase": f"phase1_{sub}_{split}",
                                }

    # Phase 2
    p2_base = os.path.join(m4d_raw_dir, "phase2", "annotations")
    if os.path.exists(p2_base):
        for sub in os.listdir(p2_base):
            sub_dir = os.path.join(p2_base, sub)
            if os.path.isdir(sub_dir):
                for f in os.listdir(sub_dir):
                    p = os.path.join(sub_dir, f)
                    if os.path.isfile(p):
                        with open(p, "r", encoding="utf-8") as file:
                            for line in file:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    sid, lbl = parts[0], parts[1]
                                    annotations[sid] = {
                                        "label": lbl,
                                        "phase": f"phase2_{sub}",
                                    }

    return annotations


def extract_features_from_requests(requests: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extracts all 27 common features from a list of parsed HTTP requests for a single session.
    """
    if not requests:
        return None

    # Sort sequentially by timestamp
    requests = sorted(requests, key=lambda r: r["dt"])
    n = len(requests)
    if n == 0:
        return None

    t0 = requests[0]["dt"]
    t_end = requests[-1]["dt"]
    total_duration = max(0.0, float((t_end - t0).total_seconds()))

    # Inter-arrival times
    if n > 1:
        diffs = np.array([(requests[i]["dt"] - requests[i - 1]["dt"]).total_seconds() for i in range(1, n)])
        avg_time = float(np.mean(diffs))
        std_time = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    else:
        avg_time = 0.0
        std_time = 0.0

    # HTTP Status Code Distributions
    status_counts = Counter(r["status"] for r in requests)
    r_2xx = sum(cnt for s, cnt in status_counts.items() if s.startswith("2")) / n
    r_3xx = sum(cnt for s, cnt in status_counts.items() if s.startswith("3")) / n
    r_4xx = sum(cnt for s, cnt in status_counts.items() if s.startswith("4")) / n
    r_5xx = sum(cnt for s, cnt in status_counts.items() if s.startswith("5")) / n

    # HTTP Methods
    method_counts = Counter(r["method"].upper() for r in requests)
    get_m = method_counts.get("GET", 0) / n
    post_m = method_counts.get("POST", 0) / n
    head_m = method_counts.get("HEAD", 0) / n
    other_m = max(0.0, (n - (method_counts.get("GET", 0) + method_counts.get("POST", 0) + method_counts.get("HEAD", 0)))) / n

    # Night ratio (00:00 to 06:00 local server time)
    night_count = sum(1 for r in requests if 0 <= r["dt"].hour < 6)
    night_ratio = night_count / n

    # Resource categories from URLs
    images_ext = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp")
    css_ext = (".css",)
    js_ext = (".js",)
    html_ext = (".html", ".htm", ".php", "/")

    urls = [r["url"].split("?")[0].lower() for r in requests]

    img_cnt = sum(1 for u in urls if u.endswith(images_ext))
    css_cnt = sum(1 for u in urls if u.endswith(css_ext))
    js_cnt = sum(1 for u in urls if u.endswith(js_ext))
    html_cnt = sum(1 for u in urls if u.endswith(html_ext) or ("." not in u.split("/")[-1]))

    images_ratio = img_cnt / n
    html_ratio = html_cnt / n
    html_to_image = (html_cnt / img_cnt) if img_cnt > 0 else float(html_cnt)
    html_to_css = (html_cnt / css_cnt) if css_cnt > 0 else float(html_cnt)
    html_to_js = (html_cnt / js_cnt) if js_cnt > 0 else float(html_cnt)

    # Breadth and Depth
    unique_urls = set(urls)
    width = float(len(unique_urls))
    depths = [u.count("/") for u in urls]
    depth = float(np.mean(depths))
    std_depth = float(np.std(depths))

    # Consecutive identical requests
    consec = sum(1 for i in range(1, n) if urls[i] == urls[i - 1]) / n if n > 1 else 0.0

    # Repeated requests (visited earlier in session)
    seen = set()
    repeated = 0
    for u in urls:
        if u in seen:
            repeated += 1
        seen.add(u)
    repeated_ratio = repeated / n

    # Total bytes transferred
    total_bytes = float(sum(int(r["bytes"]) for r in requests if r["bytes"].isdigit()))

    # Referrer presence
    sf_referrer = sum(1 for r in requests if r["referrer"] not in ("-", "")) / n

    # Filetype diversity index
    exts = set(u.split(".")[-1] for u in urls if "." in u.split("/")[-1])
    sf_filetype = len(exts) / max(1.0, width)

    # Maximum barrage burst in any 1-second interval
    sec_counts = Counter(r["dt"] for r in requests)
    max_barrage = float(max(sec_counts.values())) if sec_counts else 1.0

    return {
        "NUMBER_OF_REQUESTS": n,
        "TOTAL_DURATION": total_duration,
        "AVERAGE_TIME": avg_time,
        "STANDARD_DEVIATION": std_time,
        "REPEATED_REQUESTS": repeated_ratio,
        "HTTP_RESPONSE_2XX": r_2xx,
        "HTTP_RESPONSE_3XX": r_3xx,
        "HTTP_RESPONSE_4XX": r_4xx,
        "HTTP_RESPONSE_5XX": r_5xx,
        "GET_METHOD": get_m,
        "POST_METHOD": post_m,
        "HEAD_METHOD": head_m,
        "OTHER_METHOD": other_m,
        "NIGHT": night_ratio,
        "IMAGES": images_ratio,
        "TOTAL_HTML": html_ratio,
        "HTML_TO_IMAGE": min(120.0, float(html_to_image)),
        "HTML_TO_CSS": min(243.0, float(html_to_css)),
        "HTML_TO_JS": min(21.0, float(html_to_js)),
        "WIDTH": width,
        "DEPTH": depth,
        "STD_DEPTH": std_depth,
        "CONSECUTIVE": consec,
        "DATA": total_bytes,
        "SF_REFERRER": sf_referrer,
        "SF_FILETYPE": min(1.0, sf_filetype),
        "MAX_BARRAGE": max_barrage,
    }


def process_m4d_dataset(
    m4d_raw_dir: str,
    output_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Main extraction pipeline for M4D dataset:
    1. Collects annotations
    2. Parses Apache log files
    3. Groups requests by session
    4. Computes common schema features
    5. Saves output CSV if path is provided
    Returns DataFrame of extracted M4D sessions.
    """
    print(f"[M4D Extractor] Collecting annotations from: {m4d_raw_dir}")
    annotations = collect_m4d_annotations(m4d_raw_dir)
    print(f"[M4D Extractor] Found {len(annotations)} annotated sessions.")

    # Find all log files
    log_files: List[str] = []
    for root, _, files in os.walk(m4d_raw_dir):
        for f in files:
            if f.endswith(".log"):
                log_files.append(os.path.join(root, f))

    print(f"[M4D Extractor] Scanning {len(log_files)} web log files...")
    session_requests = defaultdict(list)
    total_lines = 0
    matched_lines = 0

    for lf in log_files:
        with open(lf, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total_lines += 1
                m = APACHE_LOG_PATTERN.match(line.strip())
                if m:
                    matched_lines += 1
                    sid = m.group("session")
                    if sid != "-":
                        d = m.groupdict()
                        dt = parse_apache_timestamp(d["time"])
                        if dt:
                            d["dt"] = dt
                            session_requests[sid].append(d)

    print(f"[M4D Extractor] Parsed {matched_lines}/{total_lines} lines into {len(session_requests)} distinct session tokens.")

    # Match annotated sessions to requests
    records = []
    for sid, ann_info in annotations.items():
        reqs = None
        if sid in session_requests:
            reqs = session_requests[sid]
        elif sid.split("_")[0] in session_requests:
            reqs = session_requests[sid.split("_")[0]]

        if reqs:
            feats = extract_features_from_requests(reqs)
            if feats:
                lbl = ann_info["label"]
                # Binary mapping: 'human' -> 0, any bot ('moderate_bot', 'advanced_bot') -> 1
                is_bot = 0 if lbl == "human" else 1
                feats["ID"] = sid
                feats["ROBOT"] = is_bot
                feats["ORIGINAL_LABEL"] = lbl
                feats["M4D_DETAILED_LABEL"] = lbl
                feats["DATASET_SOURCE"] = "M4D"
                records.append(feats)

    df_m4d = pd.DataFrame(records)
    print(f"[M4D Extractor] Successfully extracted {len(df_m4d)} sessions. Target counts: {dict(df_m4d['ROBOT'].value_counts())}")

    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        df_m4d.to_csv(output_csv_path, index=False)
        print(f"[M4D Extractor] Saved extracted features to: {output_csv_path}")

    return df_m4d
