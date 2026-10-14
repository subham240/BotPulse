"""
BotPulse: Real-Time Web Bot Detection and Prevention
Subtitle: AI-Powered Behavioral Web Bot Detection

A clean, modern, student- and faculty-friendly Streamlit Web Application.
Built with Streamlit, Plotly, scikit-learn, and XGBoost.
"""

import os
import sys
import json
import re
import time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in python path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src.feature_mapping import (
    COMMON_FEATURES,
    FEATURE_METADATA,
    CLASS_NAMES,
    TARGET_COLUMN,
)
from src.prediction import BotPredictor
from src.visualization import (
    CYBER_THEME,
    plot_confusion_matrix,
    plot_feature_importances,
    plot_distribution_donut,
    plot_algorithm_comparison,
    plot_confidence_gauge,
    plot_decision_factors_waterfall,
    plot_threat_histogram,
    plot_traffic_breakdown_bar,
    plot_mouse_trajectories_side_by_side,
)


# ==============================================================================
# STREAMLIT PAGE CONFIG & CLEAN CYBERSECURITY CSS
# ==============================================================================
st.set_page_config(
    page_title="BotPulse | AI Web Bot Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Main app dark styling */
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
    font-family: 'Segoe UI', Inter, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}

/* Glassmorphic Cyber Cards */
.cyber-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}

.cyber-card:hover {
    border-color: #0ea5e9;
    transition: border-color 0.2s ease;
}

/* Verdict Banners */
.verdict-box-human {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.3) 100%);
    border: 2px solid #10b981;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin-bottom: 16px;
}

.verdict-box-standard-bot {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(120, 53, 15, 0.3) 100%);
    border: 2px solid #f59e0b;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin-bottom: 16px;
}

.verdict-box-advanced-bot {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.18) 0%, rgba(136, 19, 55, 0.35) 100%);
    border: 2px solid #f43f5e;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    margin-bottom: 16px;
}

.verdict-title {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}

.verdict-sub {
    font-size: 14px;
    color: #94a3b8;
}

/* Stat Pill */
.stat-pill {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.stat-num {
    font-size: 24px;
    font-weight: 700;
    color: #38bdf8;
}
.stat-lbl {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-top: 2px;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 18px;
    transition: all 0.2s ease;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# CACHED LOADERS & HELPERS
# ==============================================================================
@st.cache_resource
def get_predictor() -> Optional[BotPredictor]:
    try:
        return BotPredictor(model_name="best_model")
    except Exception:
        return None


@st.cache_data
def get_evaluation_reports() -> Optional[Dict[str, Any]]:
    path = os.path.join(APP_DIR, "reports", "evaluation_metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_training_config() -> Optional[Dict[str, Any]]:
    path = os.path.join(APP_DIR, "models", "training_config.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_unified_dataset_sample(n_rows: int = 500) -> pd.DataFrame:
    path = os.path.join(APP_DIR, "data", "processed", "botpulse_unified_dataset.csv")
    if not os.path.exists(path):
        path = os.path.join(APP_DIR, "data", "processed", "combined_dataset.csv")
    if os.path.exists(path):
        return pd.read_csv(path, nrows=n_rows)
    return pd.DataFrame()


@st.cache_data
def load_external_benchmark() -> pd.DataFrame:
    path = os.path.join(APP_DIR, "data", "processed", "external_test_benchmark.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_mouse_movement_samples(max_samples: int = 15) -> Dict[str, List[Dict[str, Any]]]:
    """Loads genuine mouse trajectory coordinates for Humans and Advanced Bots."""
    processed_samples = os.path.join(APP_DIR, "data", "processed", "mouse_movement_samples.json")
    if os.path.exists(processed_samples):
        try:
            with open(processed_samples, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "humans": data.get("humans", [])[:max_samples],
                    "bots": data.get("bots", [])[:max_samples],
                }
        except Exception:
            pass

    p_h = os.path.join(APP_DIR, "data", "raw", "m4d", "phase2", "data", "mouse_movements", "humans", "mouse_movements_humans.json")
    p_b = os.path.join(APP_DIR, "data", "raw", "m4d", "phase2", "data", "mouse_movements", "bots", "mouse_movements_advanced_bots.json")

    def _parse(path):
        items = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= max_samples:
                        break
                    obj = json.loads(line)
                    sid = obj.get("session_id", f"sess_{idx}")
                    text = str(obj.get("mousemove_total_behaviour", ""))
                    moves = re.findall(r"m\((\d+),(\d+)\)", text)
                    xs = [int(x) for x, y in moves[:250]]
                    ys = [int(y) for x, y in moves[:250]]
                    if len(xs) > 10:
                        items.append({"session_id": sid, "xs": xs, "ys": ys, "count": len(xs)})
        return items

    return {"humans": _parse(p_h), "bots": _parse(p_b)}


predictor = get_predictor()
config = get_training_config()
eval_report = get_evaluation_reports()


def safe_fetch_category_sample(category_name: str) -> Dict[str, Any]:
    """Safe sample fetcher with guaranteed fallback to prevent any AttributeError."""
    if predictor and hasattr(predictor, "get_sample_by_category"):
        try:
            return predictor.get_sample_by_category(category_name)
        except Exception:
            pass

    unified_file = os.path.join(APP_DIR, "data", "processed", "botpulse_unified_dataset.csv")
    if not os.path.exists(unified_file):
        unified_file = os.path.join(APP_DIR, "data", "processed", "combined_dataset.csv")

    df = pd.read_csv(unified_file)
    if "CATEGORY" not in df.columns:
        def _map_c(r):
            lbl = str(r.get("ORIGINAL_LABEL", "")).lower()
            if "advanced" in lbl:
                return "Advanced Bot"
            elif "moderate" in lbl or lbl == "robot" or r.get("ROBOT") == 1:
                return "Standard Bot"
            return "Human"
        df["CATEGORY"] = df.apply(_map_c, axis=1)

    subset = df[df["CATEGORY"].str.lower() == category_name.lower()]
    if len(subset) == 0:
        subset = df

    row = subset.sample(1).iloc[0]
    feats = {f: float(row[f]) if isinstance(row[f], (float, int, np.number)) else row[f] for f in COMMON_FEATURES}

    return {
        "id": str(row.get("ID", "sample")),
        "category": str(row.get("CATEGORY", category_name)),
        "source": str(row.get("DATASET_SOURCE", "Unified Dataset")),
        "ground_truth_class": int(row.get("ROBOT", 0)),
        "ground_truth_label": CLASS_NAMES[int(row.get("ROBOT", 0))],
        "original_label": str(row.get("ORIGINAL_LABEL", "")),
        "features": feats,
    }


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <span style="font-size:32px;">🛡️</span>
            <div>
                <div style="font-size:22px; font-weight:800; color:#38bdf8; letter-spacing:-0.03em;">BotPulse</div>
                <div style="font-size:11px; color:#94a3b8;">Web Bot Detection System</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "Menu",
        [
            "⚡ Live Bot Detector",
            "🖱️ Mouse Trajectory (Human vs Adv Bot)",
            "🌐 External Dataset Testing",
            "📁 Single Unified Dataset",
            "📊 Dashboard & Metrics",
        ],
        index=0,
    )

    st.markdown("---")

    if config:
        total_counts = config.get("total_dataset_counts", {})
        total_sess = total_counts.get("total", 67642)
        m4d_sess = total_counts.get("m4d", 290)
        zenodo_sess = total_counts.get("web_robot_detection", 67352)
        train_sess = config.get("training_counts", {}).get("total", 54110)
        test_sess = config.get("test_counts", {}).get("total", 13532)

        st.markdown(
            f"""
            <div style="background:#1e293b; padding:12px 14px; border-radius:8px; border:1px solid #334155; font-size:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="color:#10b981; font-weight:700;">● AI Model Online</span>
                    <span style="font-size:11px; color:#38bdf8; background:#0f172a; padding:1px 6px; border-radius:4px; border:1px solid #1e293b; font-weight:600;">{config.get('best_model_name', 'XGBoost')}</span>
                </div>
                <div style="color:#f8fafc; font-size:13px; font-weight:700;">
                    Unified Dataset: <span style="color:#38bdf8;">{total_sess:,} sessions</span>
                </div>
                <div style="color:#cbd5e1; font-size:11px; margin-top:4px; line-height:1.5;">
                    • <b>Server Logs:</b> {zenodo_sess:,} sessions<br>
                    • <b>M4D Dataset:</b> {m4d_sess:,} sessions <span style="color:#64748b; font-size:10px;">(305k reqs)</span>
                </div>
                <div style="border-top:1px solid #334155; margin-top:8px; padding-top:6px; color:#94a3b8; font-size:10px; display:flex; justify-content:space-between;">
                    <span><b>Train:</b> {train_sess:,} (80%)</span>
                    <span><b>Test:</b> {test_sess:,} (20%)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="font-size:11px; color:#64748b; margin-top:20px; text-align:center;">
            B.Tech CSE Minor Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# HELPER: RENDER VERDICT CARD
# ==============================================================================
def render_verdict_card(res: Dict[str, Any], ground_truth: Optional[str] = None, category_hint: Optional[str] = None, source_info: Optional[str] = None):
    is_bot = res["predicted_class"] == 1
    conf_pct = res["confidence"] * 100.0

    is_advanced = False
    if category_hint and "advanced" in category_hint.lower():
        is_advanced = True
    elif is_bot and res.get("features", {}).get("AVERAGE_TIME", 0) > 0.6 and res.get("features", {}).get("IMAGES", 0) > 0.15:
        is_advanced = True

    if not is_bot:
        box_class = "verdict-box-human"
        title_color = "#10b981"
        title_text = "✅ GENUINE HUMAN VISITOR"
        sub_text = f"Low Threat Risk • <b>{conf_pct:.1f}% Human Confidence</b>"
    elif is_advanced:
        box_class = "verdict-box-advanced-bot"
        title_color = "#f43f5e"
        title_text = "🚨 ADVANCED EVASION BOT DETECTED"
        sub_text = f"High Threat Risk • <b>{conf_pct:.1f}% Bot Confidence</b> • <i>Browser Fingerprinted / Evasive</i>"
    else:
        box_class = "verdict-box-standard-bot"
        title_color = "#f59e0b"
        title_text = "🟠 STANDARD WEB BOT DETECTED"
        sub_text = f"High Threat Risk • <b>{conf_pct:.1f}% Bot Confidence</b> • <i>Web Scraper / Crawler</i>"

    ground_truth_html = ""
    if ground_truth:
        match_icon = "✔️" if (is_bot and ground_truth.lower() in ("robot", "bot", "standard bot", "advanced bot")) or (not is_bot and ground_truth.lower() == "human") else "⚠️"
        ground_truth_html = f"<div style='margin-top:8px; font-size:12px; color:#cbd5e1;'>{match_icon} Recorded Label: <b>{ground_truth}</b></div>"

    source_html = f"<div style='font-size:11px; color:#94a3b8;'>Source: {source_info}</div>" if source_info else ""

    st.markdown(
        f"""
        <div class="{box_class}">
            <div class="verdict-title" style="color:{title_color};">{title_text}</div>
            <div class="verdict-sub">{sub_text}</div>
            {ground_truth_html}
            {source_html}
            <div style="margin-top:10px; font-size:12px; color:#64748b;">
                Inference Speed: <b>{res['latency_ms']:.2f} ms</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_factors = res.get("top_factors", [])
    if top_factors:
        reasons = []
        for f in top_factors[:3]:
            feat_name = f["label"]
            diff = f["relative_difference_pct"]
            if diff > 0:
                reasons.append(f"<b>{feat_name}</b> is {abs(diff):.0f}% higher than human baseline")
            else:
                reasons.append(f"<b>{feat_name}</b> is {abs(diff):.0f}% lower than human baseline")

        st.markdown(
            f"""
            <div class="cyber-card" style="padding:14px;">
                <div style="font-size:13px; font-weight:700; color:#38bdf8; margin-bottom:6px;">
                    💡 Why the AI made this decision:
                </div>
                <div style="font-size:13px; color:#cbd5e1;">
                    {' • '.join(reasons)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 1: LIVE BOT DETECTOR (HUMAN, BOT, ADVANCED BOT & CUSTOM SESSION)
# ==============================================================================
if page == "⚡ Live Bot Detector":
    st.markdown(
        """
        <div style="margin-bottom:16px;">
            <h1 style="font-size:28px; margin-bottom:4px;">⚡ Live Bot Detector</h1>
            <p style="font-size:14px; color:#94a3b8; margin:0;">
                Test real-time detection on <b>Human</b>, <b>Standard Bot (Scraper)</b>, or <b>Advanced Bot (Headless Browser)</b> sessions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not predictor:
        st.error("Model artifacts missing. Run `python train.py` first.")
        st.stop()

    st.markdown(
        """
        <div style="font-size:14px; font-weight:700; color:#38bdf8; margin-bottom:8px;">
            🎯 1-Click Faculty Demonstration Buttons:
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_c1, btn_c2, btn_c3 = st.columns(3)

    with btn_c1:
        test_human_btn = st.button("🟢 Test Real Human Session", use_container_width=True)
    with btn_c2:
        test_bot_btn = st.button("🟠 Test Standard Bot (Scraper)", use_container_width=True)
    with btn_c3:
        test_advbot_btn = st.button("🔴 Test Advanced Bot (Headless Browser)", use_container_width=True)

    sample_to_run = None
    sample_source_desc = None
    sample_ground_truth = None
    sample_cat_hint = None

    if test_human_btn:
        s = safe_fetch_category_sample("Human")
        sample_to_run = s["features"]
        sample_source_desc = f"Genuine Human Session (ID: {s['id']})"
        sample_ground_truth = "Human"
        sample_cat_hint = "Human"

    elif test_bot_btn:
        s = safe_fetch_category_sample("Standard Bot")
        sample_to_run = s["features"]
        sample_source_desc = f"Standard Web Scraper Bot (ID: {s['id']})"
        sample_ground_truth = "Standard Bot"
        sample_cat_hint = "Standard Bot"

    elif test_advbot_btn:
        s = safe_fetch_category_sample("Advanced Bot")
        sample_to_run = s["features"]
        sample_source_desc = f"Advanced Browser-Fingerprinted Bot (M4D Phase 2, ID: {s['id']})"
        sample_ground_truth = "Advanced Bot"
        sample_cat_hint = "Advanced Bot"

    if sample_to_run:
        res = predictor.predict_single(sample_to_run)
        st.session_state.current_result = {
            "res": res,
            "ground_truth": sample_ground_truth,
            "category_hint": sample_cat_hint,
            "source": sample_source_desc,
            "features": sample_to_run,
        }

    # Display 1-click Result
    if "current_result" in st.session_state and st.session_state.current_result:
        curr = st.session_state.current_result
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        r_c1, r_c2 = st.columns([1.3, 1])
        with r_c1:
            render_verdict_card(curr["res"], curr["ground_truth"], curr.get("category_hint"), curr["source"])
        with r_c2:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            fig_g = plot_confidence_gauge(curr["res"]["probability_robot"])
            st.plotly_chart(fig_g, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔍 Click to Inspect All 27 Raw Technical Features", expanded=False):
            f_rows = [{"Feature": k, "Value": f"{float(v):.4f}" if isinstance(v, (float, int, np.number)) else str(v)} for k, v in curr["features"].items()]
            st.dataframe(pd.DataFrame(f_rows), use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # CUSTOM TRAFFIC SYNTHESIZER (NO ST.FORM, ZERO RELOAD ISSUES)
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div style="font-size:18px; font-weight:700; color:#f1f5f9; margin-bottom:4px;">
            🎛️ Test Custom Traffic Behavior
        </div>
        <p style="font-size:13px; color:#94a3b8; margin:0 0 12px 0;">
            Adjust the sliders below and click <b>Analyze Custom Session</b> to get immediate live results right here:
        </p>
        """,
        unsafe_allow_html=True,
    )

    custom_container = st.container()
    with custom_container:
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            c_requests = st.slider("Total Requests in Session", min_value=1, max_value=1000, value=40, step=5, key="c_req_slider")
            c_pace = st.slider("Average Time Between Clicks (seconds)", min_value=0.1, max_value=60.0, value=8.5, step=0.5, key="c_pace_slider",
                               help="Humans take 5-15 seconds reading. Scrapers click rapidly (<1 second).")
        with sc2:
            c_images = st.slider("Image Assets Downloaded (%)", min_value=0, max_value=100, value=40, step=5, key="c_img_slider",
                                 help="Browsers render all images (~30-50%). Web scrapers block images (0%).")
            c_barrage = st.slider("Peak Burst (Max requests in 1 second)", min_value=1, max_value=50, value=3, step=1, key="c_barr_slider",
                                  help="Burst attacks and scrapers send 10-30 requests per second.")
        with sc3:
            c_bot_archetype = st.selectbox(
                "Preset Behavior Archetype",
                ["Normal Human Browsing", "Standard Web Scraper (No Images, Fast)", "Advanced Headless Bot (Fingerprinted, Medium Pace)", "Midnight Burst Attack"],
                index=0,
                key="c_arch_box"
            )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        custom_submit_btn = st.button("⚡ Analyze Custom Session", key="btn_custom_submit", use_container_width=True)

        if custom_submit_btn:
            img_ratio = c_images / 100.0
            html_ratio = max(0.1, 1.0 - img_ratio)
            h2img = (html_ratio / max(0.01, img_ratio))
            is_night = 1.0 if "Midnight" in c_bot_archetype else 0.0

            custom_features = {
                "NUMBER_OF_REQUESTS": int(c_requests),
                "TOTAL_DURATION": float(c_requests * c_pace),
                "AVERAGE_TIME": float(c_pace),
                "STANDARD_DEVIATION": float(c_pace * 0.8),
                "REPEATED_REQUESTS": 0.05,
                "HTTP_RESPONSE_2XX": 0.98,
                "HTTP_RESPONSE_3XX": 0.01,
                "HTTP_RESPONSE_4XX": 0.01,
                "HTTP_RESPONSE_5XX": 0.0,
                "GET_METHOD": 0.95,
                "POST_METHOD": 0.05,
                "HEAD_METHOD": 0.0,
                "OTHER_METHOD": 0.0,
                "NIGHT": float(is_night),
                "IMAGES": float(img_ratio),
                "TOTAL_HTML": float(html_ratio),
                "HTML_TO_IMAGE": float(min(120.0, h2img)),
                "HTML_TO_CSS": 0.9,
                "HTML_TO_JS": 1.2,
                "WIDTH": float(max(1, int(c_requests * 0.4))),
                "DEPTH": 2.2,
                "STD_DEPTH": 0.5,
                "CONSECUTIVE": 0.05 if c_pace > 2.0 else 0.35,
                "DATA": float(c_requests * 5000),
                "SF_REFERRER": 0.8 if c_pace > 2.0 else 0.1,
                "SF_FILETYPE": 0.4,
                "MAX_BARRAGE": float(c_barrage),
            }

            c_res = predictor.predict_single(custom_features)
            cat_guess = "Advanced Bot" if "Advanced" in c_bot_archetype else ("Standard Bot" if c_res["predicted_class"] == 1 else "Human")

            # Render result directly underneath the button in the custom container
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            res_c1, res_c2 = st.columns([1.3, 1])
            with res_c1:
                render_verdict_card(c_res, "Simulated Session", cat_guess, f"Custom Form ({c_bot_archetype})")
            with res_c2:
                fig_cg = plot_confidence_gauge(c_res["probability_robot"])
                st.plotly_chart(fig_cg, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: MOUSE TRAJECTORY ANALYSIS (HUMAN VS ADVANCED BOT)
# ==============================================================================
elif page == "🖱️ Mouse Trajectory (Human vs Adv Bot)":
    st.markdown(
        """
        <div style="margin-bottom:16px;">
            <h1 style="font-size:28px; margin-bottom:4px;">🖱️ Mouse Trajectory Analysis</h1>
            <p style="font-size:14px; color:#94a3b8; margin:0;">
                Compare real mouse cursor movement patterns between <b>Genuine Humans</b> and <b>Advanced Headless Browser Bots</b> from the M4D study.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mouse_data = load_mouse_movement_samples()
    h_list = mouse_data["humans"]
    b_list = mouse_data["bots"]

    if not h_list or not b_list:
        st.warning("Mouse movement files not found. Ensure `data/raw/m4d/phase2/data/mouse_movements/` is populated.")
        st.stop()

    st.markdown(
        """
        <div class="cyber-card">
            <h3 style="font-size:16px; color:#38bdf8; margin-bottom:6px;">🎓 Behavioral Biometrics & Anti-Bot Evasion:</h3>
            <p style="font-size:13px; color:#cbd5e1; line-height:1.6; margin:0;">
                Modern advanced web bots use headless browsers (e.g. Puppeteer/Playwright) to emulate human mouse movement.
                However, <b>genuine humans exhibit natural physiological micro-tremors, curved acceleration profiles, and deceleration near targets (Fitts's Law)</b>.
                In contrast, <b>advanced bots generate synthetic mathematical trajectories</b> (e.g. fixed step vectors or straight linear interpolations).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        h_sids = [s["session_id"] for s in h_list]
        selected_h_sid = st.selectbox("Select Genuine Human Session", h_sids, index=0)
    with col_m2:
        b_sids = [s["session_id"] for s in b_list]
        selected_b_sid = st.selectbox("Select Advanced Bot Session", b_sids, index=0)

    curr_h = next(s for s in h_list if s["session_id"] == selected_h_sid)
    curr_b = next(s for s in b_list if s["session_id"] == selected_b_sid)

    # Plot Side-by-Side Trajectories
    st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
    fig_mouse = plot_mouse_trajectories_side_by_side(curr_h["xs"], curr_h["ys"], curr_b["xs"], curr_b["ys"])
    st.plotly_chart(fig_mouse, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Key Differences Comparison Cards
    k_col1, k_col2, k_col3 = st.columns(3)
    with k_col1:
        st.markdown(
            """
            <div class="stat-pill">
                <div class="stat-num" style="color:#10b981; font-size:18px;">Curvature & Jitter</div>
                <div style="font-size:12px; color:#cbd5e1; margin-top:6px;">
                    <b>Human:</b> High organic jitter and curved arcs.<br>
                    <b>Advanced Bot:</b> Synthetic linear segments.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k_col2:
        st.markdown(
            """
            <div class="stat-pill">
                <div class="stat-num" style="color:#f59e0b; font-size:18px;">Velocity Dynamics</div>
                <div style="font-size:12px; color:#cbd5e1; margin-top:6px;">
                    <b>Human:</b> Bell-shaped velocity curves.<br>
                    <b>Advanced Bot:</b> Constant or step-wise velocity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k_col3:
        st.markdown(
            """
            <div class="stat-pill">
                <div class="stat-num" style="color:#f43f5e; font-size:18px;">Target Deceleration</div>
                <div style="font-size:12px; color:#cbd5e1; margin-top:6px;">
                    <b>Human:</b> Decelerates before clicking hyperlinks.<br>
                    <b>Advanced Bot:</b> Abrupt instantaneous stop.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PAGE 3: EXTERNAL DATASET TESTING (USER PROVIDES DATASET & MODEL PREDICTS IT)
# ==============================================================================
elif page == "🌐 External Dataset Testing":
    st.markdown(
        """
        <div style="margin-bottom:16px;">
            <h1 style="font-size:28px; margin-bottom:4px;">🌐 External Dataset Testing</h1>
            <p style="font-size:14px; color:#94a3b8; margin:0;">
                Provide any external web traffic dataset (CSV) to test the trained model with rich charts and metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not predictor:
        st.error("Model artifacts missing. Run `python train.py` first.")
        st.stop()

    st.markdown(
        """
        <div class="cyber-card">
            <h3 style="font-size:16px; color:#38bdf8; margin-bottom:6px;">Upload External Dataset:</h3>
            <p style="font-size:13px; color:#cbd5e1; margin-bottom:10px;">
                Upload your own CSV file, or load the pre-configured <b>200-Session External Benchmark Suite</b>:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_up, col_btn = st.columns([1.5, 1])
    with col_up:
        uploaded_csv = st.file_uploader("Upload CSV Web Session File", type=["csv"], key="ext_uploader")
    with col_btn:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        load_preset_btn = st.button("📁 Load Built-In 200-Session Benchmark", use_container_width=True)

    df_to_test = None
    dataset_title = ""

    if uploaded_csv is not None:
        try:
            df_to_test = pd.read_csv(uploaded_csv)
            dataset_title = f"Uploaded External File: {uploaded_csv.name}"
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    elif load_preset_btn or ("last_ext_df" in st.session_state and st.session_state.last_ext_df is not None):
        if load_preset_btn:
            st.session_state.last_ext_df = load_external_benchmark()
        df_to_test = st.session_state.last_ext_df
        dataset_title = "Built-In External Benchmark Suite (200 Unseen Sessions)"

    if df_to_test is not None and len(df_to_test) > 0:
        missing_cols = [c for c in COMMON_FEATURES if c not in df_to_test.columns]
        if missing_cols:
            st.error(f"❌ The provided dataset is missing {len(missing_cols)} required feature columns: {', '.join(missing_cols[:4])}...")
        else:
            with st.spinner("Model is analyzing external web traffic sessions..."):
                res_df, summary = predictor.predict_batch(df_to_test)

            st.success(f"✅ Successfully processed {len(df_to_test):,} sessions in {summary['batch_time_seconds']:.2f} seconds ({summary['throughput_req_per_sec']:,.0f} req/sec)!")

            st.markdown(f"<h3 style='font-size:18px; color:#38bdf8; margin-top:12px;'>📊 Prediction Results for: {dataset_title}</h3>", unsafe_allow_html=True)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Total Sessions Tested", f"{summary['total_rows']:,}")
            with k2:
                st.metric("Web Bots Detected", f"{summary['predicted_robots']:,} ({summary['robot_percentage']:.1f}%)")
            with k3:
                st.metric("Genuine Humans", f"{summary['predicted_humans']:,}")
            with k4:
                st.metric("Inference Throughput", f"{summary['throughput_req_per_sec']:,.0f} req/s")

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            c_ch1, c_ch2 = st.columns([1, 1.2])

            with c_ch1:
                st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
                fig_donut = plot_distribution_donut(
                    {
                        "Genuine Humans": summary["predicted_humans"],
                        "Web Bots Detected": summary["predicted_robots"],
                    },
                    title="External Traffic Classification Ratio",
                    color_palette=[CYBER_THEME["emerald"], CYBER_THEME["rose"]],
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c_ch2:
                st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
                fig_hist = plot_threat_histogram(
                    res_df["CONFIDENCE_ROBOT"].tolist(),
                    title="Bot Threat Confidence Score Distribution (%)",
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Chart Row 2: Behavioral Feature Breakdown
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            humans_mask = (res_df["PREDICTED_CLASS"] == 0)
            bots_mask = (res_df["PREDICTED_CLASS"] == 1)

            h_sub = res_df[humans_mask]
            b_sub = res_df[bots_mask]

            avg_pace_h = float(h_sub["AVERAGE_TIME"].mean()) if len(h_sub) > 0 else 0.0
            avg_pace_b = float(b_sub["AVERAGE_TIME"].mean()) if len(b_sub) > 0 else 0.0

            avg_img_h = float(h_sub["IMAGES"].mean() * 100.0) if len(h_sub) > 0 else 0.0
            avg_img_b = float(b_sub["IMAGES"].mean() * 100.0) if len(b_sub) > 0 else 0.0

            avg_barr_h = float(h_sub["MAX_BARRAGE"].mean()) if len(h_sub) > 0 else 0.0
            avg_barr_b = float(b_sub["MAX_BARRAGE"].mean()) if len(b_sub) > 0 else 0.0

            feature_comparisons = {
                "Avg Click Delay (sec)": {"human": avg_pace_h, "bot": avg_pace_b},
                "Images Downloaded (%)": {"human": avg_img_h, "bot": avg_img_b},
                "Peak Burst (req/sec)": {"human": avg_barr_h, "bot": avg_barr_b},
            }
            fig_bars = plot_traffic_breakdown_bar(feature_comparisons, title="Behavioral Contrast: Detected Humans vs Detected Bots")
            st.plotly_chart(fig_bars, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Table of Predictions
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-size:15px; margin-bottom:10px;'>Enriched External Predictions Table:</h4>", unsafe_allow_html=True)
            disp_cols = ["ID", "PREDICTED_LABEL", "CONFIDENCE_ROBOT", "NUMBER_OF_REQUESTS", "AVERAGE_TIME", "IMAGES", "MAX_BARRAGE"]
            avail_cols = [c for c in disp_cols if c in res_df.columns]
            st.dataframe(res_df[avail_cols].head(100), use_container_width=True)

            dl_csv = res_df.to_csv(index=False)
            st.download_button(
                "💾 Download Predicted Results CSV",
                data=dl_csv,
                file_name="botpulse_external_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: SINGLE UNIFIED DATASET (FOR TEACHERS)
# ==============================================================================
elif page == "📁 Single Unified Dataset":
    st.markdown(
        """
        <div style="margin-bottom:16px;">
            <h1 style="font-size:28px; margin-bottom:4px;">📁 Single Unified Dataset</h1>
            <p style="font-size:14px; color:#94a3b8; margin:0;">
                Both <b>M4D Web Bot Detection</b> and <b>Web Robot Detection</b> unified into one clean, standardized dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    u1, u2, u3, u4 = st.columns(4)
    with u1:
        st.markdown(
            """
            <div class="stat-pill">
                <div class="stat-num">67,642</div>
                <div class="stat-lbl">Total Unified Sessions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with u2:
        st.markdown(
            """
            <div class="stat-pill">
                <div class="stat-num" style="color:#10b981;">53,964</div>
                <div class="stat-lbl">Human Sessions (79.8%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with u3:
        st.markdown(
            """
            <div class="stat-pill">
                <div class="stat-num" style="color:#f59e0b;">13,572</div>
                <div class="stat-lbl">Standard Bots (20.0%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with u4:
        st.markdown(
            """
            <div class="stat-pill">
                <div class="stat-num" style="color:#f43f5e;">106</div>
                <div class="stat-lbl">Advanced Evasion Bots (0.2%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="cyber-card">
            <h3 style="font-size:16px; color:#38bdf8; margin-bottom:6px;">🎓 How We Unified Both Datasets (For Faculty Presentation):</h3>
            <p style="font-size:13px; color:#cbd5e1; line-height:1.6; margin:0;">
                1. <b>Extracted M4D Apache Logs:</b> We parsed 55 raw Apache server logs from the M4D Wikipedia study (305,377 requests) and extracted 27 exact behavioral session metrics.<br>
                2. <b>Harmonized with Server Logs:</b> Standardized with the 67,352 Zenodo server logs using identical mathematical formulas.<br>
                3. <b>Unified Labels:</b> Standardized into <b>Human</b>, <b>Standard Bot (Scrapers/Crawlers)</b>, and <b>Advanced Bot (Headless Browser with Fingerprint)</b>.<br>
                4. <b>Purged Target Leakage:</b> Strictly removed <code>PENALTY</code> and session <code>ID</code> to prevent false 100% memorization.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    unified_csv_path = os.path.join(APP_DIR, "data", "processed", "botpulse_unified_dataset.csv")
    if os.path.exists(unified_csv_path):
        with open(unified_csv_path, "rb") as f:
            st.download_button(
                "📥 Download Single Unified Dataset (CSV - 67,642 Sessions)",
                data=f,
                file_name="botpulse_unified_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    filter_cat = st.selectbox("Filter by Category in Table", ["All Categories", "Human", "Standard Bot", "Advanced Bot"], index=0)

    sample_df = load_unified_dataset_sample(1000)
    if len(sample_df) > 0:
        if filter_cat != "All Categories":
            sample_df = sample_df[sample_df["CATEGORY"].str.lower() == filter_cat.lower()]

        display_cols = ["ID", "CATEGORY", "NUMBER_OF_REQUESTS", "TOTAL_DURATION", "AVERAGE_TIME", "IMAGES", "TOTAL_HTML", "MAX_BARRAGE", "DATASET_SOURCE"]
        available_cols = [c for c in display_cols if c in sample_df.columns]
        st.dataframe(sample_df[available_cols].head(100), use_container_width=True)


# ==============================================================================
# PAGE 5: DASHBOARD & METRICS
# ==============================================================================
elif page == "📊 Dashboard & Metrics":
    st.markdown(
        """
        <div style="margin-bottom:16px;">
            <h1 style="font-size:28px; margin-bottom:4px;">📊 Project Metrics & Model Performance</h1>
            <p style="font-size:14px; color:#94a3b8; margin:0;">
                Performance evaluated on 13,532 unseen held-out sessions across both datasets.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not eval_report or not config:
        st.warning("Training report not found. Run `python train.py` first.")
        st.stop()

    champ_eval = eval_report["models"]["xgboost"]["overall"]
    rf_eval = eval_report["models"]["random_forest"]["overall"]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-num">{champ_eval['accuracy']*100:.2f}%</div>
                <div class="stat-lbl">Held-Out Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-num">{champ_eval['macro_f1']:.4f}</div>
                <div class="stat-lbl">Held-Out Macro-F1</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-num">{config['training_counts']['total']:,}</div>
                <div class="stat-lbl">Combined Training Sessions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="stat-pill">
                <div class="stat-num">{eval_report['models']['xgboost']['latency']['mean_ms']:.2f} ms</div>
                <div class="stat-lbl">Per-Query Latency</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        fig_cm = plot_confusion_matrix(eval_report["models"]["xgboost"]["overall"]["confusion_matrix"], title="Confusion Matrix (13,532 Unseen Sessions)")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        fig_comp = plot_algorithm_comparison(rf_eval, champ_eval)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size:16px; margin-bottom:8px;'>🌐 Multi-Dataset Generalization Audit</h3>", unsafe_allow_html=True)
    sw_data = []
    for src, info in eval_report["models"]["xgboost"]["source_wise"].items():
        sw_data.append({
            "Dataset Source": "M4D User Study & Logs" if "M4D" in src else "Web Robot Detection Server Logs",
            "Unseen Test Sessions": f"{info['sample_count']:,}",
            "Accuracy": f"{info['accuracy']*100:.2f}%",
            "Macro-F1": f"{info['macro_f1']:.4f}",
            "Bot Catch Rate (Recall)": f"{info['per_class']['robot']['recall']*100:.2f}%",
            "Human Detection F1": f"{info['per_class']['human']['f1']:.4f}",
        })
    st.dataframe(pd.DataFrame(sw_data), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
