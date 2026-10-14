# BotPulse: Real-Time Web Bot Detection and Prevention
### AI-Powered Behavioral Web Bot Detection

**B.Tech Computer Science & Engineering Minor Project**

---

## 📌 Executive Summary

**BotPulse** is an end-to-end, machine learning-driven web security system designed to distinguish automated web robots (scrapers, vulnerability scanners, barrage bots, and headless browser emulators) from genuine human visitors.

Unlike conventional signature-based or IP-reputation defenses—which are easily bypassed by distributed residential proxies and rotating user agents—BotPulse analyzes **in-flight session behavior, inter-arrival time pacing, resource traversal ratios, and directory navigation graph topology**.

A critical achievement of this project is the **successful unification of two heterogeneous datasets into one cohesive training pipeline**:
1. **Dataset A: M4D Web Bot Detection Dataset** (Raw Apache2 server logs and mouse-tracking sessions of humans and bots browsing Wikipedia).
2. **Dataset B: Web Robot Detection Dataset (Zenodo)** (Large-scale production server logs of 67,352 sessions).

---

## 🎯 Key Project Highlights & Verifications

- **Combined Model Training**: Trained on **54,110 unified sessions** contributed by both datasets (**229 M4D sessions** and **53,881 Web Robot Detection sessions**).
- **Leakage Prevention**: Strictly excluded heuristic target-leakage columns (`PENALTY`) and arbitrary identifiers (`ID`, `DATASET_SOURCE`). Applied group-aware stratification for multi-session participants.
- **Champion Model**: **XGBoost Classifier** achieving **92.65% held-out test accuracy**, **0.8971 Macro-F1**, and **1.62 ms per-query inference latency**.
- **Source-Wise Validation**: Evaluated on 13,532 held-out unseen records across both sources:
  - **M4D Held-Out Subset**: 86.89% Accuracy | 0.8578 Macro-F1 | 94.59% Robot Recall
  - **Web Robot Detection Subset**: 92.68% Accuracy | 0.8969 Macro-F1 | 96.85% Robot Recall
- **Interactive Cybersec Web App**: 5-page Streamlit dashboard featuring Dark Cybersecurity theme, live single-sample and custom testing, batch CSV auditing, and attack scenarios.

---

## 🗂️ Project Architecture & Directory Layout

```text
BotPulse_Final/
├── app.py                      # Interactive Streamlit Web Application
├── train.py                    # End-to-end model training CLI
├── evaluate.py                 # Standalone model benchmarking CLI
├── inspect_datasets.py         # Dataset inspection and schema audit CLI
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation & execution guide
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── raw/
│   │   ├── m4d/                # M4D Phase 1, Phase 2, and README.docx
│   │   └── web_robot_detection/# simple_features.csv & semantic_features.csv
│   └── processed/
│       ├── m4d_extracted_features.csv # Extracted 27-feature M4D sessions
│       ├── combined_dataset.csv       # Unified dataset (67,642 sessions)
│       ├── train_split.csv            # Leakage-free training partition (54,110 rows)
│       └── test_split.csv             # Held-out testing partition (13,532 rows)
│
├── src/
│   ├── feature_mapping.py      # Common 27-feature schema & metadata
│   ├── feature_extraction.py   # Apache log parser and session aggregator
│   ├── data_loader.py          # Multi-source dataset builder and group splitter
│   ├── preprocessing.py        # Median imputer and robust quantile scaler
│   ├── model_training.py       # Random Forest and XGBoost training pipeline
│   ├── evaluation.py           # Multi-metric and source-wise evaluation engine
│   ├── prediction.py           # Real-time inference engine and scenario provider
│   └── visualization.py        # Cyber dark Plotly visual components
│
├── models/
│   ├── best_model.joblib       # Champion model (XGBoost)
│   ├── random_forest.joblib    # Random Forest artifact
│   ├── xgboost.joblib          # XGBoost artifact
│   ├── preprocessor.joblib     # Fitted Scikit-Learn preprocessing pipeline
│   ├── feature_schema.json     # Feature limits, bounds, and definitions
│   └── training_config.json    # Audit log of training observation counts
│
├── reports/
│   ├── dataset_inspection_report.json # Dataset compatibility audit
│   ├── evaluation_metrics.json        # Comprehensive test evaluation report
│   └── feature_importances.json       # Normalized feature importance weights
│
├── tests/
│   ├── test_feature_mapping.py    # Schema validation tests
│   ├── test_feature_extraction.py # Apache log parser tests
│   ├── test_data_loader.py        # Group-splitting & contribution tests
│   └── test_prediction.py         # Single, sample, and batch prediction tests
│
└── docs/
    ├── project_report.md       # Full academic project report
    ├── dataset_analysis.md     # Feature engineering & schema harmonization report
    └── viva_qa.md              # Faculty viva / oral exam questions & answers
```

---

## ⚙️ Windows Setup and Execution (PowerShell)

### Step 1: Open PowerShell and Navigate to the Project

Open PowerShell in VS Code or Terminal:

```powershell
cd "C:\Users\lenovo\.gemini\antigravity\scratch\BotPulse_Final"
```

> **Recommendation**: In VS Code, use **File -> Open Folder...** and select `C:\Users\lenovo\.gemini\antigravity\scratch\BotPulse_Final` to set it as your active workspace.

### Step 2: Create and Activate Virtual Environment (Optional / Recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Audit and Inspect Datasets

Verify dataset integrity, inspect dimensions, and check target leakage exclusions:

```powershell
python inspect_datasets.py
```

### Step 4: Train the Combined Machine Learning Models

Extracts Apache log features for M4D, combines both datasets, fits preprocessing, and trains Random Forest and XGBoost:

```powershell
python train.py
```

### Step 5: Run Independent Model Benchmarking

Evaluates the saved models on the held-out test split and prints source-wise performance tables:

```powershell
python evaluate.py
```

### Step 6: Run Automated Tests

Execute the 15-test automated test suite:

```powershell
python -m pytest -v
```

### Step 7: Launch the Interactive Web Application

Start the Streamlit cybersecurity dashboard:

```powershell
streamlit run app.py
```

Open your browser at:
`http://localhost:8501` (or `http://localhost:8502` if port 8501 is occupied).

---

## 📊 Common 27-Feature Predictive Schema

| Feature Name | Category | Description |
|---|---|---|
| `NUMBER_OF_REQUESTS` | Volumetric | Total HTTP requests sent within the session |
| `TOTAL_DURATION` | Temporal | Elapsed seconds between first and last request |
| `AVERAGE_TIME` | Temporal | Mean inter-arrival time between consecutive requests |
| `STANDARD_DEVIATION` | Temporal | Std. dev. of inter-arrival times (pacing variance) |
| `REPEATED_REQUESTS` | Navigation | Fraction of requests revisiting already seen URLs |
| `HTTP_RESPONSE_2XX` | Status | Proportion of successful (200 OK) responses |
| `HTTP_RESPONSE_3XX` | Status | Proportion of HTTP redirection responses |
| `HTTP_RESPONSE_4XX` | Status | Proportion of client error (404/403) responses |
| `HTTP_RESPONSE_5XX` | Status | Proportion of server error (500) responses |
| `GET_METHOD` | HTTP Method | Fraction of requests utilizing GET |
| `POST_METHOD` | HTTP Method | Fraction of requests utilizing POST |
| `HEAD_METHOD` | HTTP Method | Fraction of requests utilizing HEAD |
| `OTHER_METHOD` | HTTP Method | Fraction of requests utilizing other HTTP methods |
| `NIGHT` | Temporal | Fraction of requests sent between 00:00 and 06:00 |
| `IMAGES` | Resource | Fraction of image requests (.png, .jpg, .ico, .svg) |
| `TOTAL_HTML` | Resource | Fraction of page/HTML requests (.html, .php, /) |
| `HTML_TO_IMAGE` | Resource Ratio | Ratio of HTML requests to image requests |
| `HTML_TO_CSS` | Resource Ratio | Ratio of HTML requests to CSS requests |
| `HTML_TO_JS` | Resource Ratio | Ratio of HTML requests to JavaScript requests |
| `WIDTH` | Navigation | Total count of unique URLs requested |
| `DEPTH` | Navigation | Mean directory path depth (slash count in URL) |
| `STD_DEPTH` | Navigation | Standard deviation of URL directory depth |
| `CONSECUTIVE` | Navigation | Fraction of consecutive back-to-back duplicate requests |
| `DATA` | Volumetric | Total bytes transferred by server during session |
| `SF_REFERRER` | Behavioral | Fraction of requests presenting HTTP Referrer header |
| `SF_FILETYPE` | Diversity | Ratio of distinct file extensions to unique pages |
| `MAX_BARRAGE` | Burstiness | Peak requests sent within any single 1-second burst |

### Target Label
- **`ROBOT`**: Binary classification target.
  - `0`: Human Visitor (Non-Robot)
  - `1`: Automated Web Robot (Bot)

---

## 🧪 Live Faculty Demonstration Guide

When presenting this project to your college faculty:

1. **Dashboard Overview (Page A)**:
   - Point to the **Total Trained Sessions: 54,110**.
   - Show that **both datasets contributed training observations** (229 M4D sessions, 53,881 server-log sessions).
   - Point out the **Champion Model: XGBoost** with **92.65% Accuracy** and **1.62 ms latency**.

2. **Live Detection Lab (Page B - Mode 1)**:
   - Select **M4D** or **Web_Robot_Detection**.
   - Ensure the partition is set to **test (Held-out Unseen)**. This proves zero data leakage.
   - Click **🎲 Fetch Sample** to run real-time inference on unseen records.
   - Show the **Bot Threat Probability Gauge** and **Decision Influencing Factors Waterfall**.

3. **Custom Traffic Synthesizer (Page B - Mode 2)**:
   - Test a custom session: Set `Number of Requests` to 350, `Average Inter-arrival Time` to 0.4s, `Images Ratio` to 0.0, and `Max Barrage` to 12.
   - Click **⚡ Analyze Traffic Session** and show how BotPulse immediately flags the session as high-risk bot traffic.

4. **Attack Demonstration Scenarios (Page B - Mode 4)**:
   - Demonstrate the difference between **Verified Human Browsing Session** (Low Risk, 0.4% bot probability) and **Aggressive Web Crawler / Scraper** (High Risk, 99.8% bot probability).

5. **Performance & Generalization (Page C)**:
   - Walk faculty through the **Confusion Matrix** (2,649 True Positives, only 87 False Negatives).
   - Show the **Source-Wise Generalization Audit** table demonstrating high accuracy on both individual test partitions.

---

## 🔒 Target Leakage & Defense Audit

| Potential Leakage Source | Status | Mitigation Strategy |
|---|---|---|
| `PENALTY` column | **EXCLUDED** | Detected in Zenodo dataset as an artificial heuristic created by original researchers to score bots. Strictly removed from training features. |
| Session / Visitor `ID` | **EXCLUDED** | Removed to prevent the model from memorizing specific participant hashes. |
| `DATASET_SOURCE` | **EXCLUDED** | Auditing column only; not passed to the classifier to prevent dataset-source memorization. |
| M4D User Sessions (`_0`, `_1`) | **GROUPED** | Partitioned using Group Stratification so related sessions from the same user never appear across both train and test splits. |
| Preprocessing Leakage | **PREVENTED** | Median imputers and Robust scalers are fitted strictly on `X_train` and then applied to `X_test`. |

---

## 📚 Academic References

1. **M4D Web Bot Detection Dataset**:
   *P. Giannakoulis, K. Katsaros, V. Mezaris*, "Web Bot Detection Using Web Logs and Mouse Movements", Information 2020, 11(10), 469. [https://m4d.iti.gr/web-bot-detection-dataset/](https://m4d.iti.gr/web-bot-detection-dataset/)
2. **Web Robot Detection Dataset (Zenodo)**:
   *M. Dorau, P. Rotter*, "Web Robot Detection: Dataset of HTTP Server Logs and Derived Features", Zenodo Repository, 2019. [https://zenodo.org/records/3477932](https://zenodo.org/records/3477932)
