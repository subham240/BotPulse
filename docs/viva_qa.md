# BotPulse: Faculty Viva & Oral Examination Q&A Guide

Prepare for your B.Tech CSE minor project defense with these key technical answers.

---

### Q1: What is the main objective of BotPulse?
**Answer:**
BotPulse detects automated web bots in real-time by analyzing HTTP server-log behavioral patterns rather than easily spoofed client identifiers (like IP address or User-Agent). It distinguishes genuine human browsing from scrapers, crawlers, and headless browser bots using a 27-feature behavioral schema.

---

### Q2: How did you use BOTH datasets in ONE trained model?
**Answer:**
We unified the **M4D Web Bot Detection Dataset** and the **Zenodo Web Robot Detection Dataset**:
1. We identified that both datasets reflect web server session traffic.
2. The Zenodo dataset provided structured session features (`simple_features.csv`).
3. For M4D, we developed an Apache log parser (`src/feature_extraction.py`) that read 55 raw Apache access logs (305,377 requests) and extracted the exact equivalent 27 features for each of the 290 annotated sessions.
4. We mapped labels into a shared binary task: Class 0 = Human, Class 1 = Bot (both moderate bots and advanced headless bots).
5. The combined training set contains **54,110 sessions**, with **229 contributed by M4D** and **53,881 contributed by Web Robot Detection**.

---

### Q3: What is "Target Leakage", and how did you prevent it?
**Answer:**
Target leakage occurs when a feature contains information that would not be available at prediction time or directly encodes the target label:
- In the Zenodo dataset, the authors created a column called `PENALTY` using a rule-based formula designed to identify bots. Including `PENALTY` would cause artificial 100% accuracy that fails on real-world data. We strictly identified and **excluded `PENALTY`**.
- We also excluded arbitrary session IDs (`ID`) and dataset source tags (`DATASET_SOURCE`) so the model learns behavioral dynamics rather than participant identifiers.
- For users with multiple recorded sessions in M4D, we used **group-aware splitting** so no user appeared in both train and test partitions.

---

### Q4: Why did you compare Random Forest and XGBoost?
**Answer:**
Both are state-of-the-art tree ensemble algorithms well-suited for tabular data:
- **Random Forest** uses bagging (bootstrap aggregation) with unpruned trees to reduce variance.
- **XGBoost** uses gradient boosting, building sequential trees that correct prior residual errors.
- **Outcome**: XGBoost achieved superior Macro-F1 (**0.8971** vs 0.8963) and dramatically faster real-time inference latency (**1.62 ms** vs 75.52 ms), making it our champion model for production deployment.

---

### Q5: How do you verify the model works on unseen data?
**Answer:**
We evaluated on a held-out test split of **13,532 unseen sessions**:
- Total Held-out Accuracy: **92.65%**
- Held-out Macro-F1: **0.8971**
- On M4D held-out test sessions: **86.89% Accuracy** and **94.59% Bot Recall**
- On Web Robot Detection test sessions: **92.68% Accuracy** and **96.85% Bot Recall**
In the web application's **Live Bot Detection Lab**, faculty can select "test (Held-out Unseen)" to test predictions on records the model never saw during training.

---

### Q6: What are the most important predictive features?
**Answer:**
Ensemble split gain analysis reveals the top discriminative features:
1. `IMAGES`: Automated scrapers rarely request or render image assets.
2. `SF_REFERRER`: Web bots frequently omit or falsify the HTTP Referer header.
3. `MAX_BARRAGE`: Automated tools send bursts of requests in a single second far exceeding human capability.
4. `TOTAL_DURATION` & `AVERAGE_TIME`: Rapid, machine-driven pacing without reading pauses.
5. `WIDTH` & `DEPTH`: Deep recursive crawling patterns vs shallow human reading.

---

### Q7: What are the project's limitations?
**Answer:**
1. Server logs cannot capture client-side DOM events or mouse velocities directly unless combined with JavaScript telemetry.
2. Extremely sophisticated headless browser bots with humanized random delays can partially mimic human timing unless multi-modal mouse features are enabled.
3. Offline log analysis is designed for upstream ingestion or reverse proxy session scoring rather than immediate Layer 3 packet dropping.
