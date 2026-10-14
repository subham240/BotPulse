# BotPulse: Dataset Analysis & Feature Harmonization

---

## 1. Overview of Raw Datasets

### Dataset A: M4D Web Bot Detection Dataset
- **Provider**: Information Technologies Institute, Centre for Research and Technology Hellas (CERTH-ITI).
- **Format**:
  - Apache2 access logs (Combined Log Format + PHP Session ID).
  - Mouse movement JSON logs.
  - Annotation text files mapping Session IDs to `human`, `moderate_bot`, or `advanced_bot`.
- **Phases**:
  - Phase 1: 100 sessions across 61 Wikipedia articles.
  - Phase 2: 112 sessions across 110 Wikipedia articles.
- **Total Unique Sessions**: 290 annotated sessions (106 Human, 106 Advanced Bot, 78 Moderate Bot).

### Dataset B: Web Robot Detection (Zenodo)
- **Provider**: Zenodo Open Science Repository (Dorau & Rotter).
- **Files**:
  - `simple_features.csv`: 67,352 sessions x 32 columns.
  - `semantic_features.csv`: 67,352 sessions x 7 topic/semantic columns.
- **Target Distribution**: 53,858 Human (79.96%) and 13,494 Robot (20.04%).

---

## 2. Common Feature Harmonization Pipeline

To avoid forcing incompatible schemas or replacing features with arbitrary zeros, we implemented an Apache log feature extraction engine (`src/feature_extraction.py`).

Each log line was parsed using the regular expression:
```python
r'^(?:[^\s\[]+(?:\s+[^\s\[]+)*)?\s*\[(?P<time>[^\]]+)\]\s+"(?P<method>\S+)\s+(?P<url>\S+)(?:\s+(?P<proto>[^"]*))?"\s+(?P<status>\d+)\s+(?P<bytes>\S+)\s+"(?P<referrer>[^"]*)"\s+(?P<session>\S+)\s+"(?P<ua>[^"]*)"'
```

### Feature Engineering Formulas

1. **Session Volumetrics**:
   $$N = \text{len}(\text{requests})$$
   $$T = t_{\text{end}} - t_{\text{start}} \quad (\text{seconds})$$
   $$\bar{\Delta t} = \frac{T}{N - 1} \quad (\text{for } N > 1)$$

2. **Status Distribution**:
   $$R_{2xx} = \frac{\sum \mathbb{I}(\text{status} \in [200, 299])}{N}$$
   $$R_{4xx} = \frac{\sum \mathbb{I}(\text{status} \in [400, 499])}{N}$$

3. **Resource Ratios**:
   Images are detected via extensions `.png`, `.jpg`, `.jpeg`, `.gif`, `.ico`, `.svg`, `.webp`.
   $$R_{\text{img}} = \frac{N_{\text{images}}}{N}$$
   $$R_{\text{html}} = \frac{N_{\text{html}}}{N}$$
   $$\text{HTML\_TO\_IMAGE} = \frac{N_{\text{html}}}{\max(1, N_{\text{images}})}$$

4. **Navigational Topology**:
   $$W = |\text{unique}(URLs)|$$
   $$D = \frac{1}{N} \sum_{i=1}^N \text{count\_slashes}(URL_i)$$
   $$C = \frac{1}{N-1} \sum_{i=2}^N \mathbb{I}(URL_i == URL_{i-1})$$

5. **Burst Barrage**:
   $$\text{MAX\_BARRAGE} = \max_{s \in \text{seconds}} \text{requests\_in\_second}(s)$$

---

## 3. Target Leakage & Exclusions Audit

1. **`PENALTY`**:
   - In `simple_features.csv`, `PENALTY` represents an artificial heuristic score computed by the original authors to flag bots based on hardcoded rules.
   - Including `PENALTY` causes severe target leakage.
   - **Resolution**: Excluded from the predictive feature matrix.
2. **`ID`**:
   - Session identifiers are arbitrary strings (`obSnwGoBCue8G08E_WCX`, `97hf7ciplt2k54f5j6109nekn0`).
   - Retaining IDs causes tree algorithms to memorize session hashes.
   - **Resolution**: Excluded from predictive features.
3. **`DATASET_SOURCE`**:
   - Kept only as a metadata audit column for source-wise validation.
