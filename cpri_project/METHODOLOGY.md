# Methodology Note — Team KDG

## 1. Approach Used

We built a single, fully automated Python pipeline (`src/pipeline.py`) that processes both `training_data.csv` and `test_data.csv` end-to-end with no manual editing of individual records. The pipeline performs four stages:

1. **Data Cleaning** — Exact-duplicate `Test_ID` rows are removed (keeping the first occurrence). Rows duplicated across all feature columns are flagged as potential re-logged tests. Missing numeric values are median-imputed, chosen for robustness to outliers and full reproducibility.

2. **Abnormality Detection (Task 1)** — A rule-based engine flags physically or statistically implausible records (detailed in Section 3). This is combined with a Random Forest classifier (400 trees, balanced class weights) trained on the historical `Validity_Label` column. A record is marked `Invalid` if either the rules or the ML model flag it, ensuring both interpretable physical checks and learned patterns from engineer judgement are captured.

3. **Reference Parameter Prediction (Task 2)** — A Random Forest Regressor (500 trees) is trained exclusively on records labelled `Valid` in the training set. This prevents corrupted or anomalous records from distorting the learned relationship. The model uses all eight input features: `Applied_Voltage_kV`, `Load_Current_A`, `Ambient_Temperature_C`, `Test_Duration_min`, and `Sensor_S1` through `Sensor_S4`.

4. **Automated Summary (Task 3)** — The script generates `summary.json` containing record counts, invalid counts, predicted Reference Parameter statistics, the three highest-attention Test IDs, feature importances, and a ≤100-word approach summary.

## 2. Parameters Considered Important

Feature importance analysis from the trained Random Forest Regressor reveals:

| Rank | Feature | Relative Importance |
|------|---------|---------------------|
| 1 | Load_Current_A | ~0.903 (dominant) |
| 2 | Sensor_S2 | ~0.057 |
| 3 | Ambient_Temperature_C | ~0.032 |
| 4 | Sensor_S1 | ~0.002 |
| 5 | Sensor_S3 | ~0.002 |
| 6 | Applied_Voltage_kV | ~0.001 |
| 7 | Test_Duration_min | ~0.001 |
| 8 | Sensor_S4 | ~0.001 |

`Load_Current_A` overwhelmingly drives the Reference Parameter, consistent with resistive heating physics (I²R). `Sensor_S2` provides the strongest secondary signal among the temperature-rise sensors. `Sensor_S4` (the auxiliary condition-monitoring sensor) contributes negligible predictive power — it appears to be largely irrelevant for Reference Parameter estimation, though it is retained for robustness.

## 3. Method Used for Detecting Abnormal Data

Our rule-based engine applies the following physically motivated checks identically to both training and test data:

- **Negative sensor readings**: Temperature-rise sensors S1–S3 cannot physically be negative; any negative value indicates a sensor fault.
- **Robust z-score spike detection**: For each sensor (S1–S4), robust z-scores are computed using median and MAD (Median Absolute Deviation). Values exceeding |z| > 4.0 are flagged as spikes. This threshold was chosen to be insensitive to genuine high-load operating regimes.
- **Sensor incoherence**: S1, S2, and S3 measure temperature rise at different points on the same specimen during the same test. Abnormally high standard deviation across the three indicates a sensor fault rather than a regime change (which would affect all three consistently).
- **Out-of-range set-points**: Operating parameters beyond the 0.1st/99.9th percentile of the historical distribution are flagged as likely logging errors.
- **Excessive missing data**: Records with ≥2 of 8 input features originally missing are flagged.
- **Duplicated feature rows**: Identical feature vectors under different Test IDs indicate data-logging errors.

This rule engine is layered with a supervised Random Forest classifier to capture patterns the engineers used that fixed rules cannot express. The key distinction maintained throughout is: *genuine operating regime shifts produce jointly consistent changes across voltage, current, duration, and all sensors, while faults break that consistency.*

## 4. Assumptions Made

- Historical `Validity_Label` values provided by CPRI engineers are treated as ground truth for supervised training.
- Missing feature values are missing at random and adequately represented by column medians.
- The physical relationship between inputs and `Reference_Parameter` learned from `Valid` training records generalizes to test conditions, including any regime shifts in `test_data.csv`.
- No test-set labels were used for tuning; all thresholds were determined from the training data or hold-out splits only.
- `RANDOM_STATE = 42` ensures full reproducibility across runs.

## 5. Steps to Automate Through a Digital Twin

1. **Data Ingestion Layer** — Connect the data acquisition system to a structured store (e.g., time-series database). Each test record is tagged with a unique Test_ID and timestamp upon completion.

2. **Automated Cleaning & Validation Service** — Deploy the cleaning and rule-based flagging logic as a triggered microservice that runs automatically after each test completes, rather than in batch.

3. **Online Scoring Engine** — Apply the trained validity classifier and Reference Parameter regressor to each new record in near real-time, producing predictions and confidence/attention scores immediately.

4. **Model Drift Monitoring** — Continuously track input distributions, rule-flag rates, and model confidence. Trigger automated retraining when statistical drift exceeds defined thresholds, and periodically re-validate against newly engineer-verified records.

5. **Alerting & Dashboard** — Surface high-attention and invalid records to test engineers through a live dashboard (rendering the `summary.json` structure in real time), enabling prompt review of flagged tests.

6. **Feedback Loop** — Feed engineer confirmations and corrections back into the training set, enabling the classifier and regressor to improve over successive test campaigns — closing the loop between the physical test bench and its digital twin.

---
*Team KDG — CPRI State-Level Hackathon, PowerNext-AI 2026*
