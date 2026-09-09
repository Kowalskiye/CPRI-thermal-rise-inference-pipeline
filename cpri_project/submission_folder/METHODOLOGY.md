# Methodology Note — <Your Team Name>
*(Template — edit every section before final submission. Keep to ≤2 pages when exported to PDF/Word.)*

## 1. Approach used

We built a single, reproducible Python pipeline (`src/pipeline.py`) that runs
end-to-end on `training_data.csv` and `test_data.csv` with no manual editing
of individual records:

1. **Cleaning** — exact-duplicate `Test_ID` rows are dropped; rows duplicated
   across all feature columns are flagged; missing feature values are
   median-imputed (robust to outliers, fully reproducible).
2. **Abnormality detection (Task 1)** — a rule engine flags physically or
   statistically implausible records (see §3), combined with a Random Forest
   classifier trained on the historical `Validity_Label` to learn the
   engineers' judgement beyond what fixed rules capture. A record is called
   `Invalid` if either the rules or the model flag it.
3. **Reference Parameter prediction (Task 2)** — a Random Forest Regressor is
   trained only on historical records labelled `Valid`, using
   `Applied_Voltage_kV, Load_Current_A, Ambient_Temperature_C,
   Test_Duration_min, Sensor_S1..S4` as inputs, then applied to every record
   in `test_data.csv`.
4. **Automated summary (Task 3)** — `summary.json` is generated directly by
   the script: record counts, invalid count, min/max/average predicted
   Reference Parameter, three highest-attention Test IDs, and this
   explanation text.

*(Replace/extend this section with whatever you actually changed —
different model family, feature engineering, ensembling, etc.)*

## 2. Parameters considered important

The baseline Random Forest's feature-importance ranking on the training data
(printed at runtime) is:

| Rank | Feature | Relative importance |
|---|---|---|
| 1 | Load_Current_A | highest |
| 2 | Sensor_S2 | moderate |
| 3 | Ambient_Temperature_C | moderate |
| 4–8 | Sensor_S1, Sensor_S3, Applied_Voltage_kV, Test_Duration_min, Sensor_S4 | low |

`Sensor_S4` (the auxiliary condition-monitoring sensor) showed the **lowest**
importance in the baseline model — investigate this explicitly (e.g. via
permutation importance or an ablation run) and state your conclusion here:
is it noise, or does it matter under specific regimes?

*(Fill in your team's actual findings — this table will change as you tune
the model and engineer new features.)*

## 3. Method used for detecting abnormal data

Rule-based checks (interpretable, applied identically to train and test):

- Negative or physically invalid sensor readings (S1–S3 are temperature
  rises and cannot be negative).
- Robust (median/MAD) z-score spike detection per sensor, threshold |z| > 4,
  chosen to be insensitive to the heavier tails introduced by genuine
  high-load operating regimes.
- Sensor incoherence: S1/S2/S3 are temperature rises measured on the same
  specimen during the same test and should move together; abnormally high
  spread across the three flags a likely sensor fault rather than a regime
  change (which affects all three sensors jointly).
- Out-of-range operating set-points (beyond the 0.1st/99.9th percentile of
  the historical envelope).
- Excess missing values on a single record (≥2 of 8 inputs missing).
- Duplicated feature rows (same test logged more than once).

This is deliberately layered with a supervised model (rather than rules
alone) because the brief warns that *"an unusual value is not necessarily an
invalid value"* — genuine regime changes are internally consistent across
voltage/current/duration/sensors, whereas faults tend to break that
consistency. The rules target the latter; the classifier learns any residual
pattern the engineers used that isn't captured by hand-written rules.

## 4. Assumptions made

- Historical `Validity_Label` values are ground truth for training purposes.
- Missing feature values are missing-at-random and can be reasonably
  represented by the column median (state here if your team used a different
  imputation strategy, e.g. KNN or regression imputation).
- The relationship between inputs and `Reference_Parameter` learned from
  `Valid` historical records generalizes to the new test conditions,
  including any regime shifts present in `test_data.csv`.
- No test-set record's true label was available for tuning; all thresholds
  (z-score limit, missing-value cutoff, out-of-range percentile) were chosen
  using only the training data / hold-out split.

## 5. Steps to automate this into a digital twin

1. **Ingestion layer** — stream or batch-load new test-bench readings
   (voltage, current, ambient temperature, duration, sensors S1–S4) directly
   from the data acquisition system into a structured store (e.g. a
   time-series database or a landing table), tagged with a unique Test_ID
   and timestamp.
2. **Automated cleaning & validation service** — run the same cleaning and
   rule-based flagging logic from `pipeline.py` as a scheduled/triggered job
   immediately after each test completes, rather than in batch.
3. **Online scoring** — apply the trained validity classifier and Reference
   Parameter regressor to each new record as it arrives, producing a
   prediction and a confidence/attention score in near real time.
4. **Drift monitoring** — track the distribution of inputs, rule-flag rates,
   and model confidence over time; trigger retraining when drift exceeds a
   defined threshold, and periodically re-validate against any newly
   engineer-verified records.
5. **Alerting & dashboarding** — surface high-attention / invalid records to
   test engineers through a dashboard (e.g. the `summary.json` structure
   rendered live), so genuine regime changes can be reviewed and confirmed
   rather than silently accepted or rejected.
6. **Feedback loop** — feed engineer confirmations/corrections back into the
   training set so the classifier and regressor improve over successive test
   campaigns, closing the loop between the physical test bench and its
   digital twin.

---
*Word/page count check: keep this file to ≤2 pages when exported. Trim
Sections 1–4 to your team's actual final approach before submission.*
