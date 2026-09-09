"""
CPRI State-Level Hackathon - Screening Round
Black-Box Test Bench Challenge - Master Solution Pipeline
============================================================

This single script performs, end-to-end, without any manual editing of
individual records:

  Task 1 : Identify abnormal / invalid records (rule-based + ML)
  Task 2 : Predict the Reference Parameter for test_data.csv
  Task 3 : Generate an automated test summary (summary.json)

Run:
    python src/pipeline.py

Outputs (written to ./outputs/):
    <TeamName>.csv   -> Test_ID, Predicted_Reference_Parameter, Validity_Label
    summary.json     -> automated summary required by Task 3
    cleaning_report.csv -> per-record diagnostics (audit trail, optional)

Edit the CONFIG block below to set your team name and tune thresholds.
Everything downstream is driven by that configuration - no per-record
hand edits are made anywhere in this script.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG - edit this block for your team
# --------------------------------------------------------------------------
TEAM_NAME = "TeamAlpha"                 # -> used to name the output CSV
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
TRAIN_FILE = DATA_DIR / "training_data.csv"
TEST_FILE = DATA_DIR / "test_data.csv"

RANDOM_STATE = 42
Z_SCORE_LIMIT = 4.0          # sensor spike threshold (robust z-score)
TOP_ATTENTION_N = 3          # how many Test IDs to flag for Task 3

FEATURE_COLS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]

OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Step 0 : Load
# --------------------------------------------------------------------------
def load_data():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    return train, test


# --------------------------------------------------------------------------
# Step 1 : Cleaning - duplicates & missing values
# --------------------------------------------------------------------------
def clean_frame(df, is_train):
    df = df.copy()

    # Remove exact duplicate rows (keep first occurrence)
    df["is_duplicate"] = df.duplicated(subset=["Test_ID"], keep="first")
    df = df[~df["is_duplicate"]].drop(columns=["is_duplicate"])

    # Also drop rows that are duplicated across ALL feature columns
    # (same physical test logged twice under different IDs)
    feat_dupe = df.duplicated(subset=FEATURE_COLS, keep="first")
    df["feature_duplicate_flag"] = feat_dupe

    # Track which numeric columns had missing values before imputation
    df["missing_count"] = df[FEATURE_COLS].isna().sum(axis=1)

    # Median-impute missing feature values (robust to outliers, reproducible)
    for col in FEATURE_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Step 2 : Rule-based abnormality checks
# --------------------------------------------------------------------------
def robust_z(series):
    median = series.median()
    mad = (series - median).abs().median()
    mad = mad if mad > 1e-9 else series.std() + 1e-9
    return 0.6745 * (series - median) / mad


def rule_based_flags(df):
    """
    Physically-motivated / statistical rules that catch sensor faults,
    corrupted values and logging errors WITHOUT penalising genuine
    operating-regime shifts (which show up as jointly-consistent,
    physically plausible combinations of high voltage/current/duration).
    """
    flags = pd.DataFrame(index=df.index)

    # (a) Negative or non-physical temperature-rise sensor readings
    for s in ["Sensor_S1", "Sensor_S2", "Sensor_S3"]:
        flags[f"{s}_negative"] = df[s] < 0

    # (b) Sensor spikes: robust (median/MAD) z-score outliers per sensor
    for s in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]:
        flags[f"{s}_spike"] = robust_z(df[s]).abs() > Z_SCORE_LIMIT

    # (c) Physically inconsistent sensor behaviour:
    #     S1/S2/S3 are temperature RISES at different points on the same
    #     specimen during the same test; they should move together and
    #     should not diverge wildly from each other for a given test.
    sensor_std = df[["Sensor_S1", "Sensor_S2", "Sensor_S3"]].std(axis=1)
    flags["sensor_incoherence"] = robust_z(sensor_std).abs() > Z_SCORE_LIMIT

    # (d) Out-of-range operating set-points (well beyond the observed
    #     envelope of the historical data => likely logging error)
    for col in ["Applied_Voltage_kV", "Load_Current_A",
                "Ambient_Temperature_C", "Test_Duration_min"]:
        lo, hi = df[col].quantile([0.001, 0.999])
        flags[f"{col}_out_of_range"] = (df[col] < lo) | (df[col] > hi)

    # (e) Excessive missing data on the original record
    flags["excessive_missing"] = df["missing_count"] >= 2

    # (f) Duplicated feature row (logging error / repeated test entry)
    flags["feature_duplicate"] = df["feature_duplicate_flag"]

    df["rule_flag_count"] = flags.sum(axis=1)
    df["rule_based_abnormal"] = df["rule_flag_count"] >= 1
    return df, flags


# --------------------------------------------------------------------------
# Step 3 : ML-based validity classifier (learns the engineers' judgement)
# --------------------------------------------------------------------------
def train_validity_classifier(train_df):
    y = (train_df["Validity_Label"].str.strip().str.lower() == "valid").astype(int)
    X = train_df[FEATURE_COLS + ["rule_flag_count"]]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    clf.fit(X_tr, y_tr)

    val_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred)
    print(f"[Validity classifier] hold-out accuracy={acc:.3f}  f1={f1:.3f}")

    # Refit on full training data for the final model
    clf.fit(X, y)
    return clf


# --------------------------------------------------------------------------
# Step 4 : Regression model for the Reference Parameter
#           (trained only on records the engineers/labels marked Valid,
#            so the model learns the true physical relationship rather
#            than being distorted by corrupted rows)
# --------------------------------------------------------------------------
def train_reference_regressor(train_df):
    valid_mask = train_df["Validity_Label"].str.strip().str.lower() == "valid"
    train_valid = train_df[valid_mask]

    X = train_valid[FEATURE_COLS]
    y = train_valid["Reference_Parameter"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    reg = RandomForestRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=2,
        random_state=RANDOM_STATE
    )
    reg.fit(X_tr, y_tr)

    val_pred = reg.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    r2 = r2_score(y_val, val_pred)
    print(f"[Reference regressor] hold-out MAE={mae:.3f}  R2={r2:.3f}")

    importances = pd.Series(reg.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False)
    print("[Feature importance]\n", importances.round(3).to_string())

    # Refit on the full set of valid training rows
    reg.fit(X, y)
    return reg, importances


# --------------------------------------------------------------------------
# Step 5 : Apply everything to test_data.csv
# --------------------------------------------------------------------------
def build_predictions(train_df, test_df, clf, reg):
    X_test = test_df[FEATURE_COLS + ["rule_flag_count"]]
    prob_valid = clf.predict_proba(X_test)[:, 1]
    ml_valid = prob_valid >= 0.5

    # Final validity decision: a record is Invalid if EITHER the rule
    # engine or the learned classifier flags it. This combines
    # interpretable physical rules with the model's learned judgement
    # of the historical engineers' labels.
    final_valid = ml_valid & (~test_df["rule_based_abnormal"])

    reg_X = test_df[FEATURE_COLS]
    pred_ref = reg.predict(reg_X)

    result = pd.DataFrame({
        "Test_ID": test_df["Test_ID"],
        "Predicted_Reference_Parameter": pred_ref.round(4),
        "Validity_Label": np.where(final_valid, "Valid", "Invalid"),
    })

    diagnostics = test_df.copy()
    diagnostics["ml_prob_valid"] = prob_valid.round(4)
    diagnostics["final_validity"] = result["Validity_Label"]
    diagnostics["predicted_reference_parameter"] = result["Predicted_Reference_Parameter"]

    return result, diagnostics


# --------------------------------------------------------------------------
# Step 6 : Task 3 - automated summary
# --------------------------------------------------------------------------
def build_summary(result, diagnostics, importances):
    n_total = len(result)
    n_invalid = int((result["Validity_Label"] == "Invalid").sum())

    preds = result["Predicted_Reference_Parameter"]

    # "Highest attention" = records that are Invalid AND have the
    # largest rule_flag_count / lowest ML confidence -> most in need
    # of manual engineering review.
    attention = diagnostics.copy()
    attention["attention_score"] = (
        attention["rule_flag_count"] * 2
        + (1 - attention["ml_prob_valid"])
    )
    top_attention = (
        attention.sort_values("attention_score", ascending=False)
        .head(TOP_ATTENTION_N)["Test_ID"]
        .tolist()
    )

    explanation = (
        "We cleaned duplicates and imputed missing values, then flagged "
        "abnormal records using physical rules (sensor spikes, negative "
        "or incoherent temperature rises, out-of-range set-points) "
        "combined with a Random Forest classifier trained on engineer "
        "labels. A separate Random Forest, trained only on Valid "
        "historical records, predicts the Reference Parameter from "
        "voltage, current, temperature, duration and sensors, "
        "distinguishing genuine regime shifts from faults."
    )
    # Enforce the 100-word limit
    words = explanation.split()
    if len(words) > 100:
        explanation = " ".join(words[:100])

    summary = {
        "team_name": TEAM_NAME,
        "records_analysed": n_total,
        "abnormal_invalid_records_identified": n_invalid,
        "predicted_reference_parameter": {
            "min": round(float(preds.min()), 4),
            "max": round(float(preds.max()), 4),
            "average": round(float(preds.mean()), 4),
        },
        "top_attention_test_ids": top_attention,
        "most_important_features": importances.round(3).to_dict(),
        "approach_summary_max_100_words": explanation,
    }
    return summary


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("Loading data ...")
    train_raw, test_raw = load_data()

    print("Cleaning (duplicates, missing values) ...")
    train = clean_frame(train_raw, is_train=True)
    test = clean_frame(test_raw, is_train=False)

    print("Applying rule-based abnormality checks ...")
    train, _ = rule_based_flags(train)
    test, _ = rule_based_flags(test)

    print("Training validity classifier ...")
    clf = train_validity_classifier(train)

    print("Training Reference Parameter regressor ...")
    reg, importances = train_reference_regressor(train)

    print("Scoring test_data.csv ...")
    result, diagnostics = build_predictions(train, test, clf, reg)

    print("Building automated summary ...")
    summary = build_summary(result, diagnostics, importances)

    # ---- Write deliverables ----
    out_csv = OUT_DIR / f"{TEAM_NAME}.csv"
    result.to_csv(out_csv, index=False)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    diagnostics.to_csv(OUT_DIR / "cleaning_report.csv", index=False)

    print(f"\nDone. Wrote:\n  {out_csv}\n  {OUT_DIR / 'summary.json'}\n  {OUT_DIR / 'cleaning_report.csv'}")
    print("\nSummary preview:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
