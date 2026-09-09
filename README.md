<a name="top"></a>
# CPRI Hackathon — Black-Box Test Bench Challenge

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Submission%20Ready-success)
![Team](https://img.shields.io/badge/Team-KDG-orange)
![Hackathon](https://img.shields.io/badge/Hackathon-PowerNext--AI-purple)

This is the master project for the CPRI State-Level Hackathon screening round ("The Black-Box Test Bench Challenge"). It contains a working, end-to-end automated solution for all three tasks, finalized and prepared for submission by Team KDG.

---

## 📑 Table of Contents
- [1. Problem Statement](#problem-statement)
- [2. Our Approach](#our-approach)
- [3. What's Inside](#whats-inside)
- [4. What the Pipeline Does](#pipeline)
- [5. Running the Project](#running)
- [6. Design Principles](#design)
- [7. Troubleshooting](#troubleshooting)

---

<a name="problem-statement"></a>
## 1. Problem Statement [🔝](#top)

The objective of this challenge is to analyze data from a CPRI laboratory electrical test bench. The system records input parameters (voltage, current, temperature, duration) and output sensors (S1-S4) during repeated tests. The dataset is noisy, containing missing values, duplicates, sensor errors, and abnormal test runs. 

The goal is to:
1. **Identify Abnormal Records**: Distinguish between genuine changes in the equipment's operating regime and erroneous measurements (sensor faults, corrupted data).
2. **Predict the Reference Parameter**: Discover the hidden mathematical relationship between the input parameters and a verified Reference Parameter, predicting it for new unseen tests.
3. **Automate the Process**: Create an end-to-end automated pipeline that generates predictions and a summary report without any manual intervention.

---

<a name="our-approach"></a>
## 2. Our Approach [🔝](#top)

Team KDG developed a robust, two-stage machine learning pipeline to solve the challenge:

- **Data Cleaning & Imputation**: We handle missing data using median imputation (robust to outliers) and systematically drop exact duplicate records to ensure data integrity.
- **Hybrid Abnormality Detection**: We combine physically motivated rules (e.g., negative temperature rises, robust z-score spike detection) with a Random Forest classifier trained on the historical `Validity_Label`. This ensures both known physical impossibilities and complex learned patterns are flagged as invalid.
- **Physics-Informed Regression**: A Random Forest Regressor is trained to predict the Reference Parameter. Crucially, we train this model *only on Valid historical records* to prevent corrupted data from distorting the learned physical relationship.
- **Fully Automated Pipeline**: The entire workflow runs via a single script (`pipeline.py`), automatically producing the required deliverables (`KDG.csv` and `summary.json`) ready for a production Digital Twin environment.

---

<a name="whats-inside"></a>
## 3. What's inside [🔝](#top)

```text
cpri_project/
├── README.md                 <- this file
├── METHODOLOGY.md            <- Detailed methodology note (Deliverable 4)
├── data/
│   ├── training_data.csv     <- Historical records
│   └── test_data.csv         <- New records to be scored
├── src/
│   ├── pipeline.py           <- MAIN script: Tasks 1 + 2 + 3, fully automated
│   └── requirements.txt      <- Dependencies
├── notebook/
│   └── solution.ipynb        <- Executable notebook version
└── outputs/                  <- Auto-generated on run
    ├── KDG.csv               <- Task 1 + 2 deliverable
    ├── summary.json          <- Task 3 deliverable
    └── cleaning_report.csv   <- Audit trail
```

---

<a name="pipeline"></a>
## 4. What the baseline pipeline already does [🔝](#top)

| Task | Approach used in `pipeline.py` |
|------|----------------------------------|
| **1. Abnormal record detection** | Rule-based checks (negative/incoherent sensor readings, robust z-score spike detection, out-of-range set-points, excess missing values, duplicate rows) **combined with** a Random Forest classifier trained on the historical `Validity_Label` column. A record is called `Invalid` if either fires. |
| **2. Reference Parameter prediction** | A Random Forest Regressor trained **only on historical `Valid` records** (so corrupted rows don't distort the learned physics), using all 4 operating parameters + 4 sensors as features. |
| **3. Automated summary** | `summary.json` — records analysed, invalid count, min/max/avg predicted Reference Parameter, top-3 "highest attention" Test IDs, and an auto-generated ≤100-word explanation. |

---

<a name="running"></a>
## 5. Running it in Antigravity IDE [🔝](#top)

[Antigravity](https://antigravity.google/) is Google's agentic, AI-native IDE.

### Step-by-step

1. **Let Antigravity set up the Python environment**
   Open the integrated terminal and run:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r src/requirements.txt
   ```

2. **Run the full pipeline**
   From the integrated terminal:
   ```bash
   python src/pipeline.py
   ```
   Alternatively, open `notebook/solution.ipynb` and **Run All**.

3. **Check the outputs**
   After the run finishes, look in `outputs/`:
   - `KDG.csv` — the prediction deliverable
   - `summary.json` — the automated Task-3 summary
   - `cleaning_report.csv` — per-record diagnostics

---

<a name="design"></a>
## 6. Design principles baked into this scaffold [🔝](#top)

- **Fully reproducible / no manual record edits.** Every decision (imputation, rule thresholds, model training) is a function of the data.
- **Genuine regime change ≠ fault.** The rule engine flags *statistical and physical* anomalies, not merely "unusual" values.
- **Separation of concerns.** Cleaning, rule-based flagging, ML classification, regression and summary generation are separate functions.
- **Deterministic.** `RANDOM_STATE = 42` everywhere a model or split is used.

---

<a name="troubleshooting"></a>
## 7. Troubleshooting [🔝](#top)

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: sklearn` | Run `pip install -r src/requirements.txt` inside the active environment. |
| Numbers change slightly between runs | Ensure `RANDOM_STATE = 42` hasn't been removed; the pipeline is otherwise deterministic. |
| Notebook can't find `pipeline` module | Run the first cell (`sys.path.insert(...)`) before any other cell. |
