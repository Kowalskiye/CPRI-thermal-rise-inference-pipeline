# CPRI Hackathon — Black-Box Test Bench Challenge
### Master Solution Scaffold (ready to deploy in Antigravity IDE)

This is a **basic master project skeleton** for the CPRI State-Level Hackathon
screening round ("The Black-Box Test Bench Challenge"). It already contains a
working, end-to-end baseline solution for all three tasks — you extend it, you
don't start from scratch.

---

## 1. What's inside

```
cpri_project/
├── README.md                 <- this file
├── METHODOLOGY.md             <- 2-page methodology note (edit before submission)
├── data/
│   ├── training_data.csv      <- ~1000 historical records (with Reference_Parameter + Validity_Label)
│   └── test_data.csv          <- ~350 new records to be scored (no ground truth)
├── src/
│   ├── pipeline.py             <- MAIN script: Tasks 1 + 2 + 3, fully automated
│   └── requirements.txt
├── notebook/
│   └── solution.ipynb          <- thin notebook wrapper around pipeline.py (for the "executable notebook" deliverable)
└── outputs/                    <- generated on every run (git-ignored)
    ├── <TeamName>.csv          <- Task 1 + 2 deliverable
    ├── summary.json            <- Task 3 deliverable
    └── cleaning_report.csv     <- audit trail (which rule fired on which record — optional but useful for your methodology note)
```

## 2. What the baseline pipeline already does

| Task | Approach used in `pipeline.py` |
|------|----------------------------------|
| **1. Abnormal record detection** | Rule-based checks (negative/incoherent sensor readings, robust z-score spike detection, out-of-range set-points, excess missing values, duplicate rows) **combined with** a Random Forest classifier trained on the historical `Validity_Label` column. A record is called `Invalid` if either fires. |
| **2. Reference Parameter prediction** | A Random Forest Regressor trained **only on historical `Valid` records** (so corrupted rows don't distort the learned physics), using all 4 operating parameters + 4 sensors as features. Feature importances are printed so you can see which sensors actually matter (spoiler: check whether `Sensor_S4` earns its place). |
| **3. Automated summary** | `summary.json` — records analysed, invalid count, min/max/avg predicted Reference Parameter, top-3 "highest attention" Test IDs (highest combined rule-flag + lowest ML confidence), and an auto-generated ≤100-word explanation. |

Baseline hold-out performance on the training split (printed when you run it):
`Validity classifier ≈ 0.90 accuracy / 0.95 F1`, `Reference regressor ≈ R² 0.98`.
**This is a starting point, not your final answer** — the hackathon expects you
to improve on it (see §5).

---

## 3. Running it in Antigravity IDE

[Antigravity](https://antigravity.google/) is Google's agentic, AI-native IDE.
This project is structured so an agent (or you, manually) can pick it up and
run it with zero configuration guesswork.

### Step-by-step

1. **Unzip and open the folder as a workspace**
   Open Antigravity → `File → Open Folder…` → select the unzipped
   `cpri_project/` (or `your-teamname-submission/`) directory.

2. **Let Antigravity set up the Python environment**
   Antigravity's agent can detect `src/requirements.txt` and provision a
   virtual environment automatically. If you want to do it yourself, open the
   integrated terminal (`` Ctrl+` `` / `` Cmd+` ``) and run:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r src/requirements.txt
   ```

3. **Set your team name**
   Open `src/pipeline.py` and edit the `CONFIG` block near the top:
   ```python
   TEAM_NAME = "YourTeamName"   # <- change this
   ```
   This is the *only* thing that must change for the output file to be named
   correctly (`YourTeamName.csv`).

4. **Run the full pipeline**
   From the integrated terminal:
   ```bash
   python src/pipeline.py
   ```
   Or, in Antigravity, right-click `src/pipeline.py` → **Run Python File**.
   Or use the agent chat panel: *"Run src/pipeline.py and show me the
   summary.json it produces."* — Antigravity's agent can execute this
   directly and read back the console output for you.

   Alternatively, open `notebook/solution.ipynb` and **Run All** — it calls
   the exact same code.

5. **Check the outputs**
   After the run finishes, look in `outputs/`:
   - `YourTeamName.csv` — the prediction deliverable
   - `summary.json` — the automated Task-3 summary
   - `cleaning_report.csv` — per-record diagnostics for your methodology note

6. **Iterate with the agent**
   Because everything is one script with clearly named functions
   (`clean_frame`, `rule_based_flags`, `train_validity_classifier`,
   `train_reference_regressor`, `build_predictions`, `build_summary`), you can
   ask Antigravity's agent things like:
   - *"Try replacing the RandomForestRegressor with XGBoost and compare hold-out MAE."*
   - *"Add a rule that also flags records where Sensor_S1 and Sensor_S2 diverge by more than X."*
   - *"Plot Reference_Parameter vs Load_Current_A colored by Validity_Label."*
   and it can edit `pipeline.py` directly and re-run it inside the IDE.

7. **Package your final submission**
   Once you're happy with the results, zip the required deliverables per the
   challenge brief:
   ```
   your-teamname-submission.zip
   ├── YourTeamName.csv
   ├── summary.json
   ├── src/ (or notebook/solution.ipynb)
   └── METHODOLOGY.md
   ```

---

## 4. Design principles baked into this scaffold

- **Fully reproducible / no manual record edits.** Every decision (imputation,
  rule thresholds, model training) is a function of the data and the config
  block — never a hard-coded row/ID.
- **Genuine regime change ≠ fault.** The rule engine flags *statistical and
  physical* anomalies (spikes, incoherent sensors, out-of-range values), not
  merely "unusual" values — high-voltage/high-current/long-duration
  combinations that are internally consistent will *not* be auto-flagged.
- **Separation of concerns.** Cleaning, rule-based flagging, ML classification,
  regression and summary generation are separate functions — easy for you (or
  an agent) to swap out any one stage independently.
- **Deterministic.** `RANDOM_STATE = 42` everywhere a model or split is used,
  so re-runs are reproducible for judging.

## 5. Ideas for improving on the baseline (screening → final round)

- Feature-engineer physically meaningful ratios (e.g. `Sensor_S1 - Sensor_S2`,
  `Load_Current_A**2 * Test_Duration_min` as a proxy for I²t heating).
- Formally test whether `Sensor_S4` carries any signal (permutation importance,
  ablation) and drop it if not — the brief explicitly calls this out.
- Try gradient boosting (XGBoost/LightGBM) or a simple physics-informed
  regression (e.g. Newton's law of heating form) alongside the Random Forest,
  and report whichever generalizes better.
- Use isolation forest / local outlier factor as a second, unsupervised check
  against the rule-based + supervised approach, and reconcile disagreements.
- Calibrate the classifier's decision threshold instead of the default 0.5,
  using the training hold-out.
- For the "digital twin" question in the methodology note: describe how this
  same pipeline would run automatically on a schedule against live test-bench
  data streams (ingestion → cleaning → scoring → alerting on invalid/high-
  attention records → dashboard), and what would need to be monitored for
  model drift.

---

