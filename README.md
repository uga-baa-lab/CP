# CHEAT-CP Kinematic Analysis

A modular Python package for analyzing upper-limb kinematic data from KINARM robot experiments, comparing motor control in children with **Cerebral Palsy (CP)** and **Typically Developing Children (TDC)**.

The analysis covers two task types:
- **Reaching** — moving the hand to a stationary target
- **Interception** — moving the hand to intercept a moving target

---

## Repository Structure

```
CP/
├── cp_analysis/                  # Main analysis package (use this)
│   ├── __init__.py
│   ├── config.py                 # All paths and default parameters
│   ├── signal_processing.py      # Butterworth filtering, hand speed, cursor positions
│   ├── kinematics.py             # Per-trial kinematic measures (RT, CT, IDE, PLR, ...)
│   ├── regression.py             # RANSAC regression and Initial Estimate (IE)
│   ├── pvar.py                   # Path Variability (PVar) computation
│   ├── pipeline.py               # Orchestrates the full analysis — run this
│   ├── plotting/
│   │   ├── plot_trials.py        # Single-trial and group visualization
│   │   └── plotting.py           # Legacy plotting (old version)
│   └── reporting/
│       ├── report_generator.py   # Generates all 6 Excel mastersheets
│       └── helpers.py            # Pivot, reindex, save Excel utilities
│
└── working_version_2024/         # Original monolithic scripts (reference only)
```

---

## What the Pipeline Computes

For every trial, the pipeline extracts:

| Measure | Description |
|---|---|
| **RT** | Reaction Time — when the hand starts moving |
| **CT** | Completion Time — when the cursor crosses the target Y-position |
| **MT** | Movement Time — CT minus RT |
| **velPeak** | Peak hand speed (mm/s) |
| **IDE** | Initial Direction Error — angle between actual and ideal movement at RT+50ms |
| **EndPointError** | Cursor-to-target distance at CT (mm) |
| **PLR** | Path Length Ratio — actual path / ideal straight path |
| **IE** | Initial Estimate error — where the hand *planned* to intercept vs where the target was |
| **PVar** | Path Variability — consistency of hand paths across trials |
| **pathlength** | Total hand path length from start to CT (mm) |
| **maxpathoffset** | Maximum perpendicular deviation from straight line |

---

## Requirements

- Python 3.10+
- A KINARM `.mat` data directory
- A master Excel file listing all subjects and visit days

---

## Environment Setup

Pick **one** of the two approaches below.

---

### Option A — `venv` (built-in, no extra tools needed)

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd CP

# 2. Create the virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Verify everything imported correctly
cd cp_analysis
python3 -c "import signal_processing, kinematics, regression, pvar, pipeline; print('All good')"
```

To deactivate when done:
```bash
deactivate
```

---

### Option B — Conda / Anaconda

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd CP

# 2. Create a new conda environment (Python 3.11 recommended)
conda create -n cheatcp python=3.11

# 3. Activate it
conda activate cheatcp

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Verify everything imported correctly
cd cp_analysis
python -c "import signal_processing, kinematics, regression, pvar, pipeline; print('All good')"
```

To deactivate when done:
```bash
conda deactivate
```

To remove the environment entirely:
```bash
conda env remove -n cheatcp
```

---

## Setup

### 1. Clone and set up the environment

Follow **Option A** or **Option B** above, then come back here.

### 2. Set your data paths in `config.py`

Open `cp_analysis/config.py` and update **one line** — the `BASE_DIR`:

```python
# Line 23 — change this to wherever your data lives
BASE_DIR = r"/path/to/your/data/folder"
```

Everything else derives from `BASE_DIR` automatically:

```
BASE_DIR/
├── data/
│   ├── matfiles/           ← .mat files go here (CHEAT-CP001Day1.mat, etc.)
│   └── KINARM_Test.xlsx    ← master subject list
└── results_macos_2026/     ← all output is written here (auto-created)
```

### 3. Verify the master Excel file

The master Excel file (`KINARM_Test.xlsx`) must have a sheet named **`KINARM_AllVisitsMaster`** with at least these columns:

| Column | Description |
|---|---|
| `KINARM ID` | Subject ID, e.g. `CHEAT-CP001` or `cpvib001` |
| `Subject ID` | Study-level ID, e.g. `cpvib001` |
| `Visit ID` | Visit identifier |
| `Visit_Day` | Day string, e.g. `Day1`, `Day2` |
| `Age at Visit (yr)` | Age at that visit |
| `Group` | `0` = TDC, `1` = CP |

### 4. Name your `.mat` files correctly

Files must follow this naming convention:

```
CHEAT-CP{subject_suffix}{Visit_Day}.mat
```

For example, if `KINARM ID` is `CHEAT-CP001` and `Visit_Day` is `Day1`:

```
CHEAT-CP001Day1.mat
```

---

## Running the Analysis

### Full pipeline (recommended)

From inside the `cp_analysis/` directory:

```bash
cd cp_analysis
python3 pipeline.py
```

This runs all steps end-to-end:
1. Loads the master Excel subject list
2. Processes every subject/visit (kinematics + RANSAC regression + IE)
3. Saves `all_processed_trials_final.csv`
4. Computes Path Variability (PVar)
5. Generates all 6 Excel report mastersheets

### Run from the project root (as a package)

```python
from cp_analysis.pipeline import run_pipeline
from cp_analysis.config import (
    MASTER_FILE, MATFILES_DIR, DEFAULTS,
    RESULTS_DIR, REPORT_RESULTS_DIR, PVAR_RESULTS_DIR
)

run_pipeline(
    master_file=MASTER_FILE,
    matfiles=MATFILES_DIR,
    defaults=DEFAULTS,
    results_dir=RESULTS_DIR,
    report_results_dir=REPORT_RESULTS_DIR,
    pvar_results_dir=PVAR_RESULTS_DIR,
)
```

### Process only specific subjects (for testing)

```python
run_pipeline(
    ...,
    subject_filter=["CHEAT-CP001", "CHEAT-CP002"]
)
```

---

## Outputs

All outputs are written under `BASE_DIR/results_macos_2026/`:

```
results_macos_2026/
├── all_processed_trials_final.csv          ← Trial-level data for all subjects
│
├── IE_CSV_Results/
│   ├── All_Trials_IE.csv                   ← IE values for all trials
│   └── Grouped_IE.csv                      ← IE sorted by subject and condition
│
├── PVAR_RESULTS/
│   ├── pvar_results.csv                    ← PVar per subject × arm × duration
│   └── PVAR_Plots/                         ← Trajectory plots per group
│
├── {subject}/
│   ├── Robust_Reg/
│   │   ├── {subject}_{day}_{arm}_all_data.png     ← Linear regression plot
│   │   └── {subject}_{day}_{arm}_inliers_only.png ← RANSAC inliers plot
│   ├── Outlier_Trials/                            ← Plots of flagged outlier trials
│   └── IE_values_{subject}.csv                    ← Per-subject IE breakdown
│
├── Subject_Arm_Block_Level_Outliers.csv    ← Outlier count summary
├── regression_results.csv                  ← R² comparison (Linear vs Robust)
│
└── IE_plots_window_based/report_results/
    ├── UL_KINARM_Mastersheet_Auto_Format_means.xlsx
    ├── UL_KINARM_Mastersheet_Auto_Format_stds.xlsx
    ├── UL_KINARM_Mastersheet_Only_IDE_means.xlsx
    ├── UL_KINARM_Mastersheet_Only_IDE_STD.xlsx
    ├── UL_KINARM_Mastersheet_PVar_Auto_Format_means.xlsx
    └── UL_KINARM_Mastersheet_PVar_Auto_Format_stds.xlsx
```

Each Excel mastersheet has **one sheet per visit day (Day1–Day5)**. Each sheet is pivoted wide — rows are subjects, columns are `Variable × Condition × Arm` with individual duration columns (500, 625, 750, 900 ms) plus combined columns (500+625, 750+900).

---

## Key Configuration Options

All in `cp_analysis/config.py`:

| Setting | Default | Description |
|---|---|---|
| `BASE_DIR` | *(set this)* | Root of your data folder |
| `DEFAULTS['fs']` | `1000` Hz | KINARM sampling frequency |
| `DEFAULTS['fc']` | `5` Hz | Low-pass filter cutoff |
| `DEFAULTS['fdfwd']` | `0.06` | KINARM feedforward constant |
| `consider_window_for_intial_plan` | `True` | Use 50–100 ms window for x_intersect |
| `TOTAL_IDS` | `88` | Total number of subjects |
| `MAX_DAYS` | `5` | Maximum number of visit days |

---

## Module Overview

| Module | Responsibility |
|---|---|
| `config.py` | Paths, defaults, subject ID generation |
| `signal_processing.py` | Butterworth filtering, hand speed, cursor position, IQR cleaning |
| `kinematics.py` | All per-trial measures: RT, CT, IDE, PLR, path lengths, x_intersect |
| `regression.py` | RANSAC regression, Initial Estimate (IE), outlier detection and plotting |
| `pvar.py` | Time-normalize trajectories, compute Path Variability per arm/duration |
| `pipeline.py` | Top-level orchestration — loads data, calls all modules, saves outputs |
| `plotting/plot_trials.py` | Trial-level and group-level visualization |
| `reporting/report_generator.py` | Generates all 6 Excel mastersheets |
| `reporting/helpers.py` | Pivot tables, reindexing, Excel saving, std combining |

---

## How the Initial Estimate (IE) Works

IE quantifies **predictive motor planning** in interception trials.

1. For **Reaching** trials, we compute `x_intersect` — where the initial movement direction projects onto the target's Y-level.
2. Since the target is stationary, `x_intersect` should correlate with the target's actual X position. We fit this with **RANSAC regression** per arm: `x_target ≈ f(x_intersect)`.
3. For **Interception** trials, we apply the same model: `IE = f(x_intersect) − x_target_at_RT`. A large positive IE means the hand aimed *ahead* of the target.

---

## Troubleshooting

**`.mat` file not found**
> Check that the file name matches `CHEAT-CP{suffix}{Visit_Day}.mat` exactly. The subject suffix is the last 3 characters of the `KINARM ID` column.

**`KeyError: 'KINARM_AllVisitsMaster'`**
> The master Excel file must have a sheet named exactly `KINARM_AllVisitsMaster`.

**`ModuleNotFoundError` when running `pipeline.py`**
> Run from inside `cp_analysis/` directly (`cd cp_analysis && python3 pipeline.py`), or ensure the package root is on your Python path.

**PVar report not generated**
> PVar runs after the main pipeline. If `pvar_results.csv` is missing, check that at least 2 valid Reaching trials exist per subject/day.

**IE values are all NaN**
> This usually means `x_intersect` was not computed (RT was invalid for most trials). Check the `.mat` data quality for that subject.

---

## Authors

Brain and Action Lab, University of Georgia

For questions: dbarany@uga.edu
  

  