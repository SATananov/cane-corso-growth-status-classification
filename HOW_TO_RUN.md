# How to Run

This project has one main review path.

## Main notebook

```text
notebooks/final_growth_status_classification.ipynb
```

## 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the experiment

```bash
python src/run_growth_status_experiment.py
```

Expected console signal:

```text
GROWTH_STATUS_EXPERIMENT_READY
```

## 4. Run the smoke test

```bash
python tests/smoke_test_growth_status_experiment.py
```

Expected console signal:

```text
GROWTH_STATUS_SMOKE_TEST_PASS
```

## 5. Read the results

Start with:

```text
reports/growth_status/experiment_summary.md
```

Then inspect:

```text
reports/growth_status/metrics.csv
reports/growth_status/confusion_matrix.csv
reports/growth_status/error_examples.csv
reports/growth_status/correct_examples.csv
reports/growth_status/borderline_examples.csv
reports/growth_status/feature_importance.csv
```

## 6. Data source

The committed processed file is:

```text
data/processed/dog_growth_classification_sample.csv
```

Original public source:

```text
University of Liverpool DataCat
Growth standard charts for monitoring bodyweight in dogs of different sizes - SUPPORTING DATA
DOI: https://doi.org/10.17638/datacat.liverpool.ac.uk/377
```
