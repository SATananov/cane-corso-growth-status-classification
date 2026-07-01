# Reproducibility Checklist

This checklist explains how another person can reproduce the experiment.

## Environment

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Main experiment

Run:

```bash
python src/run_growth_status_experiment.py
```

## Smoke test

Run:

```bash
python tests/smoke_test_growth_status_experiment.py
```

## Expected output folder

```text
reports/growth_status/
```

## Expected outputs

- `metrics.csv`
- `confusion_matrix.csv`
- `feature_importance.csv`
- `correct_examples.csv`
- `error_examples.csv`
- `borderline_examples.csv`
- `test_predictions.csv`
- `experiment_summary.md`
- `experiment_metadata.json`

## Model artifact

```text
models/growth_status/growth_status_pipeline.joblib
```

## Fixed decisions

The experiment uses a fixed train/test split seed so that results are reproducible across runs with the same package versions.

