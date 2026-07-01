"""Smoke test for the focused growth-status classification experiment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "growth_status"
MODEL_PATH = PROJECT_ROOT / "models" / "growth_status" / "growth_status_pipeline.joblib"


def main() -> None:
    command = [sys.executable, "src/run_growth_status_experiment.py"]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    required_files = [
        REPORT_DIR / "metrics.csv",
        REPORT_DIR / "confusion_matrix.csv",
        REPORT_DIR / "correct_examples.csv",
        REPORT_DIR / "error_examples.csv",
        REPORT_DIR / "borderline_examples.csv",
        REPORT_DIR / "feature_importance.csv",
        REPORT_DIR / "experiment_summary.md",
        REPORT_DIR / "experiment_metadata.json",
        MODEL_PATH,
    ]

    missing = [path.as_posix() for path in required_files if not path.exists()]
    if missing:
        raise AssertionError(f"Missing experiment output files: {missing}")

    metrics = pd.read_csv(REPORT_DIR / "metrics.csv")
    if metrics.empty:
        raise AssertionError("metrics.csv is empty.")

    expected_experiments = {
        "baseline_majority_class",
        "logistic_regression",
        "random_forest",
    }
    actual_experiments = set(metrics["experiment"].astype(str))
    if expected_experiments != actual_experiments:
        raise AssertionError(
            f"Unexpected experiment set: {sorted(actual_experiments)}"
        )

    best = metrics.sort_values(["f1", "roc_auc", "accuracy"], ascending=False).iloc[0]
    if float(best["f1"]) <= 0.70:
        raise AssertionError(f"Best F1 is unexpectedly weak: {best['f1']}")

    errors = pd.read_csv(REPORT_DIR / "error_examples.csv")
    if errors.empty:
        raise AssertionError("Expected at least one concrete error example.")

    summary_text = (REPORT_DIR / "experiment_summary.md").read_text(encoding="utf-8")
    required_phrases = [
        "One concrete ML question",
        "Data source",
        "Where the model makes mistakes",
        "Reproducibility",
    ]
    for phrase in required_phrases:
        if phrase not in summary_text:
            raise AssertionError(f"Missing summary section: {phrase}")

    print("GROWTH_STATUS_SMOKE_TEST_PASS")
    print(completed.stdout.strip())


if __name__ == "__main__":
    main()
