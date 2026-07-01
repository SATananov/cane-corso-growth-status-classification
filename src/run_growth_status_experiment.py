"""Focused dog growth-status classification experiment.

This script answers one concrete machine-learning question:

Can simple growth-related measurements classify whether a dog growth record is
normal_growth or needs_attention?

The script is intentionally compact and reproducible. It uses only the committed
processed public sample, a fixed random seed, a baseline model, two supervised
models, and explicit error analysis examples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dog_growth_classification_sample.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "growth_status"
MODEL_DIR = PROJECT_ROOT / "models" / "growth_status"
MODEL_PATH = MODEL_DIR / "growth_status_pipeline.joblib"

TARGET_COLUMN = "growth_status_binary"

# The target in the processed dataset was derived from body-condition information.
# The processed target was derived from body-condition information.
# We deliberately exclude bcs_recorded, bcs_predicted, bcs_source and
# growth_status text from model features to reduce direct leakage.
NUMERIC_FEATURES = [
    "visit_age_months",
    "weight_kg",
    "average_adult_breed_weight_kg",
    "weight_to_expected_adult_ratio",
    "weight_age_interaction",
]

CATEGORICAL_FEATURES = [
    "gender",
    "preventive_care_visit",
    "healthy_pet_diagnosis",
]

LABEL_NAMES = {
    0: "normal_growth",
    1: "needs_attention",
}


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_PATH}")

    frame = pd.read_csv(INPUT_PATH)

    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]) - {
        "weight_to_expected_adult_ratio",
        "weight_age_interaction",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame.copy()
    frame["weight_to_expected_adult_ratio"] = (
        pd.to_numeric(frame["weight_kg"], errors="coerce")
        / pd.to_numeric(frame["average_adult_breed_weight_kg"], errors="coerce").replace(0, np.nan)
    )
    frame["weight_age_interaction"] = (
        pd.to_numeric(frame["weight_kg"], errors="coerce")
        * pd.to_numeric(frame["visit_age_months"], errors="coerce")
    )

    return frame


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "baseline_majority_class": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_depth=6,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)
        if probability.shape[1] == 2:
            return probability[:, 1]
    prediction = model.predict(X)
    return np.asarray(prediction, dtype=float)


def evaluate_model(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, Any], Pipeline, pd.DataFrame]:
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    probability = positive_probability(model, X_test)

    if len(np.unique(probability)) <= 1:
        roc_auc: float | None = None
    else:
        roc_auc = round(float(roc_auc_score(y_test, probability)), 6)

    metrics = {
        "experiment": name,
        "accuracy": round(float(accuracy_score(y_test, prediction)), 6),
        "precision": round(float(precision_score(y_test, prediction, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, prediction, zero_division=0)), 6),
        "f1": round(float(f1_score(y_test, prediction, zero_division=0)), 6),
        "roc_auc": roc_auc,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    predictions = X_test.copy()
    predictions["actual"] = y_test.to_numpy()
    predictions["predicted"] = prediction
    predictions["actual_label"] = predictions["actual"].map(LABEL_NAMES)
    predictions["predicted_label"] = predictions["predicted"].map(LABEL_NAMES)
    predictions["probability_needs_attention"] = np.round(probability, 6)
    predictions["experiment"] = name
    predictions["is_correct"] = predictions["actual"] == predictions["predicted"]
    predictions["confidence_distance"] = (predictions["probability_needs_attention"] - 0.5).abs()
    return metrics, model, predictions


def add_human_explanation(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()

    def explain(row: pd.Series) -> str:
        ratio = row.get("weight_to_expected_adult_ratio")
        age = row.get("visit_age_months")
        prob = row.get("probability_needs_attention")
        true_label = row.get("actual_label")
        predicted_label = row.get("predicted_label")

        if pd.isna(ratio):
            ratio_text = "unknown weight/adult-weight ratio"
        elif ratio < 0.45:
            ratio_text = "low weight/adult-weight ratio"
        elif ratio < 0.85:
            ratio_text = "developing weight/adult-weight ratio"
        elif ratio < 1.15:
            ratio_text = "near expected adult-weight ratio"
        else:
            ratio_text = "high weight/adult-weight ratio"

        return (
            f"Age={age:.2f} months, weight ratio={ratio:.3f} ({ratio_text}), "
            f"probability_needs_attention={prob:.3f}. "
            f"True label is {true_label}, model predicted {predicted_label}."
        )

    rows["explanation"] = rows.apply(explain, axis=1)
    return rows


def build_example_tables(best_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    display_columns = [
        "visit_age_months",
        "weight_kg",
        "average_adult_breed_weight_kg",
        "weight_to_expected_adult_ratio",
        "gender",
        "preventive_care_visit",
        "healthy_pet_diagnosis",
        "actual_label",
        "predicted_label",
        "probability_needs_attention",
    ]

    correct = (
        best_predictions[best_predictions["is_correct"]]
        .sort_values("confidence_distance", ascending=False)
        .head(8)
        [display_columns]
    )
    errors = (
        best_predictions[~best_predictions["is_correct"]]
        .sort_values("confidence_distance", ascending=False)
        .head(8)
        [display_columns]
    )
    borderline = (
        best_predictions
        .sort_values("confidence_distance", ascending=True)
        .head(8)
        [display_columns]
    )

    return add_human_explanation(correct), add_human_explanation(errors), add_human_explanation(borderline)


def feature_importance_summary(model: Pipeline) -> pd.DataFrame:
    fitted_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocess"]

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_
        kind = "random_forest_importance"
    elif hasattr(fitted_model, "coef_"):
        values = np.ravel(np.abs(fitted_model.coef_))
        kind = "absolute_logistic_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "value", "importance_type"])

    summary = pd.DataFrame({
        "feature": feature_names,
        "value": np.round(values, 8),
        "importance_type": kind,
    })
    return summary.sort_values("value", ascending=False).head(15)


def write_summary(
    metrics: pd.DataFrame,
    best_name: str,
    best_metrics: dict[str, Any],
    confusion: pd.DataFrame,
    correct: pd.DataFrame,
    errors: pd.DataFrame,
    borderline: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    source_note = (
        "University of Liverpool DataCat, 'Growth standard charts for monitoring bodyweight "
        "in dogs of different sizes - SUPPORTING DATA', DOI: https://doi.org/10.17638/datacat.liverpool.ac.uk/377"
    )

    error_section = (
        errors[[
            "visit_age_months",
            "weight_kg",
            "weight_to_expected_adult_ratio",
            "actual_label",
            "predicted_label",
            "probability_needs_attention",
            "explanation",
        ]].to_markdown(index=False)
        if not errors.empty
        else "No misclassified examples were found in the selected test split."
    )

    metrics_for_display = metrics.copy()
    if "roc_auc" in metrics_for_display.columns:
        metrics_for_display["roc_auc"] = metrics_for_display["roc_auc"].apply(
            lambda value: "not_applicable" if pd.isna(value) else value
        )

    summary = f"""# Growth Status Classification Experiment

## One concrete ML question

Can simple growth-related measurements classify whether a dog growth record is `normal_growth` or `needs_attention`?

This project is intentionally limited to one supervised-learning analysis so the question, data, model behaviour and errors are easy to follow.

## Data source

{source_note}

The committed file used by this experiment is:

```text
data/processed/dog_growth_classification_sample.csv
```

The project does not claim that this is a private Cane Corso-only clinical dataset. The public dataset supplies the real dog-growth data foundation; the Cane Corso context supplies the practical product motivation.

## Target

```text
0 = normal_growth
1 = needs_attention
```

Important leakage control: the model deliberately excludes `bcs_recorded`, `bcs_predicted`, `bcs_source`, and the text label `growth_status` from the input features, because the processed target was derived from body-condition information.

## Features used

Numeric features:

```text
{", ".join(NUMERIC_FEATURES)}
```

Categorical features:

```text
{", ".join(CATEGORICAL_FEATURES)}
```

## Models compared

{metrics_for_display.to_markdown(index=False)}

Best model selected by F1 score:

```text
{best_name}
```

Best model metrics:

```json
{json.dumps(best_metrics, indent=2)}
```

## Confusion matrix for best model

Rows are actual labels and columns are predicted labels.

{confusion.to_markdown()}

## What the model learned

The best model mainly learns relationships between age, bodyweight, adult breed-weight expectation, and visit context. In practical terms, the engineered weight-to-adult-weight ratio and the age-weight interaction help the model separate many records that look closer to normal development from records that look more suspicious.

Top feature signals:

{feature_importance.to_markdown(index=False)}

## Where the model makes mistakes

The most important diagnostic evidence is the error analysis below. These are concrete test-set records where the best model predicted the wrong label.

{error_section}

The main weakness is that borderline records can look similar using only simple tabular features. Some records have weight ratios or visit contexts that do not clearly separate normal and needs-attention labels. This is why the output should be treated as an educational growth-monitoring signal, not as veterinary diagnosis.

## Correct prediction examples

{correct.head(5).to_markdown(index=False)}

## Borderline / uncertain examples

{borderline.head(5).to_markdown(index=False)}

## Reproducibility

Run from the project root:

```bash
python src/run_growth_status_experiment.py
```

Optional smoke test:

```bash
python tests/smoke_test_growth_status_experiment.py
```

The script writes all experiment artifacts into:

```text
reports/growth_status/
models/growth_status/
```
"""
    (REPORT_DIR / "experiment_summary.md").write_text(summary, encoding="utf-8")


def run_growth_status_experiment() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_data()
    X = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = frame[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    metric_rows: list[dict[str, Any]] = []
    prediction_tables: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        metrics, fitted_model, predictions = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        metric_rows.append(metrics)
        prediction_tables[name] = predictions
        fitted_models[name] = fitted_model

    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["f1", "roc_auc", "accuracy"],
        ascending=False,
        na_position="last",
    )
    metrics_frame.to_csv(REPORT_DIR / "metrics.csv", index=False)

    best_name = str(metrics_frame.iloc[0]["experiment"])
    best_metrics = metrics_frame.iloc[0].to_dict()
    best_model = fitted_models[best_name]
    best_predictions = prediction_tables[best_name].copy()

    joblib.dump(best_model, MODEL_PATH)

    best_predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    labels = [0, 1]
    matrix = confusion_matrix(best_predictions["actual"], best_predictions["predicted"], labels=labels)
    confusion = pd.DataFrame(
        matrix,
        index=[f"actual_{LABEL_NAMES[label]}" for label in labels],
        columns=[f"predicted_{LABEL_NAMES[label]}" for label in labels],
    )
    confusion.to_csv(REPORT_DIR / "confusion_matrix.csv")

    correct, errors, borderline = build_example_tables(best_predictions)
    correct.to_csv(REPORT_DIR / "correct_examples.csv", index=False)
    errors.to_csv(REPORT_DIR / "error_examples.csv", index=False)
    borderline.to_csv(REPORT_DIR / "borderline_examples.csv", index=False)

    feature_importance = feature_importance_summary(best_model)
    feature_importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    metadata = {
        "project_focus": "growth_status_classification",
        "question": "Can simple growth-related measurements classify whether a dog growth record is normal_growth or needs_attention?",
        "input_path": INPUT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "target_column": TARGET_COLUMN,
        "excluded_leakage_columns": ["bcs_recorded", "bcs_predicted", "bcs_source", "growth_status"],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "random_state": RANDOM_STATE,
        "best_experiment": best_name,
        "model_path": MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }
    (REPORT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_summary(
        metrics=metrics_frame,
        best_name=best_name,
        best_metrics=best_metrics,
        confusion=confusion,
        correct=correct,
        errors=errors,
        borderline=borderline,
        feature_importance=feature_importance,
    )

    return {
        "best_experiment": best_name,
        "metrics_path": (REPORT_DIR / "metrics.csv").relative_to(PROJECT_ROOT).as_posix(),
        "summary_path": (REPORT_DIR / "experiment_summary.md").relative_to(PROJECT_ROOT).as_posix(),
        "model_path": MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }


def main() -> None:
    result = run_growth_status_experiment()
    print("GROWTH_STATUS_EXPERIMENT_READY")
    print(f"BEST_EXPERIMENT {result['best_experiment']}")
    print(f"METRICS {result['metrics_path']}")
    print(f"SUMMARY {result['summary_path']}")
    print(f"MODEL {result['model_path']}")


if __name__ == "__main__":
    main()
