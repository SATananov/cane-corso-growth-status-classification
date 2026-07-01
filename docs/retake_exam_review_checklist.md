# Retake Exam Review Checklist

This document summarizes how the focused retake project answers the main review questions.

## 1. What problem is solved?

The project solves one supervised machine learning problem:

Can simple growth-related measurements classify whether a dog growth record is `normal_growth` or `needs_attention`?

The project is intentionally focused on one analysis, not on a broad machine learning portfolio.

## 2. What is the target label?

The target label is the processed growth-status class:

- `normal_growth`
- `needs_attention`

This label is used only for educational machine learning analysis. It is not a veterinary diagnosis.

## 3. What are the input features?

The model uses simple growth-related tabular features, including:

- age in months;
- current weight;
- expected adult breed weight;
- engineered weight-to-expected-adult-weight ratio;
- interaction features;
- selected categorical and binary indicators from the processed dataset.

## 4. What is the baseline?

The baseline is a majority-class classifier. It gives a simple reference point that trained models should improve on.

## 5. Which models are compared?

The project compares:

1. majority-class baseline;
2. logistic regression;
3. random forest classifier.

## 6. Which model performs best?

In the current experiment, the random forest classifier is the best-performing model. The results are saved in:

```text
reports/growth_status/metrics.csv
```

## 7. Which metrics are used?

The project evaluates the models with:

- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- confusion matrix.

These metrics are used because a single accuracy score is not enough to understand classification behavior.

## 8. Where does the model make mistakes?

The project saves concrete prediction examples:

```text
reports/growth_status/error_examples.csv
reports/growth_status/correct_examples.csv
reports/growth_status/borderline_examples.csv
```

The main errors appear in borderline cases where the available tabular features are not enough to clearly separate `normal_growth` from `needs_attention`.

## 9. What do the errors mean?

The errors show that the features contain useful signal, but they are not sufficient for medical or veterinary conclusions. Additional real-world context would be needed for responsible practical use.

## 10. How can another person reproduce the experiment?

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python src/run_growth_status_experiment.py
```

Run the smoke test:

```bash
python tests/smoke_test_growth_status_experiment.py
```

The generated outputs are saved in:

```text
reports/growth_status/
```

## 11. Main grading entry point

The main notebook for review is:

```text
notebooks/final_growth_status_classification.ipynb
```

## 12. Repository history

The retake exam guidelines require at least 10 meaningful commits. This repository contains 10+ topic-based commits covering the initial project, documentation, mathematical formulation, dataset validation, error analysis, reproducibility, result summary, and retake notes.
