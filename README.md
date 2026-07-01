# Dog Growth Status Classification

This project answers one supervised machine-learning question:

```text
Can simple growth-related measurements classify whether a dog growth record is normal_growth or needs_attention?
```

The project is intentionally small and focused. It contains one main notebook, one reproducible training script, concrete metrics, and manual examples of correct, wrong and borderline predictions.

## Main file to read

```text
notebooks/final_growth_status_classification.ipynb
```

## Data source

The experiment uses a committed processed sample:

```text
data/processed/dog_growth_classification_sample.csv
```

Original public source:

```text
University of Liverpool DataCat
Growth standard charts for monitoring bodyweight in dogs of different sizes - SUPPORTING DATA
DOI: https://doi.org/10.17638/datacat.liverpool.ac.uk/377
```

The dataset is public dog-growth data. The project does not claim to use a private clinical Cane Corso dataset. The Cane Corso context is the practical motivation for a future monitoring tool.

## Target

```text
0 = normal_growth
1 = needs_attention
```

The target is a processed educational growth-status label, not a veterinary diagnosis.

## Features used

The model uses simple growth and visit-context features:

```text
visit_age_months
weight_kg
average_adult_breed_weight_kg
weight_to_expected_adult_ratio
weight_age_interaction
gender
preventive_care_visit
healthy_pet_diagnosis
```

To reduce direct target leakage, the model deliberately excludes fields that are directly connected to the processed label source:

```text
bcs_recorded
bcs_predicted
bcs_source
growth_status
```

## Models compared

The script compares:

1. majority-class baseline;
2. Logistic Regression;
3. Random Forest.

Current committed result:

| experiment | accuracy | precision | recall | f1 | roc_auc |
|---|---:|---:|---:|---:|---:|
| random_forest | 0.8308 | 0.7862 | 0.9088 | 0.8430 | 0.9125 |
| logistic_regression | 0.8112 | 0.7709 | 0.8856 | 0.8243 | 0.8916 |
| baseline_majority_class | 0.5000 | 0.0000 | 0.0000 | 0.0000 | not_applicable |

The baseline is included to show whether the trained models learn something beyond always predicting the majority class.

## What the model learns

The strongest model mainly uses current weight, expected adult breed weight, age, the engineered weight-to-expected-adult-weight ratio, and the age-weight interaction. In other words, the model learns a relationship between bodyweight, age and expected adult size inside the processed sample.

## Where the model fails

The project includes concrete error analysis:

```text
reports/growth_status/error_examples.csv
reports/growth_status/borderline_examples.csv
reports/growth_status/experiment_summary.md
```

The main weakness is that borderline records can look similar when only simple tabular measurements are available. Some records are predicted as `needs_attention` although the processed label is `normal_growth`, and some `needs_attention` records are missed. This is why the output should be treated as an educational monitoring signal, not as medical advice.

## Reproduce the experiment

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

Expected main outputs:

```text
reports/growth_status/metrics.csv
reports/growth_status/confusion_matrix.csv
reports/growth_status/correct_examples.csv
reports/growth_status/error_examples.csv
reports/growth_status/borderline_examples.csv
reports/growth_status/feature_importance.csv
reports/growth_status/experiment_summary.md
models/growth_status/growth_status_pipeline.joblib
```

## Project structure

```text
data/processed/                         processed sample used by the experiment
docs/                                   short notes about data, scope and limitations
models/growth_status/                   saved best model
notebooks/final_growth_status_classification.ipynb
reports/growth_status/                  metrics and concrete prediction examples
src/run_growth_status_experiment.py     reproducible experiment script
tests/smoke_test_growth_status_experiment.py
```

## Limitations

- The dataset is public dog-growth data, not a private clinical dataset.
- The target is a processed educational label, not a medical truth.
- The model uses simple tabular features only.
- The result is useful for course analysis and reproducible ML practice, not for veterinary decision-making.
