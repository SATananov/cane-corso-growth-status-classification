# Growth Status Classification Experiment

## One concrete ML question

Can simple growth-related measurements classify whether a dog growth record is `normal_growth` or `needs_attention`?

This project is intentionally limited to one supervised-learning analysis so the question, data, model behaviour and errors are easy to follow.

## Data source

University of Liverpool DataCat, 'Growth standard charts for monitoring bodyweight in dogs of different sizes - SUPPORTING DATA', DOI: https://doi.org/10.17638/datacat.liverpool.ac.uk/377

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
visit_age_months, weight_kg, average_adult_breed_weight_kg, weight_to_expected_adult_ratio, weight_age_interaction
```

Categorical features:

```text
gender, preventive_care_visit, healthy_pet_diagnosis
```

## Models compared

| experiment              |   accuracy |   precision |   recall |       f1 | roc_auc        |   train_rows |   test_rows |
|:------------------------|-----------:|------------:|---------:|---------:|:---------------|-------------:|------------:|
| random_forest           |     0.8308 |    0.786159 |   0.9088 | 0.843043 | 0.912492       |         7500 |        2500 |
| logistic_regression     |     0.8112 |    0.770891 |   0.8856 | 0.824274 | 0.891552       |         7500 |        2500 |
| baseline_majority_class |     0.5    |    0        |   0      | 0        | not_applicable |         7500 |        2500 |

Best model selected by F1 score:

```text
random_forest
```

Best model metrics:

```json
{
  "experiment": "random_forest",
  "accuracy": 0.8308,
  "precision": 0.786159,
  "recall": 0.9088,
  "f1": 0.843043,
  "roc_auc": 0.912492,
  "train_rows": 7500,
  "test_rows": 2500
}
```

## Confusion matrix for best model

Rows are actual labels and columns are predicted labels.

|                        |   predicted_normal_growth |   predicted_needs_attention |
|:-----------------------|--------------------------:|----------------------------:|
| actual_normal_growth   |                       941 |                         309 |
| actual_needs_attention |                       114 |                        1136 |

## What the model learned

The best model mainly learns relationships between age, bodyweight, adult breed-weight expectation, and visit context. In practical terms, the engineered weight-to-adult-weight ratio and the age-weight interaction help the model separate many records that look closer to normal development from records that look more suspicious.

Top feature signals:

| feature                                 |      value | importance_type          |
|:----------------------------------------|-----------:|:-------------------------|
| numeric__average_adult_breed_weight_kg  | 0.280615   | random_forest_importance |
| numeric__weight_kg                      | 0.247515   | random_forest_importance |
| numeric__weight_age_interaction         | 0.237368   | random_forest_importance |
| numeric__weight_to_expected_adult_ratio | 0.105193   | random_forest_importance |
| numeric__visit_age_months               | 0.0621099  | random_forest_importance |
| categorical__healthy_pet_diagnosis_Y    | 0.0245524  | random_forest_importance |
| categorical__healthy_pet_diagnosis_N    | 0.0187283  | random_forest_importance |
| categorical__gender_MN                  | 0.00679837 | random_forest_importance |
| categorical__gender_M                   | 0.00623111 | random_forest_importance |
| categorical__gender_FS                  | 0.00452004 | random_forest_importance |
| categorical__gender_F                   | 0.00358469 | random_forest_importance |
| categorical__preventive_care_visit_Y    | 0.00154438 | random_forest_importance |
| categorical__preventive_care_visit_N    | 0.00123919 | random_forest_importance |

## Where the model makes mistakes

The most important diagnostic evidence is the error analysis below. These are concrete test-set records where the best model predicted the wrong label.

|   visit_age_months |   weight_kg |   weight_to_expected_adult_ratio | actual_label   | predicted_label   |   probability_needs_attention | explanation                                                                                                                                                               |
|-------------------:|------------:|---------------------------------:|:---------------|:------------------|------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|              35.4  |      51.256 |                          1.42735 | normal_growth  | needs_attention   |                      0.938726 | Age=35.40 months, weight ratio=1.427 (high weight/adult-weight ratio), probability_needs_attention=0.939. True label is normal_growth, model predicted needs_attention.   |
|              31.98 |      45.495 |                          1.26692 | normal_growth  | needs_attention   |                      0.930086 | Age=31.98 months, weight ratio=1.267 (high weight/adult-weight ratio), probability_needs_attention=0.930. True label is normal_growth, model predicted needs_attention.   |
|              35.36 |      45.813 |                          1.27577 | normal_growth  | needs_attention   |                      0.929462 | Age=35.36 months, weight ratio=1.276 (high weight/adult-weight ratio), probability_needs_attention=0.929. True label is normal_growth, model predicted needs_attention.   |
|              30.85 |      45.949 |                          1.27956 | normal_growth  | needs_attention   |                      0.924909 | Age=30.85 months, weight ratio=1.280 (high weight/adult-weight ratio), probability_needs_attention=0.925. True label is normal_growth, model predicted needs_attention.   |
|              31.19 |      45.359 |                          1.26313 | normal_growth  | needs_attention   |                      0.922889 | Age=31.19 months, weight ratio=1.263 (high weight/adult-weight ratio), probability_needs_attention=0.923. True label is normal_growth, model predicted needs_attention.   |
|              34.96 |      40.551 |                          1.12924 | normal_growth  | needs_attention   |                      0.912444 | Age=34.96 months, weight ratio=1.129 (near expected adult-weight ratio), probability_needs_attention=0.912. True label is normal_growth, model predicted needs_attention. |
|              26.95 |      42.229 |                          1.17597 | normal_growth  | needs_attention   |                      0.90611  | Age=26.95 months, weight ratio=1.176 (high weight/adult-weight ratio), probability_needs_attention=0.906. True label is normal_growth, model predicted needs_attention.   |
|              29.8  |      41.005 |                          1.14188 | normal_growth  | needs_attention   |                      0.904634 | Age=29.80 months, weight ratio=1.142 (near expected adult-weight ratio), probability_needs_attention=0.905. True label is normal_growth, model predicted needs_attention. |

The main weakness is that borderline records can look similar using only simple tabular features. Some records have weight ratios or visit contexts that do not clearly separate normal and needs-attention labels. This is why the output should be treated as an educational growth-monitoring signal, not as veterinary diagnosis.

## Correct prediction examples

|   visit_age_months |   weight_kg |   average_adult_breed_weight_kg |   weight_to_expected_adult_ratio | gender   | preventive_care_visit   | healthy_pet_diagnosis   | actual_label   | predicted_label   |   probability_needs_attention | explanation                                                                                                                                                                 |
|-------------------:|------------:|--------------------------------:|---------------------------------:|:---------|:------------------------|:------------------------|:---------------|:------------------|------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|              34.57 |      12.338 |                          12.569 |                         0.981621 | FS       | Y                       | N                       | normal_growth  | normal_growth     |                      0.001571 | Age=34.57 months, weight ratio=0.982 (near expected adult-weight ratio), probability_needs_attention=0.002. True label is normal_growth, model predicted normal_growth.     |
|              35.52 |       9.072 |                          12.569 |                         0.721776 | FS       | Y                       | N                       | normal_growth  | normal_growth     |                      0.001801 | Age=35.52 months, weight ratio=0.722 (developing weight/adult-weight ratio), probability_needs_attention=0.002. True label is normal_growth, model predicted normal_growth. |
|              30.04 |       8.709 |                          12.569 |                         0.692895 | FS       | Y                       | N                       | normal_growth  | normal_growth     |                      0.001821 | Age=30.04 months, weight ratio=0.693 (developing weight/adult-weight ratio), probability_needs_attention=0.002. True label is normal_growth, model predicted normal_growth. |
|              30.29 |       9.798 |                          12.569 |                         0.779537 | FS       | Y                       | N                       | normal_growth  | normal_growth     |                      0.001833 | Age=30.29 months, weight ratio=0.780 (developing weight/adult-weight ratio), probability_needs_attention=0.002. True label is normal_growth, model predicted normal_growth. |
|              30.26 |      10.705 |                          12.569 |                         0.851699 | FS       | Y                       | N                       | normal_growth  | normal_growth     |                      0.001833 | Age=30.26 months, weight ratio=0.852 (near expected adult-weight ratio), probability_needs_attention=0.002. True label is normal_growth, model predicted normal_growth.     |

## Borderline / uncertain examples

|   visit_age_months |   weight_kg |   average_adult_breed_weight_kg |   weight_to_expected_adult_ratio | gender   | preventive_care_visit   | healthy_pet_diagnosis   | actual_label    | predicted_label   |   probability_needs_attention | explanation                                                                                                                                                                     |
|-------------------:|------------:|--------------------------------:|---------------------------------:|:---------|:------------------------|:------------------------|:----------------|:------------------|------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|              12.01 |      24.494 |                           35.91 |                         0.682094 | MN       | Y                       | N                       | needs_attention | needs_attention   |                      0.50076  | Age=12.01 months, weight ratio=0.682 (developing weight/adult-weight ratio), probability_needs_attention=0.501. True label is needs_attention, model predicted needs_attention. |
|              18.19 |      29.937 |                           35.91 |                         0.833668 | FS       | N                       | Y                       | normal_growth   | normal_growth     |                      0.498679 | Age=18.19 months, weight ratio=0.834 (developing weight/adult-weight ratio), probability_needs_attention=0.499. True label is normal_growth, model predicted normal_growth.     |
|              33.78 |      33.566 |                           35.91 |                         0.934726 | FS       | N                       | Y                       | needs_attention | normal_growth     |                      0.497451 | Age=33.78 months, weight ratio=0.935 (near expected adult-weight ratio), probability_needs_attention=0.497. True label is needs_attention, model predicted normal_growth.       |
|              15.72 |      17.055 |                           35.91 |                         0.474937 | MN       | Y                       | N                       | normal_growth   | needs_attention   |                      0.502715 | Age=15.72 months, weight ratio=0.475 (developing weight/adult-weight ratio), probability_needs_attention=0.503. True label is normal_growth, model predicted needs_attention.   |
|              17.84 |      29.665 |                           35.91 |                         0.826093 | MN       | Y                       | Y                       | normal_growth   | needs_attention   |                      0.503012 | Age=17.84 months, weight ratio=0.826 (developing weight/adult-weight ratio), probability_needs_attention=0.503. True label is normal_growth, model predicted needs_attention.   |

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
