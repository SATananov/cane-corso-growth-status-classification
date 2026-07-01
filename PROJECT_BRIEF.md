# Project Brief

## Question

Can simple growth-related measurements classify whether a dog growth record is `normal_growth` or `needs_attention`?

## Motivation

The practical motivation is a future growth-monitoring tool for large-breed dogs such as Cane Corso. The current project does not try to diagnose health. It tests whether basic tabular growth information contains enough signal for a supervised classification exercise.

## Data

The project uses a processed sample derived from the University of Liverpool DataCat dog-growth supporting dataset:

```text
Growth standard charts for monitoring bodyweight in dogs of different sizes - SUPPORTING DATA
DOI: https://doi.org/10.17638/datacat.liverpool.ac.uk/377
```

The committed file used by the experiment is:

```text
data/processed/dog_growth_classification_sample.csv
```

## Method

The experiment compares a majority-class baseline, Logistic Regression and Random Forest. The data is split with a fixed random seed. Numeric features are imputed and scaled. Categorical features are imputed and one-hot encoded. The best model is selected by F1 score.

## Evaluation

The project reports accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, feature importance, correct examples, wrong examples and borderline examples. The error examples are included because they show the practical limits of the model.

## Main limitation

The target is a processed educational label. It is not a veterinary diagnosis. The model should be interpreted as a course project and a prototype monitoring signal only.
