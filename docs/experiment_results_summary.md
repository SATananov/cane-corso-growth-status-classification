# Experiment Results Summary

The project compares three approaches:

1. majority-class baseline;
2. logistic regression;
3. random forest.

## Best model

The best model in the current run is the random forest classifier.

## Main metrics

The main reported metrics are accuracy, precision, recall, F1-score, and ROC-AUC.

## Why these metrics are used

Accuracy shows the overall share of correct predictions. Precision shows how reliable positive predictions are. Recall shows how many positive examples are found. F1-score balances precision and recall. ROC-AUC summarizes ranking quality across decision thresholds.

## What the model learns

The model learns that age, current weight, expected adult breed weight, engineered weight ratio, and related interaction features contain useful signal for separating the processed labels.

## Where the model fails

The model mainly struggles with borderline cases where the available features do not clearly separate `normal_growth` from `needs_attention`.

