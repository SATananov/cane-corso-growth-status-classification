# Mathematical Formulation

This project is a binary classification task.

## Input data

Each record is represented as a feature vector:

```text
x = [x1, x2, ..., xn]
```

In this project, the features include age, weight, expected adult breed weight, engineered weight ratio features, and selected categorical or binary indicators.

## Target label

The target variable is binary:

```text
y = 0 -> normal_growth
y = 1 -> needs_attention
```

The label is used only for educational machine learning analysis. It is not a veterinary diagnosis.

## Baseline

The baseline model predicts the majority class. It is used as a minimum reference point. A trained model should perform better than this simple rule.

## Logistic regression

Logistic regression estimates the probability that a record belongs to the positive class:

```text
P(y = 1 | x) = 1 / (1 + exp(-(w*x + b)))
```

The model is useful because it is simple and interpretable.

## Random forest

Random forest combines multiple decision trees. Each tree makes a classification decision, and the forest aggregates the decisions. This helps capture non-linear relationships between features and the target label.

## Evaluation

The project uses accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, correct prediction examples, error examples, and borderline examples.

The goal is not only to get a score, but also to understand what the model learns and where it fails.
