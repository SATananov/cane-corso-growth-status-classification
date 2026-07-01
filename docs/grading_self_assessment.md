# Grading Self-Assessment

This document maps the project to the retake exam grading criteria.

## Problem understanding

The project studies one supervised machine learning question:

Can simple growth-related measurements classify whether a dog growth record is `normal_growth` or `needs_attention`?

The problem is intentionally narrow. The goal is not to build a medical or veterinary decision system, but to test whether basic growth-related tabular features contain a useful signal for a processed educational label.

## Writing layout

The repository is structured around one main notebook, one reproducible experiment script, and one results folder. The README explains the project question, data, target label, features, models, metrics, errors, and reproduction steps.

## Mathematical understanding

The project uses supervised classification. It compares a majority-class baseline, logistic regression, and random forest. The evaluation uses accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

## Code quality

The experiment is implemented in a reusable Python script. It saves metrics, predictions, examples, and the trained model. A smoke test checks that the main outputs exist and contain the expected structure.

## Methods and data handling

The dataset is a processed educational sample derived from a public dog growth dataset. The project documents the source, the target label, the input features, the assumptions, and the limitations.

