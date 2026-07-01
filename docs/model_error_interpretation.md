# Model Error Interpretation

This document explains how prediction errors are interpreted in the project.

## Why error analysis matters

A model score alone is not enough. The project also inspects concrete examples where the model is correct, incorrect, or uncertain.

## Error files

The experiment writes:

```text
reports/growth_status/error_examples.csv
reports/growth_status/correct_examples.csv
reports/growth_status/borderline_examples.csv
```

## Main interpretation

The model is expected to make more mistakes on borderline records. These are records where the feature values are close to the decision boundary between `normal_growth` and `needs_attention`.

## What the errors mean

The errors suggest that simple tabular growth measurements are useful but incomplete. Some records may require additional context, such as breed-specific growth curves, medical notes, repeated measurements, nutrition, or veterinary evaluation.

## Responsible use

The model output should be treated as an educational signal. It is not a medical conclusion and should not replace expert assessment.
