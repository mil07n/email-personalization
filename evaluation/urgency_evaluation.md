# Urgency Classification Model Evaluation Report

**Dataset:** Customer Care Emails (`dataset.csv`)  
**Train Samples:** 1,807 | **Test Samples:** 452  
**Selected Best Model:** **LinearSVC** (F1 Macro = 0.8242)

## Model Comparison

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LogisticRegression** | 0.8164 | 0.8036 | 0.8009 | **0.8009** | 0.8135 |
| **LinearSVC** 🏆 (Selected) | 0.8363 | 0.8244 | 0.8242 | **0.8242** | 0.8357 |
| **MultinomialNB** | 0.7478 | 0.7315 | 0.7246 | **0.7173** | 0.7343 |

## Confusion Matrix (Best Model: LinearSVC)

| Actual \ Predicted | Low | Medium | High |
| :--- | :---: | :---: | :---: |
| **Low** | 108 | 22 | 2 |
| **Medium** | 22 | 97 | 14 |
| **High** | 5 | 9 | 173 |

## Per-Class Breakdown

| Priority Level | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Low** | 0.8000 | 0.8182 | 0.8090 | 132.0 |
| **Medium** | 0.7578 | 0.7293 | 0.7433 | 133.0 |
| **High** | 0.9153 | 0.9251 | 0.9202 | 187.0 |
