# Urgency Classification Model Evaluation Report

**Dataset:** Customer Care Emails (`dataset.csv`)  
**Train Samples:** 1,807 | **Test Samples:** 452  
**Selected Best Model:** **Ensemble (LinearSVC + LogReg)** (F1 Macro = 0.8986)

## Model Comparison

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ensemble (LinearSVC + LogReg)** 🏆 (Selected) | 0.9071 | 0.8991 | 0.8989 | **0.8986** | 0.9066 |
| **LinearSVC** | 0.9049 | 0.8968 | 0.8970 | **0.8968** | 0.9047 |
| **LogisticRegression** | 0.8894 | 0.8815 | 0.8824 | **0.8814** | 0.8892 |
| **MultinomialNB** | 0.8496 | 0.8455 | 0.8380 | **0.8369** | 0.8470 |

## Confusion Matrix (Best Model: Ensemble (LinearSVC + LogReg))

| Actual \ Predicted | Low | Medium | High |
| :--- | :---: | :---: | :---: |
| **Low** | 120 | 12 | 0 |
| **Medium** | 17 | 109 | 7 |
| **High** | 1 | 5 | 181 |

## Per-Class Breakdown

| Priority Level | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Low** | 0.8696 | 0.9091 | 0.8889 | 132.0 |
| **Medium** | 0.8651 | 0.8195 | 0.8417 | 133.0 |
| **High** | 0.9628 | 0.9679 | 0.9653 | 187.0 |
