# Email Summarization Evaluation Report

Evaluated using **ROUGE-1**, **ROUGE-2**, and **ROUGE-L** metrics comparing extractive generated summaries with human reference summaries.

## Dataset: `email_thread_summary` (150 samples)

| Metric | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **ROUGE-1 (Unigram Overlap)** | 0.1318 | 0.4452 | **0.1956** |
| **ROUGE-2 (Bigram Overlap)** | 0.0422 | 0.1468 | **0.0630** |
| **ROUGE-L (Longest Common Subsequence)** | 0.0756 | 0.2622 | **0.1124** |

## Dataset: `emailsum` (150 samples)

| Metric | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **ROUGE-1 (Unigram Overlap)** | 0.3333 | 0.0339 | **0.0614** |
| **ROUGE-2 (Bigram Overlap)** | 0.0000 | 0.0000 | **0.0000** |
| **ROUGE-L (Longest Common Subsequence)** | 0.3333 | 0.0339 | **0.0614** |
