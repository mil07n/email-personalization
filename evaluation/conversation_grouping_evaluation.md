# Conversation Grouping Benchmark Report

**Benchmark Dataset:** Customer Care Emails (`dataset.csv`)  
**Evaluated Sample:** 400 messages across 120 ground truth threads  

## Grouping Performance

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Adjusted Rand Index (ARI)** | **0.2773** | Measure of similarity between predicted clusters and true threads (chance = 0.0, perfect = 1.0) |
| **Normalized Mutual Information (NMI)** | **0.8174** | Information shared between predicted clusters and true conversations |
| **Predicted Conversations** | **96** | Grouped threads (True threads: 120) |
