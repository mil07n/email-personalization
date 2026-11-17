# All Models Evaluation and Benchmark Results
**Project:** NLP-Powered Executive Email Assistant (MailMind AI)  
**Generated At:** 2026-11-17 19:23:42  
**Single Output File Path:** `C:\Users\tejai\OneDrive\Desktop\AESLC-Pipeline\evaluation\ALL_MODELS_EVALUATION_RESULTS.json`

---

## 1. Executive Summary of All Models

| # | Pipeline Capability | Winning Model | Primary Metric | Primary Dataset |
|---|---|---|---|---|
| 1 | **Urgency Classification** | **Calibrated LinearSVC (TF-IDF 1-3 ngrams)** | **90.49% Accuracy** (89.68% F1) | Customer Care Emails (2,259 records) |
| 2 | **Action-Item Extraction** | **Intent NLP Parser** | **93.99% F1** (94.66% Recall) | EPA Dataset (6,734 annotations) |
| 3 | **Task Assignee Matching** | **Addressee Entity Matcher** | **74.19% Accuracy** | EPA Dataset |
| 4 | **Conversation Grouping** | **TF-IDF Cosine Clusterer** | **81.74% NMI** | Customer Care Thread IDs |
| 5 | **Thread Summarization** | **Extractive Graph Salience** | **44.52% ROUGE-1 Recall** | Email Thread Summary (4,167 threads) |
| 6 | **Category & Spam Classification**| **Multi-Signal Lexicon Engine** | **8 Categories Classified** | Enron Corporate Corpus & Anti-Spam Lexicons |

---

## 2. Model 1: Urgency Classification Benchmark

* **Dataset:** Customer Care Emails (`datasets/customer_care/processed/customer_care_emails.csv`)
* **Split:** 80% Train (1,807 samples) | 20% Test (452 samples), Stratified
* **Classes:** `high`, `medium`, `low`

### Comparative Model Accuracies:

| Algorithm | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|:---:|:---:|:---:|:---:|:---:|
| **LinearSVC (Winning Model)** | **90.49%** | **90.44%** | **89.37%** | **89.68%** | **90.47%** |
| **Logistic Regression** | 88.94% | 88.98% | 87.73% | 88.14% | 88.92% |
| **Multinomial Naive Bayes** | 84.96% | 84.81% | 83.18% | 83.69% | 84.70% |

### Detailed Performance for Winning LinearSVC Model:
* **High Urgency:** Precision: **94.7%** | Recall: **94.1%** | F1: **94.4%**
* **Medium Urgency:** Precision: **86.2%** | Recall: **85.0%** | F1: **85.6%**
* **Low Urgency:** Precision: **89.3%** | Recall: **90.2%** | F1: **89.7%**

---

## 3. Model 2: Action-Item & Task Assignee Extraction

* **Dataset:** EPA Dataset (Microsoft Research / EMNLP 2018 W-NUT)
* **Evaluated Samples:** 300 test sentences with ground-truth task intent and assignees

| Metric | Score | Details |
|---|:---:|---|
| **Precision** | **93.33%** | 266 True Positives, 19 False Positives |
| **Recall** | **94.66%** | Only 15 False Negatives |
| **F1-Score** | **93.99%** | Harmonic mean of precision and recall |
| **Addressee Accuracy** | **74.19%** | Correctly assigned person responsible |

---

## 4. Model 3: Conversation & Thread Grouping

* **Dataset:** Customer Care Emails (400 evaluation samples across 120 true conversation threads)
* **Model:** Subject Normalization + Sublinear TF-IDF Cosine Similarity Clustering

| Metric | Score | Interpretation |
|---|:---:|---|
| **Normalized Mutual Information (NMI)** | **0.8174 (81.74%)** | Strong mutual clustering agreement with true email threads |
| **Adjusted Rand Index (ARI)** | **0.2773** | Pairwise clustering consistency measure |

---

## 5. Model 4: Email & Thread Summarization (ROUGE Metrics)

### Benchmark A: Email Thread Summary Dataset (4,167 threads)
* **ROUGE-1 Recall:** **44.52%** | Precision: 13.18% | F1: **19.56%**
* **ROUGE-2 Recall:** **14.68%** | Precision: 4.22% | F1: **6.30%**
* **ROUGE-L Recall:** **26.22%** | Precision: 7.56% | F1: **11.24%**

### Benchmark B: EmailSum Dataset (2,549 threads)
* **ROUGE-1 Precision:** **33.33%** | F1: **6.14%**
* **ROUGE-L Precision:** **33.33%** | F1: **6.14%**

---

## 6. Model 5: Category & Spam Classification
Classifies emails across 8 distinct operational buckets:
1. `SPAM` (e.g., lottery, prize, phishing keywords, unverified links)
2. `URGENT` (e.g., immediate, production outage, escalation)
3. `ACTION_REQUIRED` (e.g., please review, sign, submit, approve)
4. `MEETING` (e.g., calendar invite, zoom, sync, schedule)
5. `FINANCE` (e.g., invoice, payment, bill, wire transfer, wire)
6. `NEWSLETTER` (e.g., weekly digest, unsubscribe, marketing updates)
7. `SOCIAL` (e.g., connection request, notifications)
8. `GENERAL` (e.g., standard correspondence)

---

## 7. Sample Real-Time Pipeline Predictions

```json
[
  {
    "test_id": "sample_1_urgent",
    "subject": "CRITICAL: Production API Outage and Payment Gateway Failure",
    "sender": "cto@company.com",
    "predicted_urgency": "CRITICAL",
    "urgency_confidence": 0.95,
    "predicted_category": "OPERATIONS",
    "category_confidence": 0.4,
    "extracted_action_items": [
      {
        "id": "task_1",
        "action": "Investigate immediately and fix the server configuration by 3 pm today",
        "responsible_person": "John",
        "deadline": "today",
        "normalized_deadline": "2026-11-17 17:00",
        "priority": "HIGH",
        "status": "pending",
        "completed": false,
        "confidence": 0.9,
        "source_sentence": "John, please investigate immediately and fix the server configuration by 3 PM today."
      }
    ],
    "extracted_deadlines": [
      {
        "raw": "today",
        "type": "relative",
        "normalized": "2026-11-17 17:00",
        "start": 200,
        "end": 205
      },
      {
        "raw": "immediately",
        "type": "relative",
        "normalized": "2026-11-17 (Immediate)",
        "start": 147,
        "end": 158
      },
      {
        "raw": "by 3 PM",
        "type": "time",
        "normalized": "by 3 PM",
        "start": 192,
        "end": 199
      }
    ],
    "generated_summary": "The payment gateway is offline and customers cannot checkout. John, please investigate immediately and fix the server configuration by 3 PM today."
  },
  {
    "test_id": "sample_2_finance",
    "subject": "Invoice #4829 Overdue - Immediate Payment Required",
    "sender": "billing@vendor.com",
    "predicted_urgency": "CRITICAL",
    "urgency_confidence": 0.95,
    "predicted_category": "FINANCE",
    "category_confidence": 0.5,
    "extracted_action_items": [
      {
        "id": "task_1",
        "action": "Find attached the overdue invoice for q3 software licenses",
        "responsible_person": "Finance Team",
        "deadline": "No deadline specified",
        "normalized_deadline": null,
        "priority": "HIGH",
        "status": "pending",
        "completed": false,
        "confidence": 0.9,
        "source_sentence": "Dear Finance Team, please find attached the overdue invoice for Q3 software licenses."
      }
    ],
    "extracted_deadlines": [
      {
        "raw": "before Friday",
        "type": "relative",
        "normalized": "2026-11-20 (Friday)",
        "start": 167,
        "end": 180
      }
    ],
    "generated_summary": "Dear Finance Team, please find attached the overdue invoice for Q3 software licenses. The payment of $12,400 is due before Friday."
  },
  {
    "test_id": "sample_3_newsletter",
    "subject": "Weekly Tech Digest & Community Highlights",
    "sender": "newsletter@techworld.com",
    "predicted_urgency": "HIGH",
    "urgency_confidence": 0.457,
    "predicted_category": "SPAM",
    "category_confidence": 0.5,
    "extracted_action_items": [],
    "extracted_deadlines": [],
    "generated_summary": "Here is your weekly summary of industry news and updates. Click here to read the full blog post or unsubscribe if you no longer wish to receive these updates."
  }
]
```
