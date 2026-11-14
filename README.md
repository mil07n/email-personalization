# NLP-Powered Executive Email Assistant

An intelligent natural language processing (NLP) assistant that analyzes incoming corporate emails to **classify urgency**, **extract action items and assignees**, **recognize temporal deadlines**, **group related conversations**, and **compile personalized daily executive summaries**.

---

## 📌 Project Architecture

```
                       INCOMING EMAILS
                              |
                              v
                      DATA PREPROCESSING
               (HTML removal, normalization,
                tokenization, noise cleaning)
                              |
                              v
                    EMAIL ANALYSIS ENGINE
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
       URGENCY           ACTION-ITEM         CONVERSATION
    CLASSIFICATION       EXTRACTION            GROUPING
 (TF-IDF + LinearSVC)  (EPA Addressee)    (Subject / TF-IDF)
          |                   |                   |
          |                   v                   |
          |          DEADLINE EXTRACTION          |
          |       (NER & Temporal Patterns)       |
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                     EMAIL SUMMARIZATION
             (Sentence-importance scoring & ROUGE)
                              |
                              v
                    PERSONALIZATION ENGINE
           (VIP senders, priority topics, scoring)
                              |
                              v
                 PERSONALIZED DAILY SUMMARY
                              |
                              v
                EXECUTIVE DASHBOARD (React.js)
```

---

## 📊 Datasets

The project incorporates **five distinct real-world datasets**, each stored in its own dedicated folder under `datasets/`:

| Dataset | Location | Records | Size | Role in Project |
| :--- | :--- | :---: | :---: | :--- |
| **Enron Email Dataset** | `datasets/enron/` | **16,746** | 28.09 MB | Workplace communication corpus, real-world baseline testing |
| **Customer Care Emails** | `datasets/customer_care/` | **2,259** | 1.44 MB | Ground-truth criticality labels (`high`, `medium`, `low`) & `thread_id` |
| **Email Thread Summary** | `datasets/email_thread_summary/` | **4,167** | 45.31 MB | Multi-turn email conversations paired with human summaries |
| **EPA Dataset** | `datasets/epa/` | **6,734** | 16.33 MB | EMNLP W-NUT task intent sentences & addressee assignments |
| **EmailSum Dataset** | `datasets/emailsum/` | **2,549** | 1.43 MB | ACL 2021 multi-turn email conversation summaries |

*All datasets have raw files preserved in `datasets/<name>/raw/` and cleaned tabular representations in `datasets/<name>/processed/`.*

---

## 🧠 NLP Techniques & Modules

### 1. Data Preprocessing (`src/preprocessing/`)
- **HTML & Boilerplate Removal:** Unescapes HTML entities, strips tags, removes disclaimers, forward headers, and quoted reply chains.
- **Tokenization:** Sentence tokenization using NLTK `sent_tokenize` and word tokenization via `word_tokenize`.
- **Normalization:** NFKC unicode normalization, whitespace collapsing, contact masking.

### 2. Urgency Classification (`src/urgency_classification/`)
- **Feature Extraction:** Sublinear TF-IDF word and character n-grams (1-2) with English stopword filtering.
- **Model Comparison:** Evaluated `LogisticRegression`, `LinearSVC` (calibrated probabilities), and `MultinomialNB`.
- **Selected Model:** **LinearSVC** achieved the highest test performance:
  - **Accuracy:** `83.63%`
  - **Macro F1:** `0.8242`
  - **Weighted F1:** `0.8357`
- **Signal Boosting:** Lexical matching for system outages (`outage`, `crash`), legal threats (`subpoena`, `lawsuit`), and capital pattern markers (`URGENT`, `ASAP`).

### 3. Action-Item Extraction (`src/action_extraction/`)
- **Intent Recognition:** Regex & grammar trigger patterns for imperatives, directives, and polite requests (`please prepare`, `we need you to submit`).
- **Addressee Assignment:** Matches leading vocative names (`John, please...`) and salutations (`Hi Sarah,`) to attribute tasks to the responsible individual.
- **EPA Dataset Benchmark:**
  - **Precision:** `93.33%`
  - **Recall:** `94.66%`
  - **F1-Score:** `93.99%`
  - **Addressee Accuracy:** `74.19%`

### 4. Deadline Extraction (`src/deadline_extraction/`)
- **Temporal Patterns:** Recognizes relative days (`tomorrow afternoon`, `by EOD`, `before Friday`), duration windows (`within 24 hours`), and explicit dates (`Nov 20th, 2026`).
- **ISO Normalization:** Automatically converts relative terms to concrete ISO timestamps (`YYYY-MM-DD HH:MM`) based on reference time.
- **Association:** Binds detected deadlines to their corresponding action item sentence.

### 5. Conversation Grouping (`src/conversation_grouping/`)
- **Subject Normalization:** Recursively strips `Re:`, `Fwd:`, `FW:` prefixes to identify conversation seeds.
- **Multi-signal Clustering:** Combines subject identity, sender/receiver participant overlap, and TF-IDF cosine similarity.
- **Benchmark Performance:** Evaluated against ground-truth Customer Care `thread_id`:
  - **Normalized Mutual Information (NMI):** `0.8174`
  - **Adjusted Rand Index (ARI):** `0.2773`

### 6. Email Summarization (`src/summarization/`)
- **Extractive Sentence Importance:** Scores sentences using TF-IDF term density, positional weighting (first request + final outcome boost), and decision cue phrases (`we have decided`, `agreed to`, `approved`).
- **Evaluation Metrics (ROUGE):** Tested on Email Thread Summary:
  - **ROUGE-1 F1:** `0.1956`
  - **ROUGE-2 F1:** `0.0630`
  - **ROUGE-L F1:** `0.1124`

### 7. Personalization Engine (`src/personalization/`)
- **User Preference System:** Stored in `config/user_preferences.json`.
- **Dynamic Re-ranking:**
  - Base urgency score (`0.20` – `0.95`)
  - `+0.30` bonus if sender matches VIP list
  - `+0.25` bonus for priority topic keywords
  - `+0.15` bonus if actionable tasks exist
  - `+0.10` bonus for near-term deadlines
- **Personalized Daily Briefing:** Synthesizes an executive markdown briefing highlighting urgent action required, pending tasks, and upcoming deadlines.

---

## 📁 Project Folder Structure

```
AESLC-Pipeline/
├── datasets/
│   ├── enron/ (raw/ & processed/ aeslc.csv)
│   ├── customer_care/ (raw/ & processed/ customer_care_emails.csv)
│   ├── email_thread_summary/ (raw/ & processed/ email_thread_summary.csv)
│   ├── epa/ (raw/ & processed/ epa_tasks.csv)
│   ├── emailsum/ (raw/ & processed/ emailsum.csv)
│   ├── DATASET_STATUS.json
│   └── DATASET_STATUS.md
├── src/
│   ├── preprocessing/ (preprocessor.py)
│   ├── urgency_classification/ (classifier.py, train_urgency.py)
│   ├── action_extraction/ (extractor.py, evaluate_action_extraction.py)
│   ├── deadline_extraction/ (extractor.py)
│   ├── conversation_grouping/ (grouper.py, evaluate_grouping.py)
│   ├── summarization/ (summarizer.py, evaluate_summarization.py)
│   ├── personalization/ (engine.py)
│   └── integration/ (pipeline.py)
├── models/
│   ├── urgency_model.joblib
│   └── urgency_vectorizer.joblib
├── evaluation/
│   ├── urgency_metrics.json & urgency_evaluation.md
│   ├── action_extraction_metrics.json & action_extraction_evaluation.md
│   ├── conversation_grouping_metrics.json & conversation_grouping_evaluation.md
│   └── summarization_metrics.json & summarization_evaluation.md
├── config/
│   ├── config.py
│   └── user_preferences.json
├── backend/
│   ├── main.py (FastAPI REST server)
│   ├── models.py (Pydantic schemas)
│   └── sample_loader.py (15 curated sample emails)
├── frontend/
│   ├── public/ (index.html - React 18 single-page app)
│   └── package.json
├── tests/
│   └── test_nlp_modules.py
├── requirements.txt
├── start.bat
├── start.ps1
└── README.md
```

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.10+ (tested on Python 3.15)
- Standard packages installed via:
  ```bash
  pip install -r requirements.txt
  ```

### 1. Launch the Live Web Application
Double-click `start.bat` or run:

```powershell
python backend/main.py
```

Then open your browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

*Interactive Swagger API documentation is available at: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)***

---

### 2. Live Web Application Pages

1. **📊 Executive Dashboard:**
   - 6 statistical overview cards (Total Emails, Critical, High, Medium, Action Items, Deadlines)
   - Real-time Critical & High Priority email queue
   - Action item checklist with assignees and deadlines
   - Recent conversation threads and daily briefing preview
2. **📥 Email Inbox:**
   - Real-time search by sender, subject, or content
   - Priority filter tabs (All, Critical, High, Medium, Low)
   - Action item badges and deadline indicators
3. **🔍 Email Deep Analysis:**
   - Click any email to inspect urgency confidence and probability breakdown
   - Detected lexical urgency signals
   - Extracted tasks with responsible individuals
   - Extractive conversation summary
4. **💬 Conversation View:**
   - Grouped email threads with chronological message flow
   - Multi-turn conversation summaries and key decisions
5. **📋 Daily Briefing:**
   - Markdown-rendered personalized executive summary
   - Interactive button to re-generate the briefing
6. **⚙️ User Preferences:**
   - Editable VIP sender list and priority topic keywords
   - Dynamic re-scoring of inbox and daily summary upon saving
7. **✨ Live Demonstration Modal:**
   - **Manual Entry:** Type custom emails to test the NLP pipeline live
   - **Sample Library:** 15 preloaded emails from Enron & Customer Care
   - **File Upload:** Upload `.txt` or `.eml` files for instant analysis

---

### 3. Run Automated Tests & Evaluations

```powershell
# Run all unit and integration tests (9 test suites)
python -m unittest tests/test_nlp_modules.py

# Re-train Urgency Classification models (compares LogisticRegression, LinearSVC, Naive Bayes)
python src/urgency_classification/train_urgency.py

# Run Action-Item Extraction benchmark on EPA Dataset
python src/action_extraction/evaluate_action_extraction.py

# Run Conversation Grouping benchmark on Customer Care
python src/conversation_grouping/evaluate_grouping.py

# Run Summarization ROUGE benchmark on Email Thread Summary & EmailSum
python src/summarization/evaluate_summarization.py
```
