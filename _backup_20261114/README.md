# AESLC Email Personalization Pipeline

This project uses the AESLC Enron subject-line corpus in
`AESLC-master/enron_subject_line`. It parses each `.subject` file into an
email body and subject line, preserves the original train/dev/test split, and
removes cleaned-text duplicates before fitting models.

## Run

From the `NLP-Project` folder, use the existing virtual environment alongside
the project folder:

```bash
cd aeslc_pipeline
../.venv/bin/python pipeline.py
../.venv/bin/python app.py
```

Windows:

```powershell
cd aeslc_pipeline
..\.venv\Scripts\python pipeline.py
..\.venv\Scripts\python app.py
```

The pipeline runs preprocessing, train-only TF-IDF/LSA/KMeans/Word2Vec fitting,
held-out retrieval evaluation, report generation, and smoke tests.

## Pipeline files

- `preprocess.py` - parses AESLC and removes Enron noise.
- `train.py` - fits train-only vectorizers, embeddings, KMeans, and segment profiles.
- `evaluate.py` - evaluates held-out retrieval against random, TF-IDF, and Word2Vec baselines.
- `personalize.py` - extracts profile topics and generates segment-style emails.
- `app.py` - Gradio demo.
- `pipeline.py` - end-to-end runner.
- `tests/smoke_test.py` - automated validation.
- `embeddings.py` - gensim Word2Vec with PPMI+SVD fallback.

## Important outputs

- `data/aeslc.csv` - parsed and cleaned corpus.
- `outputs/data_report.md` - parsing, split, cleaning, and deduplication counts.
- `outputs/segments.csv` - editable segment labels, keywords, style, and examples.
- `outputs/retrieval_comparison.csv` - held-out subject-line token-overlap F1.
- `outputs/evaluation_report.md` - metrics and limitations.
- `outputs/cluster_scatter.png` - held-out 2D cluster visualization.

There is no supervised classifier. KMeans selection is restricted to `k=5..12`
and rejects any candidate with a cluster smaller than 2% of the training set.
