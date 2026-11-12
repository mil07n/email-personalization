"""Held-out AESLC retrieval evaluation and reporting."""

import ast
import json
import random
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz

import config
import embeddings as emb
from personalize import personalize


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _f1(predicted, actual):
    p, a = _tokens(predicted), _tokens(actual)
    if not p or not a:
        return 0.0
    overlap = len(p & a)
    precision, recall = overlap / len(p), overlap / len(a)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _markdown(df):
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |\n",
             "| " + " | ".join("---" for _ in headers) + " |\n"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |\n")
    return "".join(lines)


def _retrieve(test_df, train_df, tfidf, svd, train_lsa, train_w2v, test_w2v, test_lsa):
    rows = []
    methods = ("tfidf", "word2vec", "random")
    for i, row in test_df.iterrows():
        scores = {
            "tfidf": cosine_similarity(test_lsa[i:i + 1], train_lsa).ravel(),
            "word2vec": cosine_similarity(test_w2v[i:i + 1], train_w2v).ravel(),
            "random": np.random.default_rng(config.RANDOM_SEED + i).random(len(train_df)),
        }
        for method in methods:
            top = scores[method].argsort()[-3:][::-1]
            retrieved = train_df.iloc[top]
            rows.append({
                "test_email_id": row["email_id"],
                "true_subject": row["subject_line"],
                "method": method,
                "retrieved_subjects": " || ".join(retrieved["subject_line"].tolist()),
                "top1_subject_f1": round(_f1(retrieved.iloc[0]["subject_line"], row["subject_line"]), 4),
                "top3_subject_f1": round(float(np.mean([
                    _f1(subject, row["subject_line"]) for subject in retrieved["subject_line"]
                ])), 4),
            })
    result = pd.DataFrame(rows)
    result.to_csv(config.OUTPUTS_DIR / "retrieval_results.csv", index=False)
    summary = result.groupby("method")[["top1_subject_f1", "top3_subject_f1"]].mean().reset_index()
    summary.to_csv(config.OUTPUTS_DIR / "retrieval_comparison.csv", index=False)
    return result, summary


def _profile_baseline():
    profiles = [
        ("Ava", "I manage gas pipeline nominations, daily flows, and transport capacity."),
        ("Ben", "I trade power and natural gas and monitor basis risk across regional hubs."),
        ("Cara", "I review ISDA agreements, credit terms, and energy trading contracts."),
        ("Dev", "I coordinate legal review of purchase agreements and commercial schedules."),
        ("Eli", "I lead recruiting, employee relations, and HR planning for the operations team."),
        ("Faye", "I schedule executive meetings, circulate minutes, and track action items."),
        ("Gus", "I arrange client travel, flights, hotels, and itineraries for business meetings."),
        ("Hana", "I prepare monthly finance reports, budgets, and cash-flow forecasts."),
        ("Ivan", "I manage utility settlements, invoices, and daily energy market reconciliations."),
        ("Jules", "I negotiate procurement terms, vendor pricing, and delivery commitments."),
        ("Kira", "I handle client accounts, sales proposals, and commercial negotiations."),
        ("Luca", "I oversee plant maintenance schedules, outage planning, and field operations."),
    ]
    rows = []
    for name, text in profiles:
        topics = set(re.findall(r"[a-z]{4,}", text.lower()))
        result = personalize(name, text, retrieval="tfidf")
        generated = result["body"].lower()
        personalized_hits = sum(generated.count(topic) for topic in result["topics"])
        baseline_body = (
            f"Hello {name},\n\nI am reaching out regarding your professional work.\n\n"
            "Best,\nMilan"
        ).lower()
        baseline_hits = sum(baseline_body.count(topic) for topic in topics)
        rows.append({
            "name": name,
            "profile": text,
            "segment_id": result["segment_id"],
            "segment_label": result["segment_label"],
            "top_similarity_score": round(result["similar_emails"][0]["score"], 4),
            "topics_extracted": len(result["topics"]),
            "personalized_topic_mentions": personalized_hits,
            "segment_only_topic_mentions": baseline_hits,
        })
    return pd.DataFrame(rows)


def run_evaluation():
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(config.MODELS_DIR / "train_corpus.csv")
    test_df = pd.read_csv(config.MODELS_DIR / "test_corpus.csv")
    train_df["tokens"] = train_df["tokens"].map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    test_df["tokens"] = test_df["tokens"].map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    tfidf = joblib.load(config.MODELS_DIR / "tfidf_vectorizer.joblib")
    svd = joblib.load(config.MODELS_DIR / "lsa_svd.joblib")
    train_matrix = load_npz(str(config.MODELS_DIR / "tfidf_matrix.npz"))
    train_lsa = svd.transform(train_matrix)
    test_lsa = np.load(config.MODELS_DIR / "test_lsa.npy")
    train_w2v = np.load(config.MODELS_DIR / "email_vecs_w2v.npy")
    test_w2v = np.load(config.MODELS_DIR / "test_email_vecs_w2v.npy")
    retrieval, retrieval_summary = _retrieve(
        test_df, train_df, tfidf, svd, train_lsa, train_w2v, test_w2v, test_lsa
    )
    labels = np.load(config.MODELS_DIR / "test_segments.npy")
    silhouette = None
    if len(test_lsa) > 2 and len(np.unique(labels)) > 1:
        silhouette = float(silhouette_score(test_lsa, labels))
    plot_svd = TruncatedSVD(n_components=2, random_state=config.RANDOM_SEED)
    test_2d = plot_svd.fit_transform(load_npz(str(config.MODELS_DIR / "tfidf_test_matrix.npz")))
    plt.figure(figsize=(10, 7))
    plt.scatter(test_2d[:, 0], test_2d[:, 1], c=labels, cmap="tab20", alpha=0.6, s=12)
    plt.title(f"AESLC test clusters (k={len(set(labels))}, silhouette={silhouette})")
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "cluster_scatter.png", dpi=120)
    plt.close()
    baseline = _profile_baseline()
    baseline.to_csv(config.OUTPUTS_DIR / "profile_baseline.csv", index=False)
    (config.OUTPUTS_DIR / "evaluation_report.md").write_text(f"""# AESLC Evaluation Report

## Held-out evaluation

- Train index: **{len(train_df)}** emails.
- Test index: **{len(test_df)}** emails.
- KMeans silhouette on held-out test assignments: **{silhouette if silhouette is not None else "not defined"}**.
- The clustering model and all vectorizers/embeddings were fit on train only.

## Retrieval subject-line token-overlap F1

{_markdown(retrieval_summary)}

The random row is a retrieval baseline. `top3_subject_f1` averages F1 against each of the three retrieved subjects.

## Profile-to-segment sanity check

The 12 Enron-style profiles below are a varied qualitative sanity check. Each
row reports the predicted segment and the highest similarity score among the
three retrieved train emails. The full table is in `outputs/profile_baseline.csv`.

{_markdown(baseline[["name", "segment_id", "segment_label", "top_similarity_score"]])}

Distinct segments reached: **{baseline["segment_id"].nunique()}** of **{len(set(labels))}**.

Topic coverage is retained only as a sanity check: personalized topic mentions
mean **{baseline["personalized_topic_mentions"].mean():.2f}** per profile versus
**{baseline["segment_only_topic_mentions"].mean():.2f}** for the segment-only baseline.

## Limitations

AESLC is noisy Enron mail and subject-line token overlap is a strict lexical metric. Clusters are unsupervised and can have low silhouette even when useful to generation. Human-readable segment labels in `outputs/segments.csv` are editable and are not ground-truth categories.
""")
    return {"silhouette": silhouette, "retrieval": retrieval_summary.to_dict("records")}


if __name__ == "__main__":
    result = run_evaluation()
    print(f"Evaluation complete. silhouette={result['silhouette']}")
