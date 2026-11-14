"""Leakage-safe AESLC feature fitting, clustering, and segment profiling."""

import ast
import json
import logging
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score

import config
import embeddings as emb

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_tokens(value):
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return str(value).split()


def _components(matrix):
    return max(1, min(config.LSA_COMPONENTS, matrix.shape[1] - 1, matrix.shape[0] - 1))


def _style(df):
    greetings = df["body"].str.extract(
        r"(?im)^\s*((?:dear|hi|hello|good morning|good afternoon|hey)\b[^,\n]*)"
    )[0].dropna().str.lower().str.replace(r"\s+", " ", regex=True)
    signoffs = df["body"].str.extract(
        r"(?is)(?:^|\n)\s*(regards|best|thanks|thank you|sincerely|cheers|kind regards)[,!]?\s*$"
    )[0].dropna().str.lower()
    formal = (
        df["body"].str.contains(r"\b(?:dear|sincerely|regards|please)\b", case=False, regex=True)
        & ~df["body"].str.contains(r"\b(?:hey|awesome|cool|thanks!)\b", case=False, regex=True)
    )
    return {
        "common_greeting": greetings.mode().iat[0] if not greetings.empty else "Hello",
        "common_signoff": signoffs.mode().iat[0] if not signoffs.empty else "Best",
        "formality": "formal" if formal.mean() >= 0.5 else "conversational",
    }


def _profiles(train_df, x_train, tfidf, kmeans, lsa):
    names = tfidf.get_feature_names_out()
    rows = []
    for cluster_id in range(kmeans.n_clusters):
        mask = train_df["segment_id"].to_numpy() == cluster_id
        group = train_df.loc[mask]
        mean = np.asarray(x_train[mask].mean(axis=0)).ravel()
        keywords = [names[i] for i in mean.argsort()[-12:][::-1]]
        center_dist = np.linalg.norm(lsa[mask] - kmeans.cluster_centers_[cluster_id], axis=1)
        examples = group.iloc[np.argsort(center_dist)[:3]]["body"].tolist()
        style = _style(group)
        label = f"Segment {cluster_id}: " + ", ".join(keywords[:3])
        rows.append({
            "segment_id": cluster_id,
            "segment_label": label,
            "editable_label": label,
            "n_emails": int(mask.sum()),
            "keywords": keywords,
            "sample_emails": examples,
            **style,
        })
    return rows


def _select_k(lsa):
    candidates = []
    for k in config.KMEANS_K_RANGE:
        if k >= len(lsa):
            continue
        model = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        labels = model.fit_predict(lsa)
        min_cluster = np.bincount(labels).min()
        if min_cluster < max(2, int(np.ceil(0.02 * len(lsa)))):
            log.info("k=%d rejected: smallest cluster=%d (<2%%)", k, min_cluster)
            continue
        score = silhouette_score(lsa, labels, sample_size=min(2000, len(lsa)),
                                 random_state=config.RANDOM_SEED)
        candidates.append((float(score), k, model, int(min_cluster)))
        log.info("k=%d train silhouette=%.4f min_cluster=%d", k, score, min_cluster)
    if not candidates:
        raise ValueError("No k in 5..12 satisfies the 2% minimum cluster size")
    candidates.sort(reverse=True, key=lambda x: x[0])
    best = candidates[0]
    pd.DataFrame([
        {"k": k, "train_silhouette": score, "min_cluster": minimum,
         "selected": k == best[1]}
        for score, k, _, minimum in candidates
    ]).to_csv(config.OUTPUTS_DIR / "clustering_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([x[1] for x in candidates], [x[2].inertia_ for x in candidates], "bo-")
    axes[0].set(xlabel="k", ylabel="Inertia", title="Train elbow")
    axes[1].plot([x[1] for x in candidates], [x[0] for x in candidates], "rs-")
    axes[1].set(xlabel="k", ylabel="Silhouette", title="Train silhouette")
    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "elbow_silhouette.png", dpi=120)
    plt.close()
    return best[1], best[0], best[2], candidates


def train_all(df=None):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.AESLC_CSV) if df is None else df.copy()
    df["tokens"] = df["tokens"].map(_parse_tokens)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    dev_df = df[df["split"] == "dev"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("AESLC train and test splits must both contain records")

    train_texts, test_texts = train_df["clean"].tolist(), test_df["clean"].tolist()
    bow = CountVectorizer(max_features=config.TFIDF_MAX_FEAT).fit(train_texts)
    joblib.dump(bow, config.MODELS_DIR / "bow_vectorizer.joblib")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=config.TFIDF_MAX_FEAT,
                            min_df=2, max_df=0.95, sublinear_tf=True)
    x_train = tfidf.fit_transform(train_texts)
    x_test = tfidf.transform(test_texts)
    joblib.dump(tfidf, config.MODELS_DIR / "tfidf_vectorizer.joblib")
    save_npz(str(config.MODELS_DIR / "tfidf_matrix.npz"), x_train)
    save_npz(str(config.MODELS_DIR / "tfidf_test_matrix.npz"), x_test)
    svd = TruncatedSVD(n_components=_components(x_train), random_state=config.RANDOM_SEED)
    lsa_train, lsa_test = svd.fit_transform(x_train), svd.transform(x_test)
    joblib.dump(svd, config.MODELS_DIR / "lsa_svd.joblib")

    best_k, train_sil, kmeans, comparisons = _select_k(lsa_train)
    joblib.dump(kmeans, config.MODELS_DIR / "kmeans.joblib")
    train_df["segment_id"] = kmeans.labels_
    test_segments = kmeans.predict(lsa_test)
    profiles = _profiles(train_df, x_train, tfidf, kmeans, lsa_train)
    (config.MODELS_DIR / "segment_profiles.json").write_text(json.dumps(profiles, indent=2))
    pd.DataFrame(profiles).to_csv(config.OUTPUTS_DIR / "segments.csv", index=False)
    emb.train(train_df["tokens"].tolist())
    emb.save(config.MODELS_DIR / "embeddings.npz")
    np.save(config.MODELS_DIR / "email_vecs_w2v.npy",
            np.array([emb.email_vector(t) for t in train_df["tokens"]]))
    np.save(config.MODELS_DIR / "test_email_vecs_w2v.npy",
            np.array([emb.email_vector(t) for t in test_df["tokens"]]))
    np.save(config.MODELS_DIR / "test_lsa.npy", lsa_test)
    np.save(config.MODELS_DIR / "test_segments.npy", test_segments)
    (config.MODELS_DIR / "train_corpus.csv").write_text(train_df.to_csv(index=False))
    (config.MODELS_DIR / "test_corpus.csv").write_text(test_df.to_csv(index=False))
    summary = {"train_rows": len(train_df), "dev_rows": len(dev_df), "test_rows": len(test_df),
               "best_k": best_k, "train_silhouette": train_sil, "random_state": config.RANDOM_SEED}
    (config.OUTPUTS_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return {"best_k": best_k, "silhouette": train_sil, "profiles": profiles,
            "train_rows": len(train_df), "test_rows": len(test_df)}


if __name__ == "__main__":
    result = train_all()
    print(f"Training complete. k={result['best_k']} silhouette={result['silhouette']:.4f}")
