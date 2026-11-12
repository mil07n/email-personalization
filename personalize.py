"""AESLC-informed personalization with train-only retrieval artifacts."""

import ast
import json
import re
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config
import embeddings as emb
from preprocess import clean_text

_models = {}
GENERIC = {
    "about", "also", "and", "been", "being", "build", "building", "business",
    "company", "could", "focus", "focused", "have", "help", "including",
    "lead", "make", "manager", "need", "please", "project", "recent", "send",
    "specialize", "specializing", "team", "thing", "use", "using", "work",
    "working", "would", "your", "email", "message", "enron", "ect", "hou",
    "forwarded",
}


def _load():
    if _models:
        return _models
    m = {
        "tfidf": joblib.load(config.MODELS_DIR / "tfidf_vectorizer.joblib"),
        "svd": joblib.load(config.MODELS_DIR / "lsa_svd.joblib"),
        "kmeans": joblib.load(config.MODELS_DIR / "kmeans.joblib"),
        "segments": json.loads((config.MODELS_DIR / "segment_profiles.json").read_text()),
        "train": pd.read_csv(config.MODELS_DIR / "train_corpus.csv"),
        "train_matrix": __import__("scipy.sparse", fromlist=["load_npz"]).load_npz(
            str(config.MODELS_DIR / "tfidf_matrix.npz")
        ),
        "train_w2v": np.load(config.MODELS_DIR / "email_vecs_w2v.npy"),
        "templates": {},
    }
    m["train"]["tokens"] = m["train"]["tokens"].map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    emb.load(config.MODELS_DIR / "embeddings.npz")
    _models.update(m)
    return _models


def _noun_phrases(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
    try:
        import nltk
        tagged = nltk.pos_tag(words)
        phrases = []
        current = []
        for word, tag in tagged + [("", "STOP")]:
            if tag.startswith(("NN", "JJ")):
                current.append(word)
            elif current:
                phrases.append(" ".join(current))
                current = []
        return phrases
    except Exception:
        return words


def _extract_topics(profile_text, n=3):
    m = _load()
    text = profile_text.strip()
    if not text:
        return ["current priorities"]
    candidates = []
    for acronym in re.findall(r"\b[A-Z][A-Z0-9&-]{1,}\b", text):
        candidates.append((acronym, 3.0))
    for phrase in _noun_phrases(text):
        normalized = clean_text(phrase)
        if normalized and not any(word in GENERIC for word in normalized.split()):
            candidates.append((phrase.strip(), 2.0 if " " in phrase else 1.0))
    words = clean_text(text).split()
    if words:
        vec = m["tfidf"].transform([" ".join(words)]).toarray()[0]
        features = m["tfidf"].get_feature_names_out()
        for index in vec.argsort()[::-1][:20]:
            if vec[index] > 0:
                candidates.append((features[index], float(vec[index])))
    scores = {}
    for phrase, score in candidates:
        key = clean_text(phrase)
        if not key or any(word in GENERIC for word in key.split()):
            continue
        scores[key] = max(scores.get(key, 0), score)
    chosen = []
    for key, _ in sorted(scores.items(), key=lambda item: (-item[1], -len(item[0]))):
        if any(key in existing or existing in key for existing in chosen):
            continue
        chosen.append(key)
        if len(chosen) == n:
            break
    return chosen or ["current priorities"]


def _retrieve(profile_text, mode):
    m = _load()
    query_clean = clean_text(profile_text)
    query_lsa = m["svd"].transform(m["tfidf"].transform([query_clean]))
    if mode == "word2vec":
        query = emb.email_vector(query_clean.split()).reshape(1, -1)
        scores = cosine_similarity(query, m["train_w2v"]).ravel()
    elif mode == "random":
        scores = np.random.default_rng(config.RANDOM_SEED).random(len(m["train"]))
    else:
        scores = cosine_similarity(query_lsa, m["svd"].transform(m["train_matrix"])).ravel()
    top = scores.argsort()[-3:][::-1]
    return m, query_lsa, top, scores[top]


def personalize(name, profile_text, purpose="follow_up", retrieval="tfidf"):
    m, query_lsa, top, scores = _retrieve(profile_text, retrieval)
    topics = _extract_topics(profile_text)
    segment_id = int(m["kmeans"].predict(query_lsa)[0])
    segment = m["segments"][segment_id]
    first_name = str(name).strip().split()[0] if str(name).strip() else "there"
    greeting = segment.get("common_greeting", "Hello").strip()
    greeting = re.sub(r"\s+(there|all)\s*$", "", greeting, flags=re.I)
    signoff = segment.get("common_signoff", "Best").title()
    retrieved = m["train"].iloc[top]
    patterns = [str(subject).strip() for subject in retrieved["subject_line"] if str(subject).strip()]
    topic = topics[0]
    base_pattern = patterns[0] if patterns else f"Following up on {topic}"
    subject = re.sub(r"^(re|fw|fwd)\s*:\s*", "", base_pattern, flags=re.I).strip()
    subject = f"{subject[:65]} - {topic}" if topic.lower() not in subject.lower() else subject[:80]
    formal = segment.get("formality") == "formal"
    topic_phrase = ", ".join(topics)
    opening = f"{greeting}, {first_name}," if greeting.lower() not in {"hello", "hi", "hey"} else f"{greeting} {first_name},"
    if formal:
        sentence = f"I wanted to follow up regarding {topic_phrase}."
        close = "Please let me know if a discussion would be useful."
    else:
        sentence = f"I wanted to connect about {topic_phrase}."
        close = "Happy to compare notes if helpful."
    body = f"{opening}\n\n{sentence} {close}\n\n{signoff},\nMilan"
    similar = [
        {"subject": row["subject_line"], "snippet": row["body"][:180], "score": float(score)}
        for (_, row), score in zip(retrieved.iterrows(), scores)
    ]
    return {
        "segment_id": segment_id,
        "segment_label": segment.get("editable_label", segment["segment_label"]),
        "topics": topics,
        "tone": segment.get("formality", "conversational"),
        "retrieval_mode": retrieval,
        "similar_emails": similar,
        "subject": subject,
        "body": body,
    }
