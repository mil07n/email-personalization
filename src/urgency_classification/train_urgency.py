"""
Training and Evaluation Pipeline for Urgency Classification.
Compares Logistic Regression, Linear SVM, and Naive Bayes on the Customer Care dataset.
Saves the best model and an evaluation report.
"""

import os
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Project paths
ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config.config import (
    CUSTOMER_CARE_CSV, MODELS_DIR,
    EVALUATION_DIR, URGENCY_MODEL_PATH,
    URGENCY_VECTORIZER_PATH, URGENCY_METRICS_PATH
)
from src.preprocessing.preprocessor import EmailPreprocessor


def train_urgency_models():
    print("=" * 60)
    print(" TRAINING URGENCY CLASSIFIERS")
    print("=" * 60)

    # 1. Load dataset
    print(f"Loading Customer Care data from {CUSTOMER_CARE_CSV}...")
    df = pd.read_csv(CUSTOMER_CARE_CSV)
    print(f"Loaded {len(df)} records.")

    # 2. Preprocess text
    preprocessor = EmailPreprocessor()
    df["clean_body"] = df["message_body"].fillna("").map(preprocessor.clean_for_nlp)
    df["clean_subject"] = df["subject"].fillna("").map(preprocessor.clean_for_nlp)
    df["text"] = df["clean_subject"] + " " + df["clean_body"]

    # Target: email_criticality (standardized to lowercase: high, medium, low)
    df["label"] = df["email_criticality"].fillna("medium").str.lower().str.strip()
    valid_labels = {"high", "medium", "low"}
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)

    print("Class distribution:")
    print(df["label"].value_counts())

    # 3. Train-Test Split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # 4. Feature Extraction: TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Model Candidates
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "LinearSVC": CalibratedClassifierCV(LinearSVC(random_state=42, dual=False, max_iter=2000)),
        "MultinomialNB": MultinomialNB(alpha=0.5)
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_model = None

    classes = ["low", "medium", "high"]

    for name, model in candidates.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        p_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        r_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
        report = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(p_macro), 4),
            "recall_macro": round(float(r_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "confusion_matrix": {
                "labels": classes,
                "matrix": cm
            },
            "per_class": {
                c: {
                    "precision": round(report[c]["precision"], 4),
                    "recall": round(report[c]["recall"], 4),
                    "f1": round(report[c]["f1-score"], 4),
                    "support": report[c]["support"]
                } for c in classes if c in report
            }
        }

        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Macro F1:    {f1_macro:.4f}")
        print(f"  Weighted F1: {f1_weighted:.4f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name
            best_model = model

    print("\n" + "=" * 60)
    print(f" WINNING MODEL: {best_model_name} (Macro F1 = {best_f1:.4f})")
    print("=" * 60)

    # 6. Save winning model & vectorizer
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, URGENCY_MODEL_PATH)
    joblib.dump(vectorizer, URGENCY_VECTORIZER_PATH)
    print(f"Saved model to: {URGENCY_MODEL_PATH}")
    print(f"Saved vectorizer to: {URGENCY_VECTORIZER_PATH}")

    # 7. Save metrics report
    evaluation_output = {
        "best_model": best_model_name,
        "dataset": "Customer Care Emails (rtweera/customer-care-emails)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classes": classes,
        "models": results
    }

    with open(URGENCY_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_output, f, indent=2)
    print(f"Saved metrics JSON to: {URGENCY_METRICS_PATH}")

    # Generate Markdown Report
    md_path = EVALUATION_DIR / "urgency_evaluation.md"
    lines = [
        "# Urgency Classification Model Evaluation Report\n",
        f"**Dataset:** Customer Care Emails (`dataset.csv`)  ",
        f"**Train Samples:** {len(X_train):,} | **Test Samples:** {len(X_test):,}  ",
        f"**Selected Best Model:** **{best_model_name}** (F1 Macro = {best_f1:.4f})\n",
        "## Model Comparison\n",
        "| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for mname, mres in results.items():
        star = " 🏆 (Selected)" if mname == best_model_name else ""
        lines.append(
            f"| **{mname}**{star} | {mres['accuracy']:.4f} | {mres['precision_macro']:.4f} | "
            f"{mres['recall_macro']:.4f} | **{mres['f1_macro']:.4f}** | {mres['f1_weighted']:.4f} |"
        )

    lines.append("\n## Confusion Matrix (Best Model: " + best_model_name + ")\n")
    lines.append("| Actual \\ Predicted | Low | Medium | High |")
    lines.append("| :--- | :---: | :---: | :---: |")
    for idx, c in enumerate(classes):
        row = results[best_model_name]["confusion_matrix"]["matrix"][idx]
        lines.append(f"| **{c.capitalize()}** | {row[0]} | {row[1]} | {row[2]} |")

    lines.append("\n## Per-Class Breakdown\n")
    lines.append("| Priority Level | Precision | Recall | F1-Score | Support |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for c in classes:
        pc = results[best_model_name]["per_class"][c]
        lines.append(f"| **{c.capitalize()}** | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['support']} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved evaluation markdown to: {md_path}")

    return best_model_name, best_f1


if __name__ == "__main__":
    train_urgency_models()
