"""
Urgency Classifier Module
Trains TF-IDF + ML models to classify email urgency as HIGH / MEDIUM / LOW.
Selects the best model by weighted F1 score and persists it via joblib.
"""

import os
import re
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"
LABELS = [HIGH, MEDIUM, LOW]

# ---------------------------------------------------------------------------
# Rule-based labeller for ENRON / AESLC fallback
# ---------------------------------------------------------------------------
_HIGH_KW = re.compile(
    r"\b(urgent|urgently|asap|immediately|critical|emergency|deadline|"
    r"overdue|action required|time.sensitive|priority|must|crucial|"
    r"important|help|problem|issue|broken|error|fail)\b",
    re.IGNORECASE,
)
_LOW_KW = re.compile(
    r"\b(newsletter|unsubscribe|digest|weekly|monthly|update|announcement|"
    r"fyi|info|no.reply|noreply|bulletin|notification|promo|offer|deal|"
    r"sale|discount)\b",
    re.IGNORECASE,
)
_MEDIUM_KW = re.compile(
    r"\b(meeting|reminder|schedule|appointment|conference|call|review|"
    r"follow.up|followup|please|request|question|check|confirm|"
    r"tomorrow|next week|upcoming)\b",
    re.IGNORECASE,
)


def rule_label(text: str) -> str:
    """Assign a label by keyword matching (fallback training labeller)."""
    if not isinstance(text, str):
        return MEDIUM
    if _HIGH_KW.search(text):
        return HIGH
    if _LOW_KW.search(text):
        return LOW
    if _MEDIUM_KW.search(text):
        return MEDIUM
    return MEDIUM  # neutral default


# ---------------------------------------------------------------------------
# Signal extraction helper
# ---------------------------------------------------------------------------
def _extract_signals(text: str) -> list:
    """Return keyword signals found in text."""
    signals = []
    if _HIGH_KW.search(text):
        signals.extend(re.findall(_HIGH_KW.pattern, text, re.IGNORECASE)[:3])
    if _MEDIUM_KW.search(text):
        signals.extend(re.findall(_MEDIUM_KW.pattern, text, re.IGNORECASE)[:2])
    if _LOW_KW.search(text):
        signals.extend(re.findall(_LOW_KW.pattern, text, re.IGNORECASE)[:2])
    return list(dict.fromkeys(signals))  # dedup while preserving order


# ---------------------------------------------------------------------------
# Main classifier class
# ---------------------------------------------------------------------------
class UrgencyClassifier:
    """
    Train, predict, evaluate and persist an urgency classifier.

    Usage
    -----
    clf = UrgencyClassifier()
    clf.train(df, text_col="subject", label_col="urgency")
    result = clf.predict("URGENT: server is down!")
    clf.save("models/urgency_classifier.joblib")
    """

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.best_model_name: str = ""
        self.classes_: list = []

    # ------------------------------------------------------------------
    def _build_candidates(self) -> dict:
        """Return candidate pipelines."""
        tfidf_common = dict(
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
            min_df=2,
        )
        return {
            "LogisticRegression": Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_common)),
                ("clf", LogisticRegression(
                    max_iter=1000, class_weight="balanced", C=1.0, random_state=42
                )),
            ]),
            "LinearSVC": Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_common)),
                ("clf", CalibratedClassifierCV(
                    LinearSVC(max_iter=2000, class_weight="balanced", C=1.0, random_state=42),
                    cv=3,
                )),
            ]),
            "MultinomialNB": Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=20_000,
                    min_df=2,
                )),
                ("clf", MultinomialNB(alpha=0.5)),
            ]),
        }

    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame, text_col: str, label_col: str):
        """
        Train and select the best urgency classifier.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataframe.
        text_col : str
            Column with email text (subject / body).
        label_col : str
            Column with urgency labels (HIGH / MEDIUM / LOW).
        """
        if df is None or df.empty:
            raise ValueError("Training dataframe is empty.")
        if text_col not in df.columns:
            raise ValueError(f"text_col '{text_col}' not found.")
        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' not found.")

        df = df[[text_col, label_col]].dropna()
        df[text_col] = df[text_col].astype(str)
        df[label_col] = df[label_col].str.upper().str.strip()
        df = df[df[label_col].isin(LABELS)]

        if len(df) < 10:
            raise ValueError(
                f"Need at least 10 labelled samples, got {len(df)}."
            )

        X = df[text_col].tolist()
        y = df[label_col].tolist()

        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = list(self.label_encoder.classes_)

        candidates = self._build_candidates()
        best_name, best_score, best_pipeline = "", -1.0, None

        for name, pipe in candidates.items():
            try:
                scores = cross_val_score(
                    pipe, X, y_enc,
                    cv=min(5, len(set(y_enc))),
                    scoring="f1_weighted",
                    n_jobs=-1,
                )
                mean_f1 = float(np.mean(scores))
                logger.info(f"{name}: CV F1={mean_f1:.4f}")
                if mean_f1 > best_score:
                    best_score = mean_f1
                    best_name = name
                    best_pipeline = pipe
            except Exception as e:
                logger.warning(f"{name} failed CV: {e}")

        if best_pipeline is None:
            raise RuntimeError("All candidate models failed during CV.")

        # Fit on full data
        best_pipeline.fit(X, y_enc)
        self.pipeline = best_pipeline
        self.best_model_name = best_name
        logger.info(f"Selected model: {best_name} (CV F1={best_score:.4f})")

    # ------------------------------------------------------------------
    def predict(self, text: str) -> dict:
        """
        Predict urgency for a single text.

        Returns
        -------
        dict
            {label: str, confidence: float, signals: list[str]}
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        if not isinstance(text, str) or not text.strip():
            return {"label": MEDIUM, "confidence": 0.0, "signals": []}

        proba = self.pipeline.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        label = self.label_encoder.inverse_transform([idx])[0]
        confidence = float(proba[idx])
        signals = _extract_signals(text)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "signals": signals,
        }

    # ------------------------------------------------------------------
    def evaluate(self, X_test, y_test) -> dict:
        """
        Evaluate the trained model on a held-out test set.

        Parameters
        ----------
        X_test : list[str]
            Test texts.
        y_test : list[str]
            True urgency labels (HIGH / MEDIUM / LOW).

        Returns
        -------
        dict
            accuracy, precision, recall, f1, confusion_matrix, report
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")

        y_true = [l.upper().strip() for l in y_test]
        y_true_enc = self.label_encoder.transform(y_true)
        y_pred_enc = self.pipeline.predict(list(X_test))
        y_pred = self.label_encoder.inverse_transform(y_pred_enc)
        y_true_labels = self.label_encoder.inverse_transform(y_true_enc)

        cm = confusion_matrix(y_true_enc, y_pred_enc).tolist()
        report = classification_report(
            y_true_labels, y_pred, target_names=self.classes_, output_dict=True
        )

        return {
            "model": self.best_model_name,
            "accuracy": round(float(accuracy_score(y_true_enc, y_pred_enc)), 4),
            "precision": round(float(precision_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)), 4),
            "f1": round(float(f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0)), 4),
            "confusion_matrix": cm,
            "classes": self.classes_,
            "classification_report": report,
        }

    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save the trained model + label encoder to disk."""
        if self.pipeline is None:
            raise RuntimeError("No trained model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "label_encoder": self.label_encoder,
                "best_model_name": self.best_model_name,
                "classes_": self.classes_,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load a previously saved model from disk."""
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.label_encoder = data["label_encoder"]
        self.best_model_name = data.get("best_model_name", "")
        self.classes_ = data.get("classes_", [])
        logger.info(f"Model loaded from {path}")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def build_training_data_from_enron(enron_root: str) -> pd.DataFrame:
    """
    Walk the AESLC enron_subject_line directory tree and produce a
    rule-labelled training dataframe with columns [text, urgency].
    """
    rows = []
    base = Path(enron_root)
    if not base.exists():
        logger.warning(f"ENRON root not found: {enron_root}")
        return pd.DataFrame(columns=["text", "urgency"])

    for fpath in base.rglob("*.subject"):
        try:
            subject = fpath.read_text(encoding="utf-8", errors="replace").strip()
            if subject:
                rows.append({"text": subject, "urgency": rule_label(subject)})
        except Exception:
            pass

    # Also read plain text files (some AESLC splits store one email per file)
    for fpath in base.rglob("*.txt"):
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                # Use first line as "subject proxy"
                first_line = text.splitlines()[0][:200]
                rows.append({"text": first_line, "urgency": rule_label(first_line)})
        except Exception:
            pass

    df = pd.DataFrame(rows)
    logger.info(f"Enron training set: {len(df)} samples from {enron_root}")
    return df


def train_and_save(
    project_root: str = r"C:\Users\tejai\OneDrive\Desktop\AESLC-Pipeline",
):
    """
    Full training pipeline: auto-detect data source → train → evaluate → save.
    Called from scripts or tests.
    """
    project_root = Path(project_root)
    model_path = project_root / "models" / "urgency_classifier.joblib"
    eval_path = project_root / "evaluation" / "urgency_evaluation.json"

    # Ensure directories exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Source 1: Customer Care CSV ---
    cc_dir = project_root / "datasets" / "customer_care" / "raw"
    df_train = pd.DataFrame()
    text_col, label_col = "text", "urgency"

    if cc_dir.exists():
        csv_files = list(cc_dir.glob("*.csv"))
        for csv_file in csv_files:
            try:
                tmp = pd.read_csv(csv_file)
                # Look for label columns
                label_candidates = [
                    c for c in tmp.columns
                    if any(kw in c.lower() for kw in ["priority", "urgency", "criticality", "severity", "label"])
                ]
                text_candidates = [
                    c for c in tmp.columns
                    if any(kw in c.lower() for kw in ["subject", "body", "text", "content", "message", "email"])
                ]
                if label_candidates and text_candidates:
                    tmp = tmp[[text_candidates[0], label_candidates[0]]].copy()
                    tmp.columns = ["text", "urgency"]
                    # Normalize labels to HIGH/MEDIUM/LOW
                    def normalize(v):
                        v = str(v).upper().strip()
                        if any(x in v for x in ["HIGH", "URGENT", "CRITICAL", "P1", "P2"]):
                            return HIGH
                        if any(x in v for x in ["LOW", "MINOR", "P4", "P5", "INFO"]):
                            return LOW
                        return MEDIUM
                    tmp["urgency"] = tmp["urgency"].apply(normalize)
                    df_train = pd.concat([df_train, tmp], ignore_index=True)
                    logger.info(f"Loaded {len(tmp)} rows from {csv_file.name}")
            except Exception as e:
                logger.warning(f"Could not load {csv_file}: {e}")

    # --- Source 2: Enron AESLC fallback ---
    if df_train.empty or len(df_train) < 100:
        enron_path = project_root / "datasets" / "enron" / "raw" / "AESLC-master" / "enron_subject_line"
        df_enron = build_training_data_from_enron(str(enron_path))
        df_train = pd.concat([df_train, df_enron], ignore_index=True)

    # --- Source 3: Synthetic fallback if still empty ---
    if df_train.empty or len(df_train) < 20:
        logger.warning("No real data found — using synthetic training set.")
        synthetic = [
            # HIGH
            ("URGENT: server is down, need fix immediately", HIGH),
            ("ASAP: critical bug in production system", HIGH),
            ("Emergency: data breach detected, action required now", HIGH),
            ("Deadline tomorrow: submit report or we lose the contract", HIGH),
            ("CRITICAL: payment processing failing for all customers", HIGH),
            ("Help! The system crashed and clients are calling", HIGH),
            ("Immediate action needed: database backup failed", HIGH),
            ("URGENT please respond: legal deadline today", HIGH),
            ("SOS: website unreachable, losing orders every minute", HIGH),
            ("Problem: access denied to production — fix ASAP", HIGH),
            # MEDIUM
            ("Meeting tomorrow at 10am to discuss Q4 plans", MEDIUM),
            ("Reminder: please review the attached document by Friday", MEDIUM),
            ("Follow up on our call last week regarding the proposal", MEDIUM),
            ("Conference call scheduled for next Tuesday", MEDIUM),
            ("Please confirm your attendance at the team meeting", MEDIUM),
            ("Can you check the status of the pending request?", MEDIUM),
            ("Reminder: performance review due next week", MEDIUM),
            ("Request: approval needed for budget allocation", MEDIUM),
            ("Question about the upcoming project milestone", MEDIUM),
            ("Please review and sign the attached agreement", MEDIUM),
            # LOW
            ("Monthly newsletter — top stories this week", LOW),
            ("Company update: new office hours starting next month", LOW),
            ("FYI: office will be closed on the public holiday", LOW),
            ("Announcement: new team member joining next Monday", LOW),
            ("Unsubscribe from our mailing list", LOW),
            ("Weekly digest: industry news and updates", LOW),
            ("Promo: 20% off all products this weekend only", LOW),
            ("No-reply: your subscription has been renewed", LOW),
            ("Info: parking lot will be closed for maintenance", LOW),
            ("Bulletin: cafeteria menu change effective next month", LOW),
        ]
        df_train = pd.DataFrame(synthetic, columns=["text", "urgency"])

    df_train = df_train.dropna(subset=["text", "urgency"])
    df_train["text"] = df_train["text"].astype(str)
    df_train["urgency"] = df_train["urgency"].str.upper().str.strip()
    df_train = df_train[df_train["urgency"].isin(LABELS)]

    logger.info(
        f"Training set: {len(df_train)} samples | "
        f"Label dist: {df_train['urgency'].value_counts().to_dict()}"
    )

    # Train/test split
    test_size = min(0.2, max(0.1, 20 / len(df_train)))
    X = df_train["text"].tolist()
    y = df_train["urgency"].tolist()

    if len(df_train) >= 10:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    df_fit = pd.DataFrame({"text": X_train, "urgency": y_train})

    clf = UrgencyClassifier()
    clf.train(df_fit, "text", "urgency")

    # Evaluate
    eval_results = clf.evaluate(X_test, y_test)
    eval_results["train_samples"] = len(X_train)
    eval_results["test_samples"] = len(X_test)
    eval_results["label_distribution"] = df_train["urgency"].value_counts().to_dict()

    # Save
    clf.save(str(model_path))

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, default=str)

    logger.info(
        f"Evaluation saved to {eval_path}  |  "
        f"F1={eval_results['f1']:.4f}  Accuracy={eval_results['accuracy']:.4f}"
    )

    return clf, eval_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clf, results = train_and_save()
    print(json.dumps(results, indent=2, default=str))
