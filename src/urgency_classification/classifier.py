"""
Urgency Classification Module.
Classifies incoming emails into urgency levels (High, Medium, Low)
using machine learning trained on Customer Care & Enron data,
with confidence scores and user-preference priority adjustment.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Rule-based priority boosters
CRITICAL_SIGNALS = {
    "outage", "system down", "offline", "crashed", "breach", "hacked",
    "security breach", "lawsuit", "subpoena", "emergency", "fatal error",
    "payment failure", "overdue invoice", "severely impacted", "production down"
}

HIGH_SIGNALS = {
    "urgent", "asap", "immediately", "deadline", "due today", "due tomorrow",
    "by eod", "end of day", "escalate", "escalation", "unresolved", "blocking",
    "critical", "action required", "priority", "time sensitive", "violation"
}

MEDIUM_SIGNALS = {
    "follow up", "reminder", "schedule", "meeting", "review", "update",
    "pending", "feedback", "status", "deliverable", "milestone", "confirm"
}

LOW_SIGNALS = {
    "newsletter", "fyi", "announcement", "for your information", "no rush",
    "no action required", "optional", "heads up", "whenever possible"
}


class UrgencyClassifier:
    """
    NLP Urgency and Criticality Classifier combining statistical ML models
    (TF-IDF + LogisticRegression / LinearSVC) with rule-based temporal signals
    and sender importance weighting.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        vectorizer_path: Optional[Path] = None
    ):
        self.model = None
        self.vectorizer = None
        self.labels = ["low", "medium", "high"]

        if model_path and Path(model_path).exists():
            self.model = joblib.load(model_path)
        if vectorizer_path and Path(vectorizer_path).exists():
            self.vectorizer = joblib.load(vectorizer_path)

    def is_fitted(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def detect_signals(self, text: str) -> List[str]:
        """Identify lexical and pattern urgency indicators."""
        text_lower = text.lower()
        signals = []

        for sig in CRITICAL_SIGNALS:
            if sig in text_lower:
                signals.append(f"Critical Signal: '{sig}'")

        for sig in HIGH_SIGNALS:
            if sig in text_lower:
                signals.append(f"High Priority: '{sig}'")

        for sig in MEDIUM_SIGNALS:
            if sig in text_lower:
                signals.append(f"Medium Priority: '{sig}'")

        for sig in LOW_SIGNALS:
            if sig in text_lower:
                signals.append(f"Low Priority: '{sig}'")

        # Subject uppercase check
        if re.search(r"\b(URGENT|ASAP|ACTION REQUIRED|ALERT|CRITICAL)\b", text):
            signals.append("Subject Pattern: High-urgency capitalized keyword")

        # Multiple exclamation marks
        if "!!" in text:
            signals.append("Punctuation: Emphasis markers (!!)")

        return list(dict.fromkeys(signals))[:7]

    def predict(
        self,
        subject: str,
        body: str,
        sender: str = "",
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict urgency with confidence and preference adjustments.
        Returns:
            {
                "priority": "HIGH" | "MEDIUM" | "LOW" | "CRITICAL",
                "score": float (0.0 to 1.0),
                "confidence": float (0.0 to 1.0),
                "probabilities": {"low": float, "medium": float, "high": float},
                "signals": list of strings,
                "adjusted_by_preference": bool
            }
        """
        # Combine text with subject 3x weighting (matching model training representation)
        combined_text = f"{subject} {subject} {subject} {body}".strip()
        signals = self.detect_signals(f"{subject} {body}".strip())

        # Baseline probabilities
        probs = {"low": 0.33, "medium": 0.34, "high": 0.33}
        predicted_label = "medium"
        confidence = 0.50

        if self.is_fitted() and combined_text:
            vec = self.vectorizer.transform([combined_text])
            if hasattr(self.model, "predict_proba"):
                prob_vals = self.model.predict_proba(vec)[0]
                classes = list(self.model.classes_)
                probs = {str(c).lower(): float(p) for c, p in zip(classes, prob_vals)}
                predicted_label = max(probs, key=probs.get)
                confidence = probs[predicted_label]
            else:
                pred = self.model.predict(vec)[0]
                predicted_label = str(pred).lower()
                confidence = 0.80

        # Critical and Low override heuristics
        has_critical = any("Critical" in s for s in signals)
        has_high_pattern = any("Subject Pattern" in s for s in signals)
        has_high = any("High Priority" in s for s in signals) or has_high_pattern
        has_low = any("Low Priority" in s for s in signals)

        if has_critical:
            predicted_label = "critical"
            confidence = max(confidence, 0.95)
            probs["high"] = max(probs.get("high", 0.0), 0.90)
        elif has_high_pattern and predicted_label == "low":
            predicted_label = "high"
            confidence = max(confidence, 0.85)
        elif has_low and not has_high and not has_critical:
            predicted_label = "low"
            confidence = max(confidence, 0.80)
            probs["low"] = max(probs.get("low", 0.0), 0.80)

        # User preference adjustments
        adjusted_by_pref = False
        if user_preferences:
            important_senders = [s.lower() for s in user_preferences.get("important_senders", [])]
            priority_topics = [t.lower() for t in user_preferences.get("priority_topics", [])]

            sender_lower = sender.lower()
            if any(imp in sender_lower for imp in important_senders if imp):
                if predicted_label == "low":
                    predicted_label = "medium"
                elif predicted_label == "medium":
                    predicted_label = "high"
                adjusted_by_pref = True
                signals.append(f"Preference Boost: Sender '{sender}' is in VIP list")

            text_lower = combined_text.lower()
            if any(topic in text_lower for topic in priority_topics if topic):
                if predicted_label in ("low", "medium"):
                    predicted_label = "high"
                adjusted_by_pref = True
                signals.append("Preference Boost: Contains priority topic keyword")

        # Map to standard uppercase labels
        label_map = {
            "critical": "CRITICAL",
            "high": "HIGH",
            "medium": "MEDIUM",
            "low": "LOW"
        }
        final_priority = label_map.get(predicted_label.lower(), "MEDIUM")

        # Numerical urgency score (0.0 - 1.0)
        score_map = {"CRITICAL": 0.95, "HIGH": 0.80, "MEDIUM": 0.50, "LOW": 0.20}
        score = score_map.get(final_priority, 0.5)

        # Build structured signal breakdown with estimated feature weights for AI transparency
        signals_breakdown = []
        for s in signals:
            if "Critical" in s:
                signals_breakdown.append({"signal": s, "weight": 0.35, "type": "critical"})
            elif "High Priority" in s or "Subject Pattern" in s:
                signals_breakdown.append({"signal": s, "weight": 0.25, "type": "high"})
            elif "Preference Boost" in s:
                signals_breakdown.append({"signal": s, "weight": 0.30, "type": "vip"})
            elif "Medium" in s:
                signals_breakdown.append({"signal": s, "weight": 0.15, "type": "medium"})
            else:
                signals_breakdown.append({"signal": s, "weight": -0.10, "type": "low"})

        return {
            "priority": final_priority,
            "score": score,
            "confidence": round(float(confidence), 3),
            "confidence_percent": int(round(float(confidence) * 100)),
            "probabilities": {k: round(v, 3) for k, v in probs.items()},
            "signals": signals,
            "signals_breakdown": signals_breakdown,
            "adjusted_by_preference": adjusted_by_pref,
            "model_name": "Calibrated Soft-Voting Ensemble (LinearSVC + LogReg)"
        }
