"""Urgency classification package."""

from .classifier import UrgencyClassifier, rule_label, train_and_save

__all__ = ["UrgencyClassifier", "rule_label", "train_and_save"]
