"""
Unit and Integration Tests for NLP-Powered Executive Email Assistant.
"""

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.preprocessor import EmailPreprocessor
from src.urgency_classification.classifier import UrgencyClassifier
from src.action_extraction.extractor import ActionExtractor
from src.deadline_extraction.extractor import DeadlineExtractor
from src.conversation_grouping.grouper import ConversationGrouper
from src.summarization.summarizer import EmailSummarizer
from src.personalization.engine import PersonalizationEngine
from src.integration.pipeline import EmailAnalysisPipeline
from config.config import URGENCY_MODEL_PATH, URGENCY_VECTORIZER_PATH


class TestNLPExecutiveAssistant(unittest.TestCase):

    def setUp(self):
        self.preprocessor = EmailPreprocessor()
        self.urgency_clf = UrgencyClassifier(URGENCY_MODEL_PATH, URGENCY_VECTORIZER_PATH)
        self.action_ext = ActionExtractor()
        self.deadline_ext = DeadlineExtractor()
        self.grouper = ConversationGrouper()
        self.summarizer = EmailSummarizer()
        self.personalization = PersonalizationEngine()
        self.pipeline = EmailAnalysisPipeline()

    def test_preprocessing(self):
        raw = "<html><body><p>Hello world!</p><br/>Please review: https://example.com</body></html>"
        cleaned = self.preprocessor.clean_for_nlp(raw, remove_contacts=True)
        self.assertNotIn("<html>", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("Hello world!", cleaned)

    def test_urgency_classification(self):
        # Critical
        res_crit = self.urgency_clf.predict("URGENT: Production database down", "Server crashed, outage reported.")
        self.assertIn(res_crit["priority"], ["CRITICAL", "HIGH"])
        self.assertGreater(res_crit["confidence"], 0.5)

        # Low
        res_low = self.urgency_clf.predict("Company Newsletter", "Monthly updates for your information, no rush.")
        self.assertIn(res_low["priority"], ["LOW", "MEDIUM"])

    def test_action_item_extraction(self):
        email_text = "John, please prepare the quarterly financial report and send it before Friday."
        actions = self.action_ext.extract(email_text)
        self.assertGreater(len(actions), 0)
        first = actions[0]
        self.assertIn("financial report", first["action"].lower())
        self.assertEqual(first["responsible_person"], "John")
        self.assertIsNotNone(first["deadline"])

    def test_deadline_extraction(self):
        text = "Please submit the proposal by tomorrow afternoon and complete audit before Friday."
        deadlines = self.deadline_ext.extract_deadlines(text)
        self.assertGreaterEqual(len(deadlines), 2)
        raw_deadlines = [d["raw"].lower() for d in deadlines]
        self.assertTrue(any("tomorrow" in r for r in raw_deadlines))
        self.assertTrue(any("friday" in r for r in raw_deadlines))

    def test_conversation_grouping(self):
        emails = [
            {"subject": "Q4 Budget Review", "body": "Let's review the Q4 numbers.", "sender": "alice@co.com"},
            {"subject": "Re: Q4 Budget Review", "body": "Attached are the draft figures.", "sender": "bob@co.com"},
            {"subject": "Holiday Party Planning", "body": "Who wants to organize the dinner?", "sender": "carol@co.com"},
        ]
        convs = self.grouper.group_emails(emails)
        self.assertEqual(len(convs), 2)  # Q4 budget grouped into 1, Holiday Party into 1

    def test_summarization(self):
        thread = [
            {"sender": "Alice", "body": "We have an issue with client billing. The charges were duplicated."},
            {"sender": "Bob", "body": "I have reviewed the logs. We have decided to issue full refunds immediately."},
            {"sender": "Alice", "body": "Confirmed. All refunds have been processed."}
        ]
        res = self.summarizer.summarize_thread(thread)
        self.assertIn("summary", res)
        self.assertTrue(len(res["summary"]) > 20)

    def test_rouge_metric(self):
        hyp = "We agreed to issue refunds to all affected clients."
        ref = "The team decided to issue refunds for affected clients."
        metrics = EmailSummarizer.compute_rouge(hyp, ref)
        self.assertGreater(metrics["rouge1"]["f1"], 0.40)
        self.assertGreater(metrics["rougeL"]["f1"], 0.40)

    def test_personalization(self):
        prefs = self.personalization.get_preferences()
        self.assertIn("important_senders", prefs)
        email = {
            "subject": "Contract Discussion",
            "body": "Let's discuss the contract terms.",
            "sender": "ceo@company.com",
            "urgency": {"score": 0.5, "priority": "MEDIUM"}
        }
        personalized = self.personalization.personalize_email(email)
        self.assertGreater(personalized["personalized_score"], 0.70)
        self.assertTrue(personalized["is_highlighted"])

    def test_full_pipeline(self):
        email = {
            "subject": "URGENT: Legal compliance audit deadline",
            "body": "Alice, please compile the compliance documents before next Monday. The CEO has been notified.",
            "sender": "legal@company.com",
            "receiver": "Alice"
        }
        res = self.pipeline.analyze_email(
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            receiver=email["receiver"]
        )
        self.assertIn(res["urgency"]["priority"], ["CRITICAL", "HIGH"])
        self.assertGreater(len(res["action_items"]), 0)
        self.assertEqual(res["action_items"][0]["responsible_person"], "Alice")
        self.assertIn("summary", res)
        self.assertTrue(res["is_highlighted"])


if __name__ == "__main__":
    unittest.main()
