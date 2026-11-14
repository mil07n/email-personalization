"""
Master Email Intelligence Integration Pipeline.
Orchestrates Preprocessing, Urgency Classification, Action Item Extraction,
Deadline Extraction, Conversation Grouping, Thread Summarization,
and Personalization.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config.config import URGENCY_MODEL_PATH, URGENCY_VECTORIZER_PATH
from src.preprocessing.preprocessor import EmailPreprocessor
from src.urgency_classification.classifier import UrgencyClassifier
from src.action_extraction.extractor import ActionExtractor
from src.deadline_extraction.extractor import DeadlineExtractor
from src.conversation_grouping.grouper import ConversationGrouper
from src.summarization.summarizer import EmailSummarizer
from src.personalization.engine import PersonalizationEngine


class EmailAnalysisPipeline:
    """
    Unified end-to-end NLP assistant pipeline for executive email intelligence.
    """

    def __init__(self):
        self.preprocessor = EmailPreprocessor()
        self.urgency_classifier = UrgencyClassifier(URGENCY_MODEL_PATH, URGENCY_VECTORIZER_PATH)
        self.action_extractor = ActionExtractor()
        self.deadline_extractor = DeadlineExtractor()
        self.grouper = ConversationGrouper()
        self.summarizer = EmailSummarizer(self.preprocessor)
        self.personalization_engine = PersonalizationEngine()

    def analyze_email(
        self,
        subject: str,
        body: str,
        sender: str = "",
        receiver: str = "",
        timestamp: str = "",
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run full NLP analysis on a single incoming email.
        """
        clean_subj = self.preprocessor.clean_for_nlp(subject)
        clean_body = self.preprocessor.clean_for_nlp(body)

        # 1. User preferences
        prefs = self.personalization_engine.get_preferences()

        # 2. Urgency classification
        urgency_result = self.urgency_classifier.predict(
            subject=clean_subj,
            body=clean_body,
            sender=sender,
            user_preferences=prefs
        )

        # 3. Action items & addressee extraction
        action_items = self.action_extractor.extract(body, recipient_name=receiver)

        # 4. Deadlines extraction
        deadlines = self.deadline_extractor.extract_deadlines(f"{subject} {body}")

        # 5. Extractive summary
        summary = self.summarizer.summarize_single(body, max_sentences=2)

        # 6. Assembly
        email_doc = {
            "subject": subject,
            "body": body,
            "sender": sender or "Unknown",
            "receiver": receiver or "Me",
            "timestamp": timestamp or "Just now",
            "thread_id": thread_id,
            "urgency": urgency_result,
            "action_items": action_items,
            "deadlines": deadlines,
            "summary": summary
        }

        # 7. Apply personalization scoring
        personalized_doc = self.personalization_engine.personalize_email(email_doc)
        return personalized_doc

    def analyze_batch(self, emails: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process multiple emails, cluster into conversations, generate thread summaries,
        and compile an executive personalized daily briefing.
        """
        if not emails:
            return {
                "emails": [],
                "conversations": [],
                "daily_summary": {},
                "stats": {
                    "total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0,
                    "action_items": 0, "deadlines": 0
                }
            }

        # 1. Analyze each email individually
        analyzed_emails = []
        for em in emails:
            analyzed = self.analyze_email(
                subject=em.get("subject", ""),
                body=em.get("body", "") or em.get("message_body", ""),
                sender=em.get("sender", ""),
                receiver=em.get("receiver", ""),
                timestamp=em.get("timestamp", ""),
                thread_id=em.get("thread_id")
            )
            analyzed_emails.append(analyzed)

        # 2. Group into conversations
        conversations = self.grouper.group_emails(analyzed_emails)

        # 3. Summarize each conversation thread
        for conv in conversations:
            thread_msgs = [
                {"sender": e.get("sender"), "body": e.get("body"), "subject": e.get("subject")}
                for e in conv.get("emails", [])
            ]
            conv_summary_obj = self.summarizer.summarize_thread(thread_msgs, max_sentences=3)
            conv["summary"] = conv_summary_obj["summary"]
            conv["key_points"] = conv_summary_obj["key_points"]
            conv["action_items"] = conv_summary_obj["action_items"]

        # 4. Compile personalized daily executive summary
        daily_summary = self.personalization_engine.compile_daily_summary(
            analyzed_emails,
            conversations
        )

        # 5. Compute global dashboard statistics
        stats = {
            "total": len(analyzed_emails),
            "critical": sum(1 for e in analyzed_emails if e["urgency"]["priority"] == "CRITICAL"),
            "high": sum(1 for e in analyzed_emails if e["urgency"]["priority"] == "HIGH"),
            "medium": sum(1 for e in analyzed_emails if e["urgency"]["priority"] == "MEDIUM"),
            "low": sum(1 for e in analyzed_emails if e["urgency"]["priority"] == "LOW"),
            "action_items": sum(len(e["action_items"]) for e in analyzed_emails),
            "deadlines": sum(len(e["deadlines"]) for e in analyzed_emails),
            "conversations_count": len(conversations)
        }

        return {
            "emails": analyzed_emails,
            "conversations": conversations,
            "daily_summary": daily_summary,
            "stats": stats
        }
