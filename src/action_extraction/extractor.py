"""
Action-Item Extraction Module.
Detects task-intent sentences in emails, extracts the action required,
identifies the responsible person / addressee (informed by the EPA dataset),
and associates deadlines.
"""

import re
from typing import List, Dict, Any, Optional

from src.preprocessing.preprocessor import EmailPreprocessor
from src.deadline_extraction.extractor import DeadlineExtractor


# Imperative and task-intent trigger patterns
TASK_TRIGGER_PATTERNS = [
    # Polite requests & directives
    (r"\bplease\s+([a-z]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.90),
    (r"\bkindly\s+([a-z]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.85),
    (r"\bcould\s+you\s+(?:please\s+)?([a-z]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.85),
    (r"\bcan\s+you\s+(?:please\s+)?([a-z]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.80),
    (r"\bwould\s+you\s+(?:mind\s+)?(?:please\s+)?([a-z]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.80),

    # Necessity / Obligation
    (r"\bwe\s+need\s+(?:you\s+)?to\s+([a-z]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.90),
    (r"\byou\s+need\s+to\s+([a-z]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.90),
    (r"\bi\s+need\s+you\s+to\s+([a-z]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.92),
    (r"\bmake\s+sure\s+to\s+([a-z]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.85),
    (r"\bensure\s+that\s+([a-z]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.85),
    (r"\byou\s+(?:must|should|have\s+to)\s+([a-z]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.85),

    # Action required flags
    (r"\baction\s+required[:\s]+([a-z0-9,.'\"-]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.95),
    (r"\btodo[:\s]+([a-z0-9,.'\"-]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.95),
    (r"\bnext\s+steps?[:\s]+([a-z0-9,.'\"-]+(?:\s+[a-z0-9,.'\"-]+){1,10})", 0.85),

    # Direct task verbs
    (r"\b(?:prepare|submit|send|review|forward|approve|update|schedule|complete|verify|provide|fix|sign)\s+([a-z0-9,.'\"-]+(?:\s+[a-z0-9,.'\"-]+){1,8})", 0.75),
]

# Action verbs commonly starting actionable requests
ACTION_VERBS = {
    "prepare", "send", "submit", "review", "approve", "forward", "update",
    "schedule", "complete", "verify", "provide", "fix", "sign", "call",
    "contact", "check", "organize", "circulate", "distribute", "finalize",
    "follow", "write", "draft", "confirm", "share", "attach", "upload"
}


class ActionExtractor:
    """
    Extracts action items, responsible persons, and associated deadlines from email text.
    """

    def __init__(self):
        self.preprocessor = EmailPreprocessor()
        self.deadline_extractor = DeadlineExtractor()

    def extract_greeting_name(self, text: str) -> Optional[str]:
        """Detect addressee name from common salutations (e.g. 'Hi John,', 'Chris -')."""
        lines = text.strip().splitlines()
        first_lines = lines[:4] if len(lines) >= 4 else lines

        for line in first_lines:
            line_str = line.strip()
            # E.g. "Hi John," or "Dear Sarah:" or "Chris -" or "Hello Alice,"
            m = re.match(r"^(?:hi|hello|dear|hey)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", line_str, re.IGNORECASE)
            if m:
                return m.group(1).title()

            m_dash = re.match(r"^([A-Z][a-z]+)\s*[-:]\s*$", line_str)
            if m_dash:
                return m_dash.group(1).title()

        return None

    def extract_person_for_sentence(
        self,
        sentence: str,
        default_addressee: Optional[str] = None
    ) -> Optional[str]:
        """
        Identify who is assigned to this specific action item.
        Looks for leading names e.g. 'John, please prepare...' or defaults to recipient.
        """
        # Leading vocative pattern: e.g. "John, please..." or "Sarah - could you..."
        m_voc = re.match(r"^\s*([A-Z][a-z]+)\s*[,:-]\s*(?:please|can|could|kindly|you\s+need)", sentence, re.IGNORECASE)
        if m_voc:
            name_candidate = m_voc.group(1).title()
            if name_candidate.lower() not in {"hi", "hello", "dear", "thanks", "regards"}:
                return name_candidate

        return default_addressee

    def clean_action_phrase(self, raw_action: str) -> str:
        """Format raw captured text into a concise, professional action title."""
        # Trim punctuation
        action = re.sub(r"[,;:.!?]+$", "", raw_action.strip())
        # Remove trailing clauses like "and let me know", "before Friday"
        action = re.sub(r"\s+(?:before|by|due|prior to)\s+(?:friday|monday|tomorrow|today|eod|noon).*", "", action, flags=re.IGNORECASE)
        action = re.sub(r"\s+and\s+(?:let\s+me\s+know|thank\s+you|thanks).*", "", action, flags=re.IGNORECASE)

        # Capitalize first word
        words = action.split()
        if words:
            words[0] = words[0].capitalize()
            return " ".join(words[:12])
        return action

    def extract(
        self,
        text: str,
        recipient_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze email body and extract all actionable tasks.
        Returns:
            list of dicts:
            [{
                "action": "Prepare financial report",
                "responsible_person": "John",
                "deadline": "Friday",
                "confidence": 0.90,
                "source_sentence": "John, please prepare the financial report and send it before Friday."
            }]
        """
        sentences = self.preprocessor.tokenize_sentences(text)
        default_person = recipient_name or self.extract_greeting_name(text)

        action_items = []
        seen_actions = set()

        for sent in sentences:
            sent_clean = self.preprocessor.clean_for_nlp(sent)
            sent_lower = sent_clean.lower()

            matched = False
            best_action_text = ""
            best_conf = 0.0

            # 1. Check task trigger regex patterns
            for pattern, conf in TASK_TRIGGER_PATTERNS:
                m = re.search(pattern, sent_lower)
                if m:
                    candidate = m.group(1)
                    if len(candidate.split()) >= 2:
                        matched = True
                        if conf > best_conf:
                            best_conf = conf
                            best_action_text = candidate

            # 2. Check imperative verbs starting sentence
            if not matched:
                words = sent_lower.split()
                if words and words[0] in ACTION_VERBS and len(words) >= 3:
                    matched = True
                    best_conf = 0.75
                    best_action_text = " ".join(words[:8])

            if matched and best_action_text:
                action_formatted = self.clean_action_phrase(best_action_text)
                dedup_key = action_formatted.lower()[:30]

                if dedup_key and dedup_key not in seen_actions:
                    seen_actions.add(dedup_key)

                    # Extract deadline specifically associated with this sentence
                    sentence_deadlines = self.deadline_extractor.extract_deadlines(sent)
                    deadline_str = sentence_deadlines[0]["raw"] if sentence_deadlines else None

                    # Extract responsible person
                    person = self.extract_person_for_sentence(sent, default_person)

                    action_items.append({
                        "action": action_formatted,
                        "responsible_person": person or "Unassigned",
                        "deadline": deadline_str or "No deadline specified",
                        "confidence": round(best_conf, 2),
                        "source_sentence": sent.strip()
                    })

        return action_items

    def is_action_required(self, text: str) -> bool:
        """Distinguish action-required emails from informational emails."""
        return len(self.extract(text)) > 0
