"""
Action Extractor Module
Extracts action items from email text using regex patterns, NLTK POS tagging,
and optional spaCy NER (falls back to regex+NLTK when spaCy is unavailable).
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional spaCy import
# ---------------------------------------------------------------------------
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
    logger.info("spaCy loaded successfully.")
except Exception:
    _nlp = None
    _SPACY_AVAILABLE = False
    logger.info("spaCy not available — using regex+NLTK fallback for NER.")

# ---------------------------------------------------------------------------
# Optional NLTK import
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk import pos_tag, word_tokenize, ne_chunk, Tree
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Action phrase triggers
_ACTION_TRIGGERS = [
    # please <verb>
    r"please\s+(?P<verb>[a-z]+(?:\s+\w+){0,4})",
    # can you / could you / would you <verb>
    r"(?:can|could|would)\s+you\s+(?:please\s+)?(?P<verb>[a-z]+(?:\s+\w+){0,4})",
    # need to <verb>
    r"(?:need|needs|needed)\s+to\s+(?P<verb>[a-z]+(?:\s+\w+){0,4})",
    # make sure to <verb>
    r"make\s+sure\s+(?:to\s+)?(?P<verb>[a-z]+(?:\s+\w+){0,4})",
    # let me know / let us know
    r"(?:let\s+(?:me|us)\s+know)(?:\s+(?P<verb>.+?))?(?:\.|,|$)",
    # I need you to
    r"i\s+need\s+(?:you\s+to\s+)(?P<verb>[a-z]+(?:\s+\w+){0,4})",
    # action required
    r"action\s+required[:\s]+(?P<verb>.+?)(?:\.|,|$)",
    # kindly <verb>
    r"kindly\s+(?P<verb>[a-z]+(?:\s+\w+){0,4})",
    # Imperative sentences: verb at start of sentence (simple heuristic)
    r"(?:^|\.\s+|\n)(?P<verb>(?:send|review|update|fix|check|submit|confirm|"
    r"schedule|complete|provide|share|attend|fill|sign|approve|reply|respond|"
    r"contact|call|email|forward|upload|download|ensure|verify|test|deploy|"
    r"prepare|finalize|deliver|create|write|read|analyse|analyze|"
    r"look|follow)\s+[^.!?\n]{3,60})",
]
_ACTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _ACTION_TRIGGERS]

# Person name patterns (simple heuristic: Title-cased word pairs / common titles)
_PERSON_RE = re.compile(
    r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    r"|(?<![.!?]\s)(?<!\n)([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,15})\b",
    re.MULTILINE,
)

# Deadline patterns (reused from DeadlineExtractor to look for inline deadlines)
_DEADLINE_RE = re.compile(
    r"by\s+(?:(?:tomorrow|today|EOD|COB|noon|midnight|end\s+of\s+(?:day|week|month))"
    r"|(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:,?\s+\d{4})?)"
    r"|(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)"
    r"|(?:next\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|week|month))"
    r"|(?:\d+\s+(?:days?|hours?|weeks?))"
    r"|(?:\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# NER helpers
# ---------------------------------------------------------------------------
def _extract_persons_spacy(text: str) -> list:
    """Use spaCy NER to find PERSON entities."""
    if not _SPACY_AVAILABLE or _nlp is None:
        return []
    doc = _nlp(text[:5000])  # limit for performance
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]


def _extract_persons_nltk(text: str) -> list:
    """Use NLTK NE chunker to find PERSON entities."""
    if not _NLTK_AVAILABLE:
        return []
    try:
        tokens = word_tokenize(text[:3000])
        tagged = pos_tag(tokens)
        tree = ne_chunk(tagged, binary=False)
        persons = []
        for subtree in tree:
            if hasattr(subtree, "label") and subtree.label() == "PERSON":
                person = " ".join(w for w, t in subtree.leaves())
                persons.append(person)
        return persons
    except Exception as e:
        logger.debug(f"NLTK NER failed: {e}")
        return []


def _extract_persons_regex(text: str) -> list:
    """Fallback: regex-based person extraction."""
    persons = []
    for m in _PERSON_RE.finditer(text):
        if m.group(1):
            persons.append(m.group(1))
        elif m.group(2) and m.group(3):
            persons.append(f"{m.group(2)} {m.group(3)}")
    return list(dict.fromkeys(persons))


def extract_persons(text: str) -> list:
    """Extract person names using best available method."""
    if _SPACY_AVAILABLE:
        persons = _extract_persons_spacy(text)
    elif _NLTK_AVAILABLE:
        persons = _extract_persons_nltk(text)
    else:
        persons = []
    if not persons:
        persons = _extract_persons_regex(text)
    return list(dict.fromkeys(persons))


# ---------------------------------------------------------------------------
# Action Extractor class
# ---------------------------------------------------------------------------
class ActionExtractor:
    """
    Extract action items from email text.

    Returns a list of dicts with keys:
        action          : str   — the action phrase
        responsible_person : str | None — who should do it
        deadline        : str | None — any inline deadline
        confidence      : float — 0.0–1.0
    """

    def extract(self, text: str) -> list:
        """
        Extract action items from email text.

        Parameters
        ----------
        text : str
            Email body or subject text.

        Returns
        -------
        list[dict]
            [{action, responsible_person, deadline, confidence}]
        """
        if not isinstance(text, str) or not text.strip():
            return []

        persons = extract_persons(text)
        actions = []
        seen_actions = set()

        for regex in _ACTION_RE:
            for match in regex.finditer(text):
                verb_text = match.group("verb") if "verb" in regex.groupindex else match.group(0)
                if not verb_text:
                    continue

                action_str = verb_text.strip()
                action_lower = action_str.lower()

                # Deduplicate
                if action_lower in seen_actions:
                    continue
                if any(action_lower in s or s in action_lower for s in seen_actions):
                    continue
                seen_actions.add(action_lower)

                # Skip very short or trivial matches
                if len(action_str.split()) < 2:
                    continue

                # Find inline deadline near this match
                context_start = max(0, match.start() - 100)
                context_end = min(len(text), match.end() + 150)
                context = text[context_start:context_end]
                dl_match = _DEADLINE_RE.search(context)
                deadline = dl_match.group(0).strip() if dl_match else None

                # Assign responsible person: first person found near match
                responsible = persons[0] if persons else None

                # Confidence based on trigger specificity
                trigger_idx = _ACTION_RE.index(regex)
                # Earlier (more specific) patterns get higher confidence
                confidence = round(max(0.4, 0.95 - trigger_idx * 0.07), 2)

                actions.append({
                    "action": action_str,
                    "responsible_person": responsible,
                    "deadline": deadline,
                    "confidence": confidence,
                })

        # Limit to top 10 by confidence
        actions.sort(key=lambda x: x["confidence"], reverse=True)
        return actions[:10]
