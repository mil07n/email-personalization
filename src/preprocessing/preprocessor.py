"""
Email Preprocessing Module.
Handles HTML removal, boilerplate noise cleaning, tokenization,
and metadata extraction.
"""

import re
import html
import unicodedata
from typing import List, Dict, Any, Optional

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure required NLTK resources are available
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


class EmailPreprocessor:
    """
    Standard preprocessor for incoming and historical email communications.
    """

    # Noise and boilerplate patterns
    RE_HTML = re.compile(r"<[^>]+>")
    RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
    RE_PHONE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-])\d{3}[\s.-]\d{4}(?!\w)")
    RE_FORWARD_HEADER = re.compile(
        r"(?im)^(?:[-= ]*(?:forwarded by|original message|begin forwarded message)"
        r"|from:.*|sent:.*|to:.*|cc:.*|subject:.*)\s*$"
    )
    RE_QUOTED_REPLY = re.compile(r"(?m)^\s*>.*(?:\n|$)")
    RE_DISCLAIMER = re.compile(
        r"(?is)(?:this (?:e-?mail|message).{0,120}(?:confidential|intended recipient|privileged|prohibited).*)$"
    )

    def __init__(self, preserve_punctuation: bool = False):
        self.preserve_punctuation = preserve_punctuation

    def clean_html(self, text: str) -> str:
        """Strip HTML tags and unescape HTML entities."""
        if not text:
            return ""
        text = html.unescape(str(text))
        text = self.RE_HTML.sub(" ", text)
        return text

    def clean_noise(self, text: str) -> str:
        """Remove email noise: quoted replies, forwarded headers, disclaimers."""
        if not text:
            return ""
        text = self.RE_QUOTED_REPLY.sub(" ", text)
        text = self.RE_FORWARD_HEADER.sub(" ", text)
        text = self.RE_DISCLAIMER.sub(" ", text)
        return text

    def normalize_text(self, text: str, lowercase: bool = False) -> str:
        """Normalize whitespace, unicode characters, and optionally lowercase."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", str(text))
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        if lowercase:
            text = text.lower()
        return text

    def clean_for_nlp(self, text: str, remove_contacts: bool = False) -> str:
        """Full cleaning pipeline for NLP models."""
        text = self.clean_html(text)
        text = self.clean_noise(text)
        if remove_contacts:
            text = self.RE_URL.sub(" ", text)
            text = self.RE_EMAIL.sub(" ", text)
            text = self.RE_PHONE.sub(" ", text)
        return self.normalize_text(text)

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentence strings using NLTK or regex fallback."""
        cleaned = self.clean_html(text).strip()
        if not cleaned:
            return []
        try:
            sentences = sent_tokenize(cleaned)
        except Exception:
            sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return [s.strip() for s in sentences if len(s.strip()) > 3]

    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        cleaned = self.clean_for_nlp(text)
        try:
            return word_tokenize(cleaned)
        except Exception:
            return re.findall(r"\b\w+\b", cleaned)

    def extract_metadata(self, raw_email: str) -> Dict[str, Any]:
        """
        Parse raw RFC822-style or plain formatted email text for metadata.
        """
        metadata = {
            "subject": "",
            "from": "",
            "to": "",
            "date": "",
            "body": raw_email
        }
        lines = raw_email.splitlines()
        body_lines = []
        is_header = True

        for line in lines:
            if is_header and line.strip() == "":
                is_header = False
                continue

            if is_header:
                m_from = re.match(r"^from:\s*(.*)", line, re.IGNORECASE)
                m_to = re.match(r"^to:\s*(.*)", line, re.IGNORECASE)
                m_subj = re.match(r"^subject:\s*(.*)", line, re.IGNORECASE)
                m_date = re.match(r"^date:\s*(.*)", line, re.IGNORECASE)

                if m_from:
                    metadata["from"] = m_from.group(1).strip()
                elif m_to:
                    metadata["to"] = m_to.group(1).strip()
                elif m_subj:
                    metadata["subject"] = m_subj.group(1).strip()
                elif m_date:
                    metadata["date"] = m_date.group(1).strip()
                else:
                    body_lines.append(line)
            else:
                body_lines.append(line)

        metadata["body"] = "\n".join(body_lines).strip()
        return metadata
