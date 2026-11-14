"""
Email Preprocessor Module
Cleans, tokenizes, and extracts metadata from raw emails.
"""

import re
import html
from html.parser import HTMLParser
from email import message_from_string
from email.utils import parseaddr, getaddresses
import pandas as pd

try:
    import nltk
    from nltk.tokenize import word_tokenize
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


class _MLStripper(HTMLParser):
    """HTML tag stripper."""

    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
        self.convert_charrefs = True

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return " ".join(self.fed)


def _strip_html(text: str) -> str:
    """Remove HTML tags and unescape HTML entities."""
    if not text:
        return ""
    # Unescape HTML entities first
    text = html.unescape(text)
    s = _MLStripper()
    try:
        s.feed(text)
        return s.get_data()
    except Exception:
        # Fallback: simple regex
        return re.sub(r"<[^>]+>", " ", text)


class EmailPreprocessor:
    """
    Reusable email preprocessor for the AESLC pipeline.

    Methods
    -------
    clean(text) -> str
        Remove HTML, noise and normalize whitespace.
    tokenize(text) -> list[str]
        Tokenize using NLTK word_tokenize (fallback: split).
    extract_metadata(raw_email) -> dict
        Parse a raw RFC-2822 email string and return structured fields.
    preprocess_dataframe(df, text_col) -> pd.DataFrame
        Apply cleaning + tokenization to a dataframe column.
    """

    # Noise patterns to remove
    _NOISE_PATTERNS = [
        r"={2,}",                            # === dividers
        r"-{3,}",                            # --- dividers
        r"\*{2,}",                           # *** decorations
        r"_{2,}",                            # ___ decorations
        r"\[cid:[^\]]+\]",                   # Outlook inline image refs
        r"https?://\S+",                     # URLs
        r"www\.\S+",                         # bare URLs
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # emails
    ]
    _NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)

    def clean(self, text: str) -> str:
        """
        Clean email text:
        1. Strip HTML tags and unescape entities
        2. Remove common noise (URLs, inline refs, decorators)
        3. Normalize whitespace

        Parameters
        ----------
        text : str
            Raw or partially-processed email text.

        Returns
        -------
        str
            Cleaned, normalized text.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Strip HTML
        text = _strip_html(text)

        # 2. Remove noise
        text = self._NOISE_RE.sub(" ", text)

        # 3. Remove non-printable characters
        text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)

        # 4. Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text

    def tokenize(self, text: str) -> list:
        """
        Tokenize text into words.

        Uses NLTK word_tokenize if available, else splits on whitespace.

        Parameters
        ----------
        text : str
            Cleaned text to tokenize.

        Returns
        -------
        list[str]
            List of tokens (lowercase).
        """
        if not isinstance(text, str) or not text.strip():
            return []

        if _NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower())
                # Keep only alphabetic tokens and common contractions
                tokens = [t for t in tokens if re.match(r"[a-z][a-z']*", t)]
                return tokens
            except Exception:
                pass

        # Fallback: simple split + lowercase
        return [t.lower() for t in text.split() if re.match(r"[a-zA-Z]", t)]

    def extract_metadata(self, raw_email: str) -> dict:
        """
        Parse a raw RFC-2822 email string and extract structured metadata.

        Parameters
        ----------
        raw_email : str
            Full raw email including headers.

        Returns
        -------
        dict with keys:
            subject, sender, recipients, date, body, clean_body, tokens
        """
        result = {
            "subject": "",
            "sender": "",
            "recipients": [],
            "date": "",
            "body": "",
            "clean_body": "",
            "tokens": [],
        }

        if not isinstance(raw_email, str) or not raw_email.strip():
            return result

        try:
            msg = message_from_string(raw_email)

            # Subject
            result["subject"] = msg.get("Subject", "") or ""
            result["subject"] = re.sub(r"\s+", " ", result["subject"]).strip()

            # Sender
            raw_from = msg.get("From", "") or ""
            _, addr = parseaddr(raw_from)
            result["sender"] = addr or raw_from

            # Recipients (To + CC)
            to_hdr = msg.get_all("To", []) or []
            cc_hdr = msg.get_all("Cc", []) or []
            all_recipients = getaddresses(to_hdr + cc_hdr)
            result["recipients"] = [addr for _, addr in all_recipients if addr]

            # Date
            result["date"] = msg.get("Date", "") or ""

            # Body
            body_parts = []
            if msg.is_multipart():
                for part in msg.walk():
                    ct = part.get_content_type()
                    if ct == "text/plain":
                        try:
                            payload = part.get_payload(decode=True)
                            charset = part.get_content_charset() or "utf-8"
                            body_parts.append(payload.decode(charset, errors="replace"))
                        except Exception:
                            body_parts.append(str(part.get_payload()))
            else:
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        charset = msg.get_content_charset() or "utf-8"
                        body_parts.append(payload.decode(charset, errors="replace"))
                    else:
                        body_parts.append(str(msg.get_payload() or ""))
                except Exception:
                    body_parts.append(str(msg.get_payload() or ""))

            result["body"] = "\n".join(body_parts)
            result["clean_body"] = self.clean(result["body"])
            result["tokens"] = self.tokenize(result["clean_body"])

        except Exception as e:
            # Return partial result with what we managed to extract
            result["body"] = raw_email
            result["clean_body"] = self.clean(raw_email)
            result["tokens"] = self.tokenize(result["clean_body"])

        return result

    def preprocess_dataframe(
        self, df: pd.DataFrame, text_col: str
    ) -> pd.DataFrame:
        """
        Apply preprocessing to a DataFrame column.

        Adds two new columns:
        - ``<text_col>_clean``: cleaned text
        - ``<text_col>_tokens``: token list

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        text_col : str
            Name of the column containing email text.

        Returns
        -------
        pd.DataFrame
            Copy of the dataframe with added columns.
        """
        if df is None or df.empty:
            return df

        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame. "
                             f"Available columns: {df.columns.tolist()}")

        df = df.copy()
        clean_col = f"{text_col}_clean"
        token_col = f"{text_col}_tokens"

        df[clean_col] = df[text_col].fillna("").apply(self.clean)
        df[token_col] = df[clean_col].apply(self.tokenize)

        return df
