"""
Deadline Extractor Module
Extracts and normalizes date/time deadlines from email text using regex patterns.
Supports absolute dates, relative dates, and time expressions.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for deadline extraction
# ---------------------------------------------------------------------------

# Month name helper
_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_WEEKDAYS = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"

# ---------------------------------------------------------------------------
# Pattern groups: (pattern, type)
# ---------------------------------------------------------------------------
_PATTERNS = [
    # Absolute: "January 15, 2026" / "15 January 2026"
    (
        rf"(?:{_MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?)"
        rf"|(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}(?:,?\s+\d{{4}})?)",
        "absolute",
    ),
    # Absolute ISO: 2026-11-14 or 2026/11/14
    (r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", "absolute"),
    # Absolute DD/MM/YYYY or MM/DD/YYYY
    (r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", "absolute"),
    # Relative: "tomorrow", "today", "next week" etc.
    (
        r"\b(?:today|tomorrow|yesterday|next\s+(?:week|month|year|"
        + _WEEKDAYS
        + r"))\b",
        "relative",
    ),
    # Relative: "this Friday", "this week"
    (rf"\bthis\s+(?:{_WEEKDAYS}|week|month)\b", "relative"),
    # Relative: "in N days/weeks/months"
    (r"\bin\s+\d+\s+(?:days?|weeks?|months?|hours?)\b", "relative"),
    # Relative: "N days/weeks from now"
    (r"\b\d+\s+(?:days?|weeks?|months?)\s+(?:from\s+now|from\s+today|hence)\b", "relative"),
    # Time: "by 5 PM", "before noon", "by EOD", "by COB"
    (
        r"\b(?:by\s+)?(?:EOD|COB|EOM|noon|midnight|end\s+of\s+(?:day|week|month|business\s+day))\b",
        "relative",
    ),
    (r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b", "absolute"),
    (r"\b(?:by|before|until|no\s+later\s+than)\s+\d{1,2}\s*(?:AM|PM|am|pm)\b", "absolute"),
    # "by [deadline phrase]" — catch-all for preceded by preposition
    (
        rf"(?:by|before|until|due|deadline(?:\s+is)?(?:\s+on)?)\s+"
        rf"(?:{_MONTHS}\s+\d{{1,2}}|\d{{1,2}}(?:st|nd|rd|th)?|next\s+{_WEEKDAYS}"
        rf"|tomorrow|today|EOD|COB|\d{{4}}[-/]\d{{1,2}}[-/]\d{{1,2}})",
        "absolute",
    ),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), t) for p, t in _PATTERNS]

# ---------------------------------------------------------------------------
# Relative date normalization helpers
# ---------------------------------------------------------------------------
_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}


def _normalize_relative(raw: str, ref: datetime) -> Optional[str]:
    """
    Normalize a relative date expression to an ISO date string.

    Parameters
    ----------
    raw : str
        The raw relative date expression (e.g., "tomorrow", "next Friday").
    ref : datetime
        Reference date for relative calculations.

    Returns
    -------
    str | None
        ISO date string (YYYY-MM-DD) or None if unrecognizable.
    """
    text = raw.lower().strip()

    if "today" in text or "eod" in text or "cob" in text:
        return ref.strftime("%Y-%m-%d")
    if "tomorrow" in text:
        return (ref + timedelta(days=1)).strftime("%Y-%m-%d")
    if "yesterday" in text:
        return (ref - timedelta(days=1)).strftime("%Y-%m-%d")
    if "next week" in text:
        return (ref + timedelta(weeks=1)).strftime("%Y-%m-%d")
    if "next month" in text or "eom" in text or "end of month" in text:
        # First day of next month
        if ref.month == 12:
            return datetime(ref.year + 1, 1, 1).strftime("%Y-%m-%d")
        return datetime(ref.year, ref.month + 1, 1).strftime("%Y-%m-%d")
    if "this week" in text:
        # Next Friday of this week
        days_until_friday = (4 - ref.weekday()) % 7
        return (ref + timedelta(days=days_until_friday)).strftime("%Y-%m-%d")

    # "next <weekday>"
    for wd_name, wd_num in _WEEKDAY_MAP.items():
        if f"next {wd_name}" in text or f"this {wd_name}" in text:
            days_ahead = (wd_num - ref.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (ref + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # "in N days/weeks"
    m = re.search(r"in\s+(\d+)\s+(day|week|month|hour)", text)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "day":
            return (ref + timedelta(days=n)).strftime("%Y-%m-%d")
        if unit == "week":
            return (ref + timedelta(weeks=n)).strftime("%Y-%m-%d")
        if unit == "month":
            new_month = ref.month + n
            new_year = ref.year + (new_month - 1) // 12
            new_month = ((new_month - 1) % 12) + 1
            return datetime(new_year, new_month, ref.day).strftime("%Y-%m-%d")
        if unit == "hour":
            return (ref + timedelta(hours=n)).strftime("%Y-%m-%dT%H:%M")

    # "N days from now"
    m = re.search(r"(\d+)\s+(day|week|month)s?\s+(?:from\s+(?:now|today)|hence)", text)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "day":
            return (ref + timedelta(days=n)).strftime("%Y-%m-%d")
        if unit == "week":
            return (ref + timedelta(weeks=n)).strftime("%Y-%m-%d")

    return None


def _normalize_absolute(raw: str) -> Optional[str]:
    """
    Try to parse an absolute date string to ISO format.

    Parameters
    ----------
    raw : str
        Raw absolute date string.

    Returns
    -------
    str | None
        ISO date string or None if parsing fails.
    """
    text = raw.strip()
    formats = [
        "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y",
        "%d %B %Y", "%d %b %Y", "%d %B, %Y", "%d %b, %Y",
        "%Y-%m-%d", "%Y/%m/%d",
        "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
        "%d/%m/%y", "%m/%d/%y",
        "%B %d", "%b %d",  # no year — will use current year
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.year == 1900:
                dt = dt.replace(year=datetime.now().year)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Try time-only formats
    for fmt in ["%I:%M %p", "%H:%M", "%I %p"]:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%H:%M")
        except ValueError:
            continue

    return None


# ---------------------------------------------------------------------------
# Deadline Extractor class
# ---------------------------------------------------------------------------
class DeadlineExtractor:
    """
    Extract and normalize deadline expressions from email text.

    Returns a list of dicts with keys:
        raw        : str            — original matched text
        normalized : str | None     — ISO date string (YYYY-MM-DD) or None
        type       : str            — "absolute" | "relative"
    """

    def extract(self, text: str, reference_date: Optional[str] = None) -> list:
        """
        Extract deadline expressions from email text.

        Parameters
        ----------
        text : str
            Email body or subject text.
        reference_date : str | None
            ISO date string (YYYY-MM-DD) used as reference for relative dates.
            Defaults to today's date if not provided.

        Returns
        -------
        list[dict]
            [{raw, normalized, type}]
        """
        if not isinstance(text, str) or not text.strip():
            return []

        # Parse reference date
        if reference_date:
            try:
                ref = datetime.strptime(reference_date, "%Y-%m-%d")
            except ValueError:
                ref = datetime.now()
        else:
            ref = datetime.now()

        results = []
        seen_spans = []  # list of (start, end) to avoid overlapping matches

        for pattern, dtype in _COMPILED:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()

                # Skip if overlapping with an already-found match
                overlapping = any(
                    not (end <= s or start >= e) for s, e in seen_spans
                )
                if overlapping:
                    continue

                raw = match.group(0).strip()
                if not raw or len(raw) < 2:
                    continue

                # Normalize
                if dtype == "relative":
                    normalized = _normalize_relative(raw, ref)
                else:
                    normalized = _normalize_absolute(raw)

                seen_spans.append((start, end))
                results.append({
                    "raw": raw,
                    "normalized": normalized,
                    "type": dtype,
                })

        # Sort by position in text
        results.sort(key=lambda x: text.find(x["raw"]))
        return results
