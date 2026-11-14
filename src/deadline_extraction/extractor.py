"""
Deadline Extraction Module.
Uses NLP pattern matching, temporal expressions, and date normalization
to extract deadlines and temporal constraints from email text.
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


class DeadlineExtractor:
    """
    Extracts explicit dates, times, relative temporal expressions,
    and normalizes them against an optional reference date.
    """

    # Temporal patterns
    TEMPORAL_PATTERNS = [
        # Relative day / part of day
        (r"\bby\s+(?:end\s+of\s+day|eod|end\s+of\s+business|eob|close\s+of\s+business|cob)\b", "End of Day (5:00 PM)", "relative"),
        (r"\bby\s+(?:noon|midday|midnight)\b", "By Noon/Midnight", "relative"),
        (r"\b(?:due\s+)?(?:by\s+)?today(?:\s+(?:morning|afternoon|evening|eod))?\b", "Today", "relative"),
        (r"\b(?:due\s+)?(?:by\s+)?tomorrow(?:\s+(?:morning|afternoon|evening|eod|night))?\b", "Tomorrow", "relative"),
        (r"\b(?:by\s+|due\s+)?(?:next|this|coming)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|eod))?\b", "Weekday Relative", "relative"),
        (r"\b(?:before|by|due\s+by)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "By Weekday", "relative"),
        (r"\bwithin\s+(\d+)\s+(hours?|hrs?|business\s+days?|days?|weeks?)\b", "Duration Window", "relative"),
        (r"\bin\s+(\d+)\s+(days?|weeks?|months?)\b", "Future Offset", "relative"),
        (r"\basap|immediately|at\s+your\s+earliest\s+convenience\b", "Immediate / ASAP", "relative"),

        # Explicit standard dates (e.g. 2026-11-20, 11/20/2026)
        (r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", "ISO/Standard Date", "absolute"),
        (r"\b(\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?)\b", "Slash/Hyphen Date", "absolute"),

        # Written dates (e.g. November 20th, 2026 or 20 Nov 2026)
        (r"\b(?:by\s+|due\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?\b", "Month Day Year", "absolute"),
        (r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s*,?\s*(\d{4}))?\b", "Day Month Year", "absolute"),

        # Explicit times
        (r"\b(?:at|by|before)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|est|cst|pst|gmt|utc))\b", "Specific Time", "time"),
    ]

    WEEKDAYS = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }

    MONTHS = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12
    }

    def __init__(self):
        pass

    def extract_deadlines(
        self,
        text: str,
        reference_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract all deadline mentions from text and normalize where possible.
        """
        if not text:
            return []

        ref = reference_date or datetime.now()
        deadlines = []
        seen_spans = set()

        text_lower = text.lower()

        for pattern, type_name, cat in self.TEMPORAL_PATTERNS:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                raw = text[match.start():match.end()].strip()
                span = (match.start(), match.end())

                # Avoid overlapping spans
                if any(s[0] <= span[0] and span[1] <= s[1] for s in seen_spans):
                    continue
                seen_spans.add(span)

                normalized = self._normalize(raw, type_name, match, ref)

                deadlines.append({
                    "raw": raw,
                    "type": cat,
                    "normalized": normalized,
                    "start": match.start(),
                    "end": match.end()
                })

        return deadlines

    def _normalize(
        self,
        raw: str,
        type_name: str,
        match: re.Match,
        ref: datetime
    ) -> Optional[str]:
        """Normalize temporal mention to standard ISO format (YYYY-MM-DD or descriptive)."""
        raw_clean = raw.lower().strip()

        if "asap" in raw_clean or "immediately" in raw_clean:
            return ref.strftime("%Y-%m-%d (Immediate)")

        if "today" in raw_clean or "eod" in raw_clean or "close of business" in raw_clean:
            return ref.strftime("%Y-%m-%d 17:00")

        if "tomorrow" in raw_clean:
            target = ref + timedelta(days=1)
            return target.strftime("%Y-%m-%d 17:00")

        # Check weekday relative
        for w_name, w_idx in self.WEEKDAYS.items():
            if w_name in raw_clean:
                current_w = ref.weekday()
                days_ahead = (w_idx - current_w) % 7
                if days_ahead == 0 and ("next" in raw_clean or "by" in raw_clean):
                    days_ahead = 7
                target = ref + timedelta(days=days_ahead)
                return target.strftime(f"%Y-%m-%d ({w_name.capitalize()})")

        # Check within N days/hours
        m_within = re.search(r"within\s+(\d+)\s+(hours?|hrs?|days?|weeks?)", raw_clean)
        if m_within:
            num = int(m_within.group(1))
            unit = m_within.group(2)
            if "hour" in unit or "hr" in unit:
                target = ref + timedelta(hours=num)
                return target.strftime("%Y-%m-%d %H:%M")
            elif "day" in unit:
                target = ref + timedelta(days=num)
                return target.strftime("%Y-%m-%d")
            elif "week" in unit:
                target = ref + timedelta(weeks=num)
                return target.strftime("%Y-%m-%d")

        # Check written month/day (e.g. Nov 20)
        for m_name, m_idx in self.MONTHS.items():
            if m_name in raw_clean:
                m_day = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", raw_clean)
                if m_day:
                    day = int(m_day.group(1))
                    year = ref.year
                    m_year = re.search(r"\b(202\d)\b", raw_clean)
                    if m_year:
                        year = int(m_year.group(1))
                    try:
                        target = datetime(year, m_idx, day)
                        return target.strftime("%Y-%m-%d")
                    except ValueError:
                        pass

        return raw
