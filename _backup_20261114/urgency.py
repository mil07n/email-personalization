"""
urgency.py – Email Intelligence Module

Implements the core project goal:
  "An assistant that analyzes incoming emails to classify urgency,
   identify action items and deadlines, group related conversations,
   and generate personalized daily summaries."

NLP techniques used (no external model required):
  • Bag-of-words keyword matching with 4-tier weighted word banks
  • Regex-based pattern scoring on subject lines
  • Time-expression detection with regex
  • Sentence segmentation + action-verb pattern matching for action items
  • Jaccard similarity for conversation clustering
  • Priority-sorted digest generation
"""

import re
from dataclasses import dataclass, field
from typing import List


# ── Urgency signal word banks ──────────────────────────────────────────────────

CRITICAL_SIGNALS = {
    # system / service failure
    "outage", "down", "downtime", "offline", "unavailable", "crashed", "crash",
    "failure", "failed", "broken", "not working", "system error",
    # security
    "breach", "hack", "hacked", "compromised", "data leak", "unauthorized access",
    # legal / financial
    "lawsuit", "legal", "subpoena", "court", "injunction", "penalty", "fine",
    "payment failed", "overdue payment", "unpaid invoice", "bounced", "fraud",
    # extreme urgency words
    "immediately", "right now", "asap", "emergency", "critical", "urgent",
    "life-threatening", "last chance", "final warning", "final notice",
}

HIGH_SIGNALS = {
    # deadlines
    "deadline", "due today", "due tomorrow", "eod", "end of day", "end of week",
    "by friday", "by monday", "this week", "within 24 hours", "within 48 hours",
    # escalation
    "escalate", "escalated", "escalation", "unresolved", "not resolved",
    "still waiting", "no response", "ignored", "disappointed", "frustrated",
    # executive mention
    "ceo", "cfo", "cto", "president", "board", "director", "executive", "vp",
    # contract / compliance
    "contract", "agreement", "sla", "breach of contract", "compliance",
    "regulatory", "audit", "gdpr", "hipaa", "pci", "iso", "violation",
    # financial impact
    "revenue loss", "customer churn", "refund", "cancellation",
}

MEDIUM_SIGNALS = {
    "follow up", "follow-up", "reminder", "check in", "check-in",
    "meeting", "schedule", "agenda", "deliverable", "milestone", "review",
    "feedback", "approval", "sign off", "signoff", "confirm", "confirmation",
    "pending", "waiting", "update", "status", "progress", "action item",
    "response needed", "please reply", "let me know",
}

LOW_SIGNALS = {
    "newsletter", "fyi", "for your information",
    "just letting you know", "heads up", "no action required",
    "optional", "whenever you get a chance", "no rush", "low priority",
    "feel free", "at your leisure",
}

URGENT_SUBJECT_PATTERNS = [
    r"\bURGENT\b", r"\bASAP\b", r"\bACTION REQUIRED\b",
    r"\bIMPORTANT\b", r"\bPLEASE RESPOND\b", r"\bRESPONSE NEEDED\b",
    r"\bATTENTION\b", r"!{2,}", r"\bDEADLINE\b", r"\bFINAL\b",
    r"\bALERT\b", r"\bWARNING\b", r"\bCRITICAL\b", r"\bEMERGENCY\b",
]

LEVEL_BADGES = {
    "CRITICAL": "🔴 CRITICAL",
    "HIGH":     "🟠 HIGH",
    "MEDIUM":   "🟡 MEDIUM",
    "LOW":      "🟢 LOW",
}

_LEVEL_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


# ── Action item extraction ─────────────────────────────────────────────────────

_ACTION_PATTERNS = [
    r'please\s+\w.{5,80}',
    r'need\s+to\s+\w.{5,80}',
    r'must\s+\w.{5,80}',
    r'should\s+\w.{5,80}',
    r'action\s+required.{0,80}',
    r'can\s+you\s+\w.{5,80}',
    r'could\s+you\s+\w.{5,80}',
    r'you\s+need\s+to.{5,80}',
    r'kindly\s+\w.{5,80}',
    r'ensure\s+\w.{5,80}',
    r'(?:follow\s+up|followup).{0,60}',
    r'(?:provide|submit|approve|review|confirm|respond|reply|send|schedule|complete|update|fix|resolve|check)\s+\w.{3,70}',
    r'waiting\s+for\s+\w.{5,70}',
    r'required\s+from\s+\w.{5,70}',
]


def extract_action_items(text: str) -> List[str]:
    """
    Extract action item sentences from email text.

    Uses sentence segmentation + action-verb regex pattern matching.
    Returns up to 7 action item strings.
    """
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    found = []
    seen_keys = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 12:
            continue
        s_lower = sentence.lower()
        for pat in _ACTION_PATTERNS:
            if re.search(pat, s_lower):
                cleaned = sentence[:120] + ("…" if len(sentence) > 120 else "")
                dedup_key = s_lower[:40]
                if dedup_key not in seen_keys:
                    found.append(cleaned)
                    seen_keys.add(dedup_key)
                break

    return found[:7]


# ── Deadline extraction ────────────────────────────────────────────────────────

_DEADLINE_PATTERNS = [
    r'by\s+(?:end of day|eod|end of business|eob|noon|midnight|morning|evening)',
    r'by\s+(?:today|tomorrow|this\s+(?:week|month|friday|monday|tuesday|wednesday|thursday|saturday|sunday))',
    r'by\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+\d{4})?',
    r'by\s+\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?',
    r'by\s+\d{1,2}\s*(?:am|pm)',
    r'within\s+\d+\s+(?:hours?|minutes?|days?|weeks?)',
    r'due\s+(?:today|tomorrow|by\s+\w+|on\s+\w+|this\s+\w+)',
    r'deadline[:\s]+.{3,40}',
    r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?\b',
    r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
    r'\d{4}-\d{2}-\d{2}',
    r'(?:this|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
    r'end of\s+(?:day|week|month|quarter)',
    r'(?:today|tomorrow)\s+(?:morning|afternoon|evening|at\s+\d)',
    r'before\s+\d{1,2}\s*(?:am|pm)',
    r'before\s+(?:end of day|noon|midnight)',
    r'\d{1,2}:\d{2}\s*(?:am|pm)',
]


def extract_deadlines(text: str) -> List[str]:
    """
    Extract deadline, date, and time references from email text.

    Returns a deduplicated list of up to 6 deadline strings.
    """
    text_lower = text.lower()
    found = []
    seen = set()
    for pat in _DEADLINE_PATTERNS:
        for m in re.finditer(pat, text_lower):
            phrase = m.group().strip()
            if phrase and phrase not in seen:
                found.append(phrase)
                seen.add(phrase)
    return found[:6]


# ── Main urgency classifier ────────────────────────────────────────────────────

@dataclass
class UrgencyResult:
    level: str              # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    score: int              # raw weighted score
    confidence: str         # "high" | "medium" | "low"
    signals_found: List[str]
    action_items: List[str]
    deadlines: List[str]
    badge: str              # e.g. "🔴 CRITICAL"
    explanation: str        # human-readable reasoning


def classify_urgency(
    subject: str,
    body: str,
    sender_name: str = "Team",
) -> UrgencyResult:
    """
    Classify the urgency level of an incoming email.

    NLP techniques: keyword bag-of-words, regex pattern scoring,
    weighted feature aggregation, time-expression detection.

    Parameters
    ----------
    subject     : Email subject line
    body        : Email body text
    sender_name : Sender's name (used for context)

    Returns
    -------
    UrgencyResult dataclass
    """
    combined = (subject + " " + body).lower()
    subject_upper = subject.upper()

    score = 0
    signals_found = []

    # Subject-line pattern scoring (heavy weight — sender chose subject deliberately)
    for pat in URGENT_SUBJECT_PATTERNS:
        if re.search(pat, subject_upper):
            score += 20
            signals_found.append(f"subject-pattern: {pat!r}")

    # Keyword tier scoring
    for word in CRITICAL_SIGNALS:
        if word in combined:
            score += 15
            signals_found.append(f"critical: '{word}'")

    for word in HIGH_SIGNALS:
        if word in combined:
            score += 8
            signals_found.append(f"high: '{word}'")

    for word in MEDIUM_SIGNALS:
        if word in combined:
            score += 3
            signals_found.append(f"medium: '{word}'")

    for word in LOW_SIGNALS:
        if word in combined:
            score -= 5
            signals_found.append(f"low-signal: '{word}'")

    # Time proximity boost
    if re.search(r'\btoday\b|\bthis morning\b|\bthis afternoon\b', combined):
        score += 10
        signals_found.append("time-proximity: today / this morning")
    if re.search(r'\btomorrow\b|\bby \w+ morning\b', combined):
        score += 6
        signals_found.append("time-proximity: tomorrow")

    # Question density (many questions = expects rapid response)
    q_count = body.count("?")
    if q_count >= 3:
        score += 5
        signals_found.append(f"question-density: {q_count} questions")

    # Determine level
    if score >= 30:
        level, confidence = "CRITICAL", "high" if score >= 50 else "medium"
    elif score >= 15:
        level, confidence = "HIGH", "high" if score >= 25 else "medium"
    elif score >= 6:
        level, confidence = "MEDIUM", "medium"
    else:
        level, confidence = "LOW", "medium" if score >= 0 else "low"

    full_text = f"{subject} {body}"
    action_items = extract_action_items(full_text)
    deadlines = extract_deadlines(full_text)

    unique_signals = list(dict.fromkeys(signals_found))
    top3 = unique_signals[:3]
    explanation = (
        f"Urgency score: **{score}** | Confidence: **{confidence.upper()}**\n\n"
        + (f"Key signals found: {', '.join(top3)}{'…' if len(unique_signals) > 3 else ''}."
           if top3 else "No strong urgency signals detected.")
    )

    return UrgencyResult(
        level=level,
        score=score,
        confidence=confidence,
        signals_found=unique_signals,
        action_items=action_items,
        deadlines=deadlines,
        badge=LEVEL_BADGES[level],
        explanation=explanation,
    )


# ── Conversation grouping ──────────────────────────────────────────────────────

def _get_keywords(text: str) -> set:
    """Extract meaningful lowercase tokens (uses project's clean_text if available)."""
    try:
        from preprocess import clean_text
        return set(clean_text(text).split())
    except Exception:
        _BASIC_STOPS = {
            "the", "and", "for", "are", "this", "that", "with", "from",
            "you", "your", "have", "has", "was", "will", "been", "just",
            "not", "can", "our", "all", "but", "they", "their", "more",
            "its", "been", "also", "into", "some", "than", "then", "she",
            "him", "his", "her", "who", "had", "which", "when", "what",
        }
        words = re.findall(r'[a-z]{3,}', text.lower())
        return {w for w in words if w not in _BASIC_STOPS}


def group_conversations(emails: List[dict]) -> List[dict]:
    """
    Group related emails into conversation clusters using Jaccard similarity.

    Parameters
    ----------
    emails : list of dicts with keys "sender", "subject", "body"

    Returns
    -------
    list of dicts: { "group_id", "theme", "count", "emails" }
    """
    if not emails:
        return []
    if len(emails) == 1:
        return [{
            "group_id": 0,
            "theme": emails[0].get("subject", "Email 1")[:60],
            "count": 1,
            "emails": emails,
        }]

    kw = [_get_keywords(f"{e.get('subject','')} {e.get('body','')}") for e in emails]
    n = len(emails)
    assigned = [-1] * n
    groups = []
    gid = 0

    for i in range(n):
        if assigned[i] != -1:
            continue
        assigned[i] = gid
        members = [i]

        for j in range(i + 1, n):
            if assigned[j] != -1:
                continue
            if not kw[i] or not kw[j]:
                continue
            inter = len(kw[i] & kw[j])
            union = len(kw[i] | kw[j])
            # Jaccard threshold = 8%: low enough to catch related topics
            if union > 0 and inter / union >= 0.08:
                assigned[j] = gid
                members.append(j)

        theme = emails[i].get("subject", f"Group {gid + 1}")[:60]
        groups.append({
            "group_id": gid,
            "theme": theme,
            "count": len(members),
            "emails": [emails[m] for m in members],
        })
        gid += 1

    return groups


# ── Daily summary generator ────────────────────────────────────────────────────

def generate_daily_summary(emails: List[dict]) -> str:
    """
    Generate a personalized priority-sorted daily email digest.

    Classifies each email, sorts by urgency, extracts key action items
    and deadlines, and formats a readable daily briefing.

    Parameters
    ----------
    emails : list of dicts with keys "sender", "subject", "body"

    Returns
    -------
    Formatted summary string.
    """
    if not emails:
        return "No emails provided. Add emails above and click Generate."

    # Classify and sort
    results = []
    for email in emails:
        r = classify_urgency(
            email.get("subject", ""),
            email.get("body", ""),
            email.get("sender", "Unknown"),
        )
        results.append((email, r))
    results.sort(key=lambda x: _LEVEL_ORDER[x[1].level])

    # Count per level
    counts = {k: 0 for k in _LEVEL_ORDER}
    for _, r in results:
        counts[r.level] += 1

    lines = [
        "╔══════════════════════════════════════════╗",
        "║      📋  PERSONALIZED DAILY DIGEST       ║",
        "╚══════════════════════════════════════════╝",
        "",
        f"  Total emails: {len(emails)}",
    ]
    for lvl in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if counts[lvl] > 0:
            lines.append(f"  {LEVEL_BADGES[lvl]}  × {counts[lvl]}")
    lines += ["", "──────────────────────────────────────────", ""]

    for i, (email, r) in enumerate(results, 1):
        sender  = email.get("sender", "Unknown")
        subject = email.get("subject", "(no subject)")[:55]
        lines.append(f"  {i:>2}.  {r.badge}")
        lines.append(f"       From    : {sender}")
        lines.append(f"       Subject : {subject}")
        if r.action_items:
            lines.append(f"       ✅ Action  : {r.action_items[0][:80]}")
        if r.deadlines:
            lines.append(f"       ⏰ Deadline: {r.deadlines[0]}")
        lines.append("")

    # Closing advice
    lines.append("──────────────────────────────────────────")
    if counts["CRITICAL"] > 0:
        lines.append("⚠️  CRITICAL emails require your IMMEDIATE attention!")
    elif counts["HIGH"] > 0:
        lines.append("👉  Handle HIGH-priority emails before end of day.")
    else:
        lines.append("✅  No critical or high-priority emails today.")

    return "\n".join(lines)
