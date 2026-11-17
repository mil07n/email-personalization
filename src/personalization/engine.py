"""
Personalization Engine Module.
Manages user preferences (important senders, priority topics, threshold filters),
adjusts email priority scoring, and generates executive personalized daily briefings.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from config.config import USER_PREFS_PATH


DEFAULT_PREFERENCES: Dict[str, Any] = {
    "user_name": "Executive User",
    "user_role": "Director / Executive",
    "important_senders": [
        "ceo@company.com",
        "board@company.com",
        "legal@company.com",
        "vp@company.com",
        "cto@company.com",
        "Alice Johnson",
        "John Smith"
    ],
    "priority_topics": [
        "contract",
        "budget",
        "outage",
        "security",
        "compliance",
        "deadline",
        "pipeline",
        "quarterly"
    ],
    "min_urgency_to_highlight": "MEDIUM",
    "summary_format": "executive_briefing",
    "max_summary_items": 10
}


class PersonalizationEngine:
    """
    Handles user preference persistence, dynamic email re-ranking,
    and personalized daily digest compilation.
    """

    def __init__(self, prefs_path: Optional[Path] = None):
        self.prefs_path = Path(prefs_path or USER_PREFS_PATH)
        self.preferences = self.load_preferences()

    def load_preferences(self) -> Dict[str, Any]:
        """Load preferences from JSON file or return defaults."""
        if self.prefs_path.exists():
            try:
                with open(self.prefs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Merge with defaults
                    merged = DEFAULT_PREFERENCES.copy()
                    merged.update(data)
                    return merged
            except Exception:
                pass
        # If not existing or error, save and return default
        self.save_preferences(DEFAULT_PREFERENCES)
        return DEFAULT_PREFERENCES.copy()

    def save_preferences(self, prefs: Dict[str, Any]) -> None:
        """Persist preferences to disk."""
        self.prefs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.prefs_path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
        self.preferences = prefs

    def update_preferences(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update selected fields in preferences."""
        current = self.load_preferences()
        current.update(updates)
        self.save_preferences(current)
        return current

    def get_preferences(self) -> Dict[str, Any]:
        """Return current user preferences."""
        return self.preferences

    def personalize_email(self, analyzed_email: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust priority and compute personalized importance score for an email.
        Score formula:
            Base Score (from urgency: 0.2 - 0.95)
            + 0.30 if sender in important_senders
            + 0.20 if topic in priority_topics
            + 0.15 if actionable tasks exist
            + 0.10 if upcoming deadline exists
        """
        email = analyzed_email.copy()
        prefs = self.preferences

        sender = str(email.get("sender", "")).lower()
        subject = str(email.get("subject", "")).lower()
        body = str(email.get("body", "")).lower()
        text = f"{subject} {body}"

        base_score = float(email.get("urgency", {}).get("score", 0.50))
        bonus = 0.0
        reasons = []

        # 1. VIP sender bonus
        for imp in prefs.get("important_senders", []):
            if imp and imp.lower() in sender:
                bonus += 0.30
                reasons.append(f"Important Sender: '{imp}'")
                break

        # 2. Priority topic bonus
        matched_topics = []
        for topic in prefs.get("priority_topics", []):
            if topic and topic.lower() in text:
                matched_topics.append(topic)
        if matched_topics:
            bonus += min(0.25, 0.10 * len(matched_topics))
            reasons.append(f"Priority Topics: {', '.join(matched_topics[:3])}")

        # 3. Action items bonus
        action_items = email.get("action_items", [])
        if action_items:
            bonus += 0.15
            reasons.append(f"Contains {len(action_items)} action items")

        # 4. Deadlines bonus
        deadlines = email.get("deadlines", [])
        if deadlines:
            bonus += 0.10
            reasons.append("Contains explicit deadline")

        personalized_score = min(1.0, base_score + bonus)
        is_highlighted = (
            personalized_score >= 0.70 or
            email.get("urgency", {}).get("priority") in ("HIGH", "CRITICAL") or
            bool(reasons)
        )

        email["personalized_score"] = round(personalized_score, 3)
        email["is_highlighted"] = is_highlighted
        email["personalization_reasons"] = reasons
        return email

    def compile_daily_summary(
        self,
        emails: List[Dict[str, Any]],
        conversations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a personalized daily executive briefing from a batch of analyzed emails.
        """
        prefs = self.preferences
        date_str = datetime.now().strftime("%A, %B %d, %Y")

        # Personalize all emails and sort by personalized_score descending
        ranked_emails = [self.personalize_email(e) for e in emails]
        ranked_emails.sort(key=lambda x: x.get("personalized_score", 0.0), reverse=True)

        # Count priorities
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        all_actions = []
        all_deadlines = []

        for e in ranked_emails:
            p = e.get("urgency", {}).get("priority", "MEDIUM")
            counts[p] = counts.get(p, 0) + 1
            for act in e.get("action_items", []):
                all_actions.append({
                    "action": act.get("action"),
                    "responsible_person": act.get("responsible_person"),
                    "deadline": act.get("deadline"),
                    "email_subject": e.get("subject"),
                    "sender": e.get("sender")
                })
            for dl in e.get("deadlines", []):
                task_name = dl.get("task")
                if not task_name:
                    for act in e.get("action_items", []):
                        if dl.get("raw") and dl.get("raw").lower() in (act.get("deadline", "") + " " + act.get("source_sentence", "")).lower():
                            task_name = act.get("action")
                            break
                    if not task_name and e.get("action_items"):
                        task_name = e.get("action_items")[0].get("action")

                all_deadlines.append({
                    "deadline": dl.get("raw"),
                    "normalized": dl.get("normalized"),
                    "email_subject": e.get("subject"),
                    "task": task_name
                })

        # Top urgent / high priority emails
        top_urgent = [
            e for e in ranked_emails
            if e.get("urgency", {}).get("priority") in ("CRITICAL", "HIGH")
        ][:prefs.get("max_summary_items", 8)]

        # Highlighted key decisions / thread summaries
        conv_summaries = []
        if conversations:
            for c in conversations[:5]:
                conv_summaries.append({
                    "subject": c.get("subject"),
                    "participants": c.get("participants", []),
                    "num_emails": c.get("num_emails", 1),
                    "summary": c.get("summary", "")
                })

        # Structured briefing markdown text
        md_lines = [
            f"# 📋 Executive Daily Email Briefing",
            f"**Prepared for:** {prefs.get('user_name', 'Executive')} ({prefs.get('user_role', 'Leadership')})  ",
            f"**Date:** {date_str} | **Total Emails Analyzed:** {len(ranked_emails):,}\n",
            "## 🚨 Priority Overview",
            f"- 🔴 **Critical:** {counts['CRITICAL']} requiring immediate attention",
            f"- 🟠 **High Priority:** {counts['HIGH']} due today/tomorrow",
            f"- 🟡 **Medium Priority:** {counts['MEDIUM']}",
            f"- 🟢 **Low / Informational:** {counts['LOW']}\n",
        ]

        if top_urgent:
            md_lines.append("## ⚡ Critical & Urgent Action Required")
            for idx, item in enumerate(top_urgent, 1):
                p_badge = "🔴" if item.get("urgency", {}).get("priority") == "CRITICAL" else "🟠"
                md_lines.append(f"{idx}. {p_badge} **{item.get('subject')}** (From: *{item.get('sender')}*)")
                if item.get("action_items"):
                    act_desc = item["action_items"][0].get("action")
                    person = item["action_items"][0].get("responsible_person")
                    md_lines.append(f"   - **Action:** {act_desc} (Assigned to: {person})")
                if item.get("deadlines"):
                    md_lines.append(f"   - **Deadline:** {item['deadlines'][0].get('raw')}")
            md_lines.append("")

        if all_actions:
            md_lines.append("## ✅ Pending Action Items")
            for act in all_actions[:6]:
                md_lines.append(f"- [ ] **{act['action']}** — *{act['responsible_person']}* (Ref: {act['email_subject'][:40]}…)")
            md_lines.append("")

        if all_deadlines:
            md_lines.append("## ⏰ Upcoming Deadlines")
            for dl in all_deadlines[:5]:
                task_str = f" → **Task to complete:** {dl['task']}" if dl.get('task') else ""
                md_lines.append(f"- ⏳ **{dl['deadline']}**{task_str} (Normalized: `{dl['normalized']}`) — *{dl['email_subject'][:40]}…*")
            md_lines.append("")

        briefing_text = "\n".join(md_lines)

        return {
            "date": date_str,
            "total_emails": len(ranked_emails),
            "counts": counts,
            "top_urgent_emails": top_urgent,
            "pending_action_items": all_actions,
            "upcoming_deadlines": all_deadlines,
            "top_upcoming_deadlines": all_deadlines,
            "conversation_summaries": conv_summaries,
            "briefing_markdown": briefing_text
        }
