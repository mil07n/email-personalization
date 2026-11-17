"""
Email Category & Spam Classifier
Classifies emails into: SPAM, URGENT, ACTION_REQUIRED, MEETING, FINANCE,
NEWSLETTER, SOCIAL, GENERAL using rule-based signals + keyword scoring.
"""

import re
from typing import Dict, Any

# ── Keyword signal banks ───────────────────────────────────────────────────────

SPAM_SIGNALS = [
    r"\bunsubscribe\b", r"\bclick here\b", r"\byou have won\b", r"\bcongratulations\b",
    r"\bfree offer\b", r"\blimited time offer\b", r"\bact now\b", r"\bexclusive deal\b",
    r"\bverify your account\b", r"\bconfirm your identity\b", r"\bbank details\b",
    r"\blottery\b", r"\bprize\b", r"\bgift card\b", r"\bmake money\b", r"\bearn cash\b",
    r"\bno cost\b", r"\bdouble your\b", r"\bguaranteed\b", r"\bviagra\b",
    r"\bcasino\b", r"\bbet now\b", r"\bspecial promotion\b", r"\bdiscount\b",
    r"\bsale ends\b", r"\blast chance\b", r"\bget rich\b", r"\bwork from home\b",
]

URGENT_SIGNALS = [
    r"\bURGENT\b", r"\bASAP\b", r"\bimmediately\b", r"\bcritical\b",
    r"\bemergency\b", r"\boverdue\b", r"\bblocking\b", r"\bdeal breaker\b",
    r"\bescalat", r"\bceo\b", r"\bdirector\b", r"\bboard\b",
    r"\bdeadline today\b", r"\bdue today\b", r"\bby end of day\b", r"\bEOD\b",
]

ACTION_SIGNALS = [
    r"\bplease review\b", r"\baction required\b", r"\baction needed\b",
    r"\bplease approve\b", r"\bneeds your sign\b", r"\bplease respond\b",
    r"\bfollow up\b", r"\btask assigned\b", r"\bplease complete\b",
    r"\bdeadline\b", r"\bdue by\b", r"\bsubmit by\b", r"\breview and\b",
    r"\bapprove by\b", r"\brespond by\b",
]

MEETING_SIGNALS = [
    r"\bmeeting\b", r"\bcall scheduled\b", r"\bconference\b", r"\bwebinar\b",
    r"\binvitation\b", r"\bcalendar invite\b", r"\bagenda\b", r"\bstand.?up\b",
    r"\bscrum\b", r"\bsync\b", r"\binterview\b", r"\bappointment\b",
    r"\bjoin the call\b", r"\bzoom\b", r"\bteams meeting\b", r"\bgoogle meet\b",
]

FINANCE_SIGNALS = [
    r"\binvoice\b", r"\bpayment\b", r"\bbill\b", r"\bpayroll\b",
    r"\bbudget\b", r"\bquote\b", r"\bpurchase order\b", r"\bpo#\b",
    r"\breceipt\b", r"\btransaction\b", r"\bbank\b", r"\bwire transfer\b",
    r"\baccounting\b", r"\bexpense\b", r"\bfinancial\b", r"\bbalance\b",
    r"\brefund\b", r"\bsubscription charge\b",
]

NEWSLETTER_SIGNALS = [
    r"\bnewsletter\b", r"\bdigest\b", r"\bweekly update\b", r"\bmonthly report\b",
    r"\bcompany news\b", r"\bproduct update\b", r"\bannouncement\b",
    r"\bno-reply\b", r"\bnoreply\b", r"\bnotification\b", r"\bdo not reply\b",
    r"\bautomated email\b", r"\bmarketing\b",
]

OPERATIONS_SIGNALS = [
    r"\boutage\b", r"\bdowntime\b", r"\bserver down\b", r"\bserver\b", r"\bdatabase\b",
    r"\bapi gateway\b", r"\blatency\b", r"\bincident\b", r"\bpostmortem\b", r"\bdeployment\b",
    r"\bcluster\b", r"\bkubernetes\b", r"\bproduction down\b", r"\bfailover\b",
    r"\bmaintenance\b", r"\bpatch\b", r"\bmonitoring alert\b", r"\bcpu usage\b"
]

EXTERNAL_SIGNALS = [
    r"\bclient\b", r"\bcustomer\b", r"\bvendor\b", r"\bpartner\b", r"\bcontract\b",
    r"\bexternal\b", r"\bproposal\b", r"\bnda\b", r"\bsla\b", r"\brfp\b",
    r"\bconsultant\b", r"\bagreement\b", r"\bdeliverable\b", r"\baccount manager\b"
]

SOCIAL_SIGNALS = [
    r"\blinkedin\b", r"\btwitter\b", r"\bfacebook\b", r"\binstagram\b",
    r"\byou have a new connection\b", r"\bsomeone liked\b", r"\bfriend request\b",
    r"\bfollow request\b", r"\bmentioned you\b", r"\bcomment on\b",
    r"\bjoin our community\b", r"\binvited you\b",
]


def _count_signals(text: str, patterns: list) -> int:
    """Count how many signal patterns match in text (case-insensitive)."""
    count = 0
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            count += 1
    return count


class EmailCategoryClassifier:
    """
    Rule-based multi-label email category classifier.
    Returns a primary category and confidence score.
    """

    CATEGORY_ORDER = [
        "SPAM", "OPERATIONS", "URGENT", "ACTION_REQUIRED", "MEETING",
        "FINANCE", "EXTERNAL", "NEWSLETTER", "SOCIAL", "GENERAL"
    ]

    CATEGORY_COLORS = {
        "SPAM":             "#ef4444",   # red
        "OPERATIONS":       "#e11d48",   # rose
        "URGENT":           "#f97316",   # orange
        "ACTION_REQUIRED":  "#8b5cf6",   # purple
        "MEETING":          "#3b82f6",   # blue
        "FINANCE":          "#10b981",   # emerald
        "EXTERNAL":         "#06b6d4",   # cyan
        "NEWSLETTER":       "#64748b",   # slate
        "SOCIAL":           "#ec4899",   # pink
        "GENERAL":          "#64748b",   # gray
    }

    CATEGORY_ICONS = {
        "SPAM":             "🚫",
        "OPERATIONS":       "⚡",
        "URGENT":           "🚨",
        "ACTION_REQUIRED":  "✅",
        "MEETING":          "📅",
        "FINANCE":          "💰",
        "EXTERNAL":         "🌐",
        "NEWSLETTER":       "📰",
        "SOCIAL":           "💬",
        "GENERAL":          "📧",
    }

    CATEGORY_LABELS = {
        "SPAM":             "Spam & Phishing",
        "OPERATIONS":       "Operations & Outages",
        "URGENT":           "Urgent Attention",
        "ACTION_REQUIRED":  "Action Required",
        "MEETING":          "Meeting & Calendar",
        "FINANCE":          "Financial & Invoices",
        "EXTERNAL":         "Client & External",
        "NEWSLETTER":       "Newsletters & Subscriptions",
        "SOCIAL":           "Social & Networking",
        "GENERAL":          "General Communication",
    }

    def classify(self, subject: str, body: str, sender: str = "") -> Dict[str, Any]:
        """
        Classify a single email into a primary category.
        Returns dict with category, label, confidence, icon, color, and all_scores.
        """
        combined = f"{subject} {body} {sender}".lower()

        scores = {
            "SPAM":             _count_signals(combined, SPAM_SIGNALS),
            "OPERATIONS":       _count_signals(combined, OPERATIONS_SIGNALS),
            "URGENT":           _count_signals(combined, URGENT_SIGNALS),
            "ACTION_REQUIRED":  _count_signals(combined, ACTION_SIGNALS),
            "MEETING":          _count_signals(combined, MEETING_SIGNALS),
            "FINANCE":          _count_signals(combined, FINANCE_SIGNALS),
            "EXTERNAL":         _count_signals(combined, EXTERNAL_SIGNALS),
            "NEWSLETTER":       _count_signals(combined, NEWSLETTER_SIGNALS),
            "SOCIAL":           _count_signals(combined, SOCIAL_SIGNALS),
            "GENERAL":          0,
        }

        # Normalize scores to confidence (0–1)
        total = sum(scores.values()) or 1
        normalized = {k: round(v / total, 3) for k, v in scores.items()}

        # Pick winning category in priority order
        best_category = "GENERAL"
        best_score = 0
        for cat in self.CATEGORY_ORDER:
            if scores[cat] > best_score:
                best_score = scores[cat]
                best_category = cat

        confidence = normalized[best_category] if best_score > 0 else 0.5

        return {
            "category": best_category,
            "category_label": self.CATEGORY_LABELS.get(best_category, best_category),
            "confidence": confidence,
            "icon": self.CATEGORY_ICONS[best_category],
            "color": self.CATEGORY_COLORS[best_category],
            "all_scores": normalized,
        }
