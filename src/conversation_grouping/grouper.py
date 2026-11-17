"""
Conversation Grouping Module.
Groups related emails into multi-turn threads using thread IDs when available,
and fallback subject normalization, participant matching, and TF-IDF similarity clustering.
Evaluated with Adjusted Rand Index (ARI).
"""

import re
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score


class ConversationGrouper:
    """
    Groups emails into coherent conversation threads.
    Combines exact thread metadata with semantic & participant heuristics.
    """

    RE_SUBJECT_PREFIXES = re.compile(r"^(?:re|fwd|fw|aw|sv|vs)\s*:\s*", re.IGNORECASE)

    def __init__(self, similarity_threshold: float = 0.35):
        self.similarity_threshold = similarity_threshold

    def clean_subject(self, subject: str) -> str:
        """Strip 'Re:', 'Fwd:', 'FW:' prefixes to extract the conversation core."""
        s = str(subject or "").strip()
        while True:
            new_s = self.RE_SUBJECT_PREFIXES.sub("", s).strip()
            if new_s == s:
                break
            s = new_s
        return s.lower()

    def group_emails(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group a list of email dicts into conversation clusters.
        Each email dict should have: 'subject', 'body', and optionally 'sender', 'thread_id', 'timestamp'.
        Returns list of conversation dicts:
        [{
            "conversation_id": "thread_0",
            "subject": "Core conversation subject",
            "num_emails": 3,
            "participants": ["Alice", "Bob"],
            "emails": [...chronologically sorted emails...]
        }]
        """
        if not emails:
            return []

        # 1. If explicit thread_id exists on most items, group by thread_id
        has_thread_ids = sum(1 for e in emails if e.get("thread_id")) / len(emails) >= 0.5
        if has_thread_ids:
            return self._group_by_explicit_thread(emails)

        # 2. Heuristic grouping: Normalized Subject + Participant matching + TF-IDF
        return self._group_by_heuristics(emails)

    def _group_by_explicit_thread(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group by explicit thread_id field."""
        groups = {}
        for idx, email in enumerate(emails):
            tid = str(email.get("thread_id") or f"unthreaded_{idx}")
            if tid not in groups:
                groups[tid] = []
            groups[tid].append(email)

        conversations = []
        for tid, group_emails in groups.items():
            # Sort chronologically if timestamps are available
            group_emails.sort(key=lambda x: str(x.get("timestamp", "")))
            core_subj = group_emails[0].get("subject", "Conversation Thread")
            participants = list(dict.fromkeys(
                [e.get("sender") for e in group_emails if e.get("sender")] +
                [e.get("receiver") for e in group_emails if e.get("receiver")]
            ))

            conversations.append({
                "conversation_id": tid,
                "subject": core_subj,
                "num_emails": len(group_emails),
                "participants": participants,
                "emails": group_emails
            })

        return conversations

    def _group_by_heuristics(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group emails by normalized subject and content similarity."""
        n = len(emails)
        if n == 1:
            return [{
                "conversation_id": "conv_0",
                "subject": emails[0].get("subject", "Conversation"),
                "num_emails": 1,
                "participants": [emails[0].get("sender", "Unknown")],
                "emails": emails
            }]

        # Prepare normalized subjects and texts
        norm_subjects = [self.clean_subject(e.get("subject", "")) for e in emails]
        texts = [f"{e.get('subject', '')} {e.get('body', '')}" for e in emails]

        # Compute TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        tfidf_mat = vectorizer.fit_transform(texts)
        sim_mat = cosine_similarity(tfidf_mat)

        # Cluster using connected components / greedy grouping
        assigned = [-1] * n
        clusters = []
        cluster_id = 0

        for i in range(n):
            if assigned[i] != -1:
                continue

            assigned[i] = cluster_id
            current_members = [i]

            sender_i = str(emails[i].get("sender", "")).lower().strip()
            receiver_i = str(emails[i].get("receiver", "")).lower().strip()
            parts_i = {sender_i, receiver_i} - {"", "unknown"}

            for j in range(i + 1, n):
                if assigned[j] != -1:
                    continue

                sender_j = str(emails[j].get("sender", "")).lower().strip()
                receiver_j = str(emails[j].get("receiver", "")).lower().strip()
                parts_j = {sender_j, receiver_j} - {"", "unknown"}

                # Participant overlap
                has_participant_overlap = bool(parts_i & parts_j) if (parts_i and parts_j) else True

                # Match condition 1: Identical cleaned subject with participant or content coherence
                same_subject = (norm_subjects[i] == norm_subjects[j]) and len(norm_subjects[i]) > 3
                # Match condition 2: High TF-IDF body similarity (> 0.35)
                high_similarity = sim_mat[i, j] >= self.similarity_threshold

                if (same_subject and has_participant_overlap) or (same_subject and high_similarity) or (high_similarity and has_participant_overlap):
                    assigned[j] = cluster_id
                    current_members.append(j)

            clusters.append(current_members)
            cluster_id += 1

        conversations = []
        for cid, member_indices in enumerate(clusters):
            member_emails = [emails[idx] for idx in member_indices]
            core_subj = member_emails[0].get("subject", f"Conversation {cid+1}")
            participants = list(dict.fromkeys(
                [e.get("sender") for e in member_emails if e.get("sender")]
            ))

            conversations.append({
                "conversation_id": f"conv_{cid}",
                "subject": core_subj,
                "num_emails": len(member_emails),
                "participants": participants or ["Various"],
                "emails": member_emails
            })

        return conversations

    def evaluate_ari(self, true_labels: List[Any], predicted_labels: List[Any]) -> float:
        """Calculate Adjusted Rand Index (ARI) comparing predicted clustering to ground truth."""
        return float(adjusted_rand_score(true_labels, predicted_labels))
