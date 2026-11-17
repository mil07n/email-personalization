"""
Email Summarization Module.
Extractive summarization using sentence-level TF-IDF importance scoring,
positional weighting, and cue-phrase boosting.
Includes ROUGE-1, ROUGE-2, and ROUGE-L evaluation without external dependencies.
"""

import re
from collections import Counter
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.preprocessor import EmailPreprocessor


class EmailSummarizer:
    """
    Extractive multi-turn conversation and thread summarizer.
    Incorporates noise/boilerplate stripping, Maximal Marginal Relevance (MMR)
    for diverse sentence selection, and structured Executive 3-Part Briefing.
    """

    # Boilerplate patterns to strip
    SIGNATURE_PATTERNS = [
        r"(?i)\n\s*--+\s*\n.*$",
        r"(?i)\n\s*(?:best|warm|kind|regards|sincerely|thanks|thank you|cheers)[,\s].*$",
        r"(?i)\bSent from my (?:iPhone|iPad|Android|Galaxy|mobile device)\b.*$",
        r"(?i)\bSent from (?:Mail|Outlook) for Windows\b.*$",
        r"(?i)-----Original Message-----.*$",
        r"(?i)\n\s*On\s+.+wrote:\s*\n.*$",
        r"(?i)This e-?mail (?:is confidential|and any attachments|message is intended).*$",
    ]

    # Cue phrases signaling important decisions or actions
    DECISION_CUES = [
        r"\b(?:we have decided|decision is|agreed to|resolution|we will proceed with)\b",
        r"\b(?:approved|confirmed|finalized|conclusion|outcome)\b",
        r"\b(?:action item|next steps?|please ensure|scheduled for)\b",
        r"\b(?:deadline is|due by|in summary|to summarize)\b"
    ]

    def __init__(self, preprocessor: Optional[EmailPreprocessor] = None):
        self.preprocessor = preprocessor or EmailPreprocessor()

    def strip_boilerplate(self, text: str) -> str:
        """Strip signatures, quoted chains, and confidentiality disclaimers."""
        cleaned = text or ""
        for pat in self.SIGNATURE_PATTERNS:
            cleaned = re.sub(pat, "", cleaned, flags=re.DOTALL | re.MULTILINE)
        return cleaned.strip()

    def summarize_thread(
        self,
        messages: List[Dict[str, Any]],
        max_sentences: int = 3
    ) -> Dict[str, Any]:
        """
        Summarize a list of messages in an email thread.
        Each message dict has: 'sender', 'body' or 'message_body', 'subject'.
        Returns:
            {
                "summary": "...",
                "executive_brief": {
                    "tldr": "...",
                    "key_decision": "...",
                    "next_steps": "..."
                },
                "key_points": ["...", "..."],
                "action_items": ["..."],
                "participants": ["..."]
            }
        """
        if not messages:
            return {
                "summary": "No messages in conversation.",
                "executive_brief": {
                    "tldr": "No messages to summarize.",
                    "key_decision": "None",
                    "next_steps": "None"
                },
                "key_points": [],
                "action_items": [],
                "participants": []
            }

        # Collect and clean message bodies with boilerplate stripping
        all_sentences = []
        participants = []

        for msg_idx, msg in enumerate(messages):
            sender = msg.get("sender", f"Participant {msg_idx+1}")
            if sender not in participants:
                participants.append(sender)

            body = msg.get("body") or msg.get("message_body") or ""
            stripped_body = self.strip_boilerplate(body)
            cleaned_body = self.preprocessor.clean_for_nlp(stripped_body)
            sents = self.preprocessor.tokenize_sentences(cleaned_body)

            for s_idx, sent in enumerate(sents):
                if len(sent.split()) >= 4:
                    all_sentences.append({
                        "text": sent,
                        "sender": sender,
                        "msg_idx": msg_idx,
                        "sent_idx": s_idx,
                        "is_first_msg": (msg_idx == 0),
                        "is_last_msg": (msg_idx == len(messages) - 1)
                    })

        if not all_sentences:
            return {
                "summary": "Empty conversation content.",
                "executive_brief": {
                    "tldr": "No significant email body text found.",
                    "key_decision": "None",
                    "next_steps": "None"
                },
                "key_points": [],
                "action_items": [],
                "participants": participants
            }

        if len(all_sentences) <= max_sentences:
            summary_text = " ".join(s["text"] for s in all_sentences)
            return {
                "summary": summary_text,
                "executive_brief": {
                    "tldr": all_sentences[0]["text"] if all_sentences else summary_text,
                    "key_decision": all_sentences[-1]["text"] if len(all_sentences) > 1 else "Reviewed.",
                    "next_steps": "Follow up on conversation."
                },
                "key_points": [s["text"] for s in all_sentences],
                "action_items": [],
                "participants": participants
            }

        # Compute TF-IDF scores for sentences
        sentence_texts = [s["text"] for s in all_sentences]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(sentence_texts)
            base_scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
        except Exception:
            base_scores = np.ones(len(sentence_texts))
            tfidf_matrix = None

        # Adjust scores by position and cue phrases
        final_scores = np.zeros(len(all_sentences))
        for i, s in enumerate(all_sentences):
            score = float(base_scores[i])

            if s["is_first_msg"] and s["sent_idx"] == 0:
                score *= 1.40  # Problem statement / opening request
            if s["is_last_msg"]:
                score *= 1.30  # Latest status / resolution

            text_lower = s["text"].lower()
            for cue in self.DECISION_CUES:
                if re.search(cue, text_lower):
                    score *= 1.50
                    break

            final_scores[i] = score

        # Maximal Marginal Relevance (MMR) selection
        selected_indices = []
        if tfidf_matrix is not None and len(all_sentences) > max_sentences:
            sim_matrix = cosine_similarity(tfidf_matrix)
            candidates = list(range(len(all_sentences)))
            lambda_param = 0.65  # Trade-off between relevance and diversity

            # Pick highest scoring first sentence
            best_first = int(np.argmax(final_scores))
            selected_indices.append(best_first)
            candidates.remove(best_first)

            while len(selected_indices) < max_sentences and candidates:
                mmr_scores = []
                for c in candidates:
                    rel = final_scores[c]
                    max_sim = max(sim_matrix[c, s] for s in selected_indices)
                    mmr = lambda_param * rel - (1 - lambda_param) * max_sim
                    mmr_scores.append((mmr, c))

                mmr_scores.sort(reverse=True)
                best_c = mmr_scores[0][1]
                selected_indices.append(best_c)
                candidates.remove(best_c)

            top_indices = sorted(selected_indices)
        else:
            top_indices = sorted(np.argsort(final_scores)[-max_sentences:])

        selected_sentences = [all_sentences[idx]["text"] for idx in top_indices]
        summary_text = " ".join(selected_sentences)

        # Extract structured executive 3-part elements
        tldr = selected_sentences[0] if selected_sentences else "Overview of message."
        
        # Decision point
        decision_sent = next(
            (s for s in selected_sentences if any(re.search(c, s.lower()) for c in self.DECISION_CUES)),
            selected_sentences[1] if len(selected_sentences) > 1 else tldr
        )

        # Next steps
        next_step_sent = next(
            (s for s in selected_sentences if any(w in s.lower() for w in ["please", "need", "ensure", "submit", "approve", "review", "schedule"])),
            selected_sentences[-1] if len(selected_sentences) > 2 else "No pending actions required."
        )

        return {
            "summary": summary_text,
            "executive_brief": {
                "tldr": tldr,
                "key_decision": decision_sent,
                "next_steps": next_step_sent
            },
            "key_points": selected_sentences,
            "action_items": [s for s in selected_sentences if any(w in s.lower() for w in ["please", "need", "should", "will", "deadline"])],
            "participants": participants
        }

    def summarize_single(self, text: str, max_sentences: int = 3) -> Dict[str, Any]:
        """Summarize a single email body with structured brief."""
        res = self.summarize_thread([{"body": text}], max_sentences=max_sentences)
        return res

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
        """Generate n-grams from a token list."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    @classmethod
    def compute_rouge(cls, hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
        """
        Compute ROUGE-1, ROUGE-2, and ROUGE-L (Precision, Recall, F1)
        without requiring external packages.
        """
        hyp_tokens = re.findall(r"\b\w+\b", hypothesis.lower())
        ref_tokens = re.findall(r"\b\w+\b", reference.lower())

        if not hyp_tokens or not ref_tokens:
            zero_score = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            return {"rouge1": zero_score, "rouge2": zero_score, "rougeL": zero_score}

        def score_ngrams(n: int):
            hyp_ngrams = Counter(cls._get_ngrams(hyp_tokens, n))
            ref_ngrams = Counter(cls._get_ngrams(ref_tokens, n))

            overlap = sum((hyp_ngrams & ref_ngrams).values())
            hyp_total = sum(hyp_ngrams.values())
            ref_total = sum(ref_ngrams.values())

            p = overlap / hyp_total if hyp_total > 0 else 0.0
            r = overlap / ref_total if ref_total > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

        # LCS for ROUGE-L
        def lcs(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    if x[i] == y[j]:
                        dp[i+1][j+1] = dp[i][j] + 1
                    else:
                        dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
            return dp[m][n]

        lcs_len = lcs(hyp_tokens, ref_tokens)
        r_l_p = lcs_len / len(hyp_tokens) if hyp_tokens else 0.0
        r_l_r = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        r_l_f1 = (2 * r_l_p * r_l_r) / (r_l_p + r_l_r) if (r_l_p + r_l_r) > 0 else 0.0

        return {
            "rouge1": score_ngrams(1),
            "rouge2": score_ngrams(2),
            "rougeL": {
                "precision": round(r_l_p, 4),
                "recall": round(r_l_r, 4),
                "f1": round(r_l_f1, 4)
            }
        }
