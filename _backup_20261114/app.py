"""
app.py – Gradio demo with two tabs:
  Tab 1 – Outbound Email Personalization (original feature)
  Tab 2 – Inbound Email Urgency Classifier + Auto-Draft Reply (new feature)

Importing this file DOES NOT start a server.
The server only starts under `if __name__ == "__main__":`.
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr
from personalize import personalize
from urgency import classify_urgency


# ── Tab 1 helpers ─────────────────────────────────────────────────────────────

def _format_similar(similar_emails: list) -> str:
    if not similar_emails:
        return "No similar emails found."
    lines = []
    for i, em in enumerate(similar_emails, 1):
        lines.append(
            f"**{i}. {em.get('subject', '(no subject)')}** "
            f"(score: {em.get('score', 0):.3f})\n\n"
            f"> {em.get('snippet', '')[:150]}…"
        )
    return "\n\n---\n\n".join(lines)


def run_personalization(name: str, profile_text: str,
                        purpose: str, retrieval_mode: str):
    """Gradio callback: call personalize() and format outputs."""
    try:
        result = personalize(
            name=name,
            profile_text=profile_text,
            purpose=purpose,
            retrieval=retrieval_mode,
        )
        subject   = result["subject"]
        body      = result["body"]
        segment   = f"{result['segment_label']} (segment {result['segment_id']})"
        topics    = ", ".join(result["topics"])
        similar   = _format_similar(result["similar_emails"])
        return subject, body, segment, topics, similar
    except Exception as e:
        err = f"Error: {e}"
        return err, err, "unknown", "unknown", "none"


# ── Tab 2 helpers ─────────────────────────────────────────────────────────────

_URGENCY_COLORS = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
}

def run_urgency(sender_name: str, email_subject: str, email_body: str):
    """Gradio callback: classify urgency and return an auto-draft reply."""
    if not email_subject.strip() and not email_body.strip():
        return "⚠️ Please enter an email subject and/or body.", "", "", "", ""
    try:
        r = classify_urgency(email_subject, email_body, sender_name or "Team")
        badge_md = (
            f"## {r.badge}\n\n"
            f"**Confidence:** {r.confidence.upper()}\n\n"
            f"**Raw Score:** {r.score}\n\n"
            f"**Why:** {r.explanation}"
        )
        signals_text = "\n".join(f"• {s}" for s in r.signals_found[:10]) or "None detected"
        return badge_md, signals_text, r.draft_subject, r.draft_body, r.level
    except Exception as e:
        return f"Error: {e}", "", "", "", ""


# ── Interface ─────────────────────────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    """Build and return the Gradio Blocks interface (does NOT launch)."""

    with gr.Blocks(title="Email Intelligence Suite") as demo:
        gr.Markdown(
            """# 📧 Email Intelligence Suite
Powered by **TF-IDF · LSA · KMeans · Word2Vec** (outbound) and **rule-based urgency scoring** (inbound).
"""
        )

        with gr.Tabs():

            # ── TAB 1: Outbound Personalization ──────────────────────────────
            with gr.Tab("✉️ Personalized Email Generator"):
                gr.Markdown(
                    "Generate a **personalized outbound email** based on a recipient's profile. "
                    "Uses TF-IDF, LSA, KMeans segmentation, and Word2Vec retrieval."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Recipient Profile")
                        name_input = gr.Textbox(
                            label="Recipient Name",
                            placeholder="e.g. Alice Johnson",
                        )
                        profile_input = gr.Textbox(
                            label="Recipient work or past email",
                            placeholder="Describe the recipient's work or paste one of their past emails",
                            lines=5,
                        )
                        purpose_input = gr.Dropdown(
                            label="Email Purpose",
                            choices=["follow_up", "offer", "update", "meeting_request", "introduction"],
                            value="follow_up",
                        )
                        retrieval_input = gr.Radio(
                            label="Retrieval Mode",
                            choices=["tfidf", "word2vec", "random"],
                            value="tfidf",
                        )
                        generate_btn = gr.Button("✉️ Generate Personalized Email", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📨 Generated Email")
                        subject_out  = gr.Textbox(label="Subject Line")
                        body_out     = gr.Textbox(label="Email Body", lines=12)
                        segment_out  = gr.Textbox(label="Predicted Segment")
                        topics_out   = gr.Textbox(label="Extracted Topics (POS-filtered nouns)")
                        similar_out  = gr.Markdown(label="Similar Emails Retrieved")

                generate_btn.click(
                    fn=run_personalization,
                    inputs=[name_input, profile_input, purpose_input, retrieval_input],
                    outputs=[subject_out, body_out, segment_out, topics_out, similar_out],
                )

                gr.Examples(
                    examples=[
                        [
                            "Alice Johnson",
                            "I manage natural gas pipeline nominations, transport capacity, and daily flow schedules.",
                            "follow_up",
                            "tfidf",
                        ],
                        [
                            "Bob Chen",
                            "I review ISDA agreements, credit terms, and power trading contracts for the commercial team.",
                            "offer",
                            "word2vec",
                        ],
                        [
                            "Carol Patel",
                            "Please schedule the executive meeting, circulate the agenda, and track the action items afterward.",
                            "introduction",
                            "tfidf",
                        ],
                    ],
                    inputs=[name_input, profile_input, purpose_input, retrieval_input],
                    label="Example Inputs",
                )

                gr.Markdown(
                    """---
**Pipeline Overview:** Raw Email CSV → Preprocessing (tokenize, stopwords, lemmatize)
→ TF-IDF + LSA → KMeans Segmentation → Word2Vec Embeddings → Personalization Engine → Gradio UI
"""
                )

            # ── TAB 2: Inbound Urgency Classifier ────────────────────────────
            with gr.Tab("🚨 Urgency Classifier & Auto-Draft"):
                gr.Markdown(
                    "Paste an **incoming email** to classify its urgency (**CRITICAL / HIGH / MEDIUM / LOW**) "
                    "and instantly get an **auto-draft reply** you can send right away.\n\n"
                    "> **NLP techniques used:** Bag-of-words keyword matching, regex pattern scoring, "
                    "weighted feature aggregation, time-expression detection."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 Incoming Email")
                        sender_input  = gr.Textbox(
                            label="Sender Name",
                            placeholder="e.g. John Smith",
                        )
                        subj_input    = gr.Textbox(
                            label="Email Subject",
                            placeholder="e.g. URGENT: Server is down!",
                        )
                        body_input    = gr.Textbox(
                            label="Email Body",
                            placeholder="Paste the full email body here…",
                            lines=10,
                        )
                        classify_btn  = gr.Button("🔍 Classify Urgency + Generate Draft", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Urgency Analysis")
                        urgency_badge  = gr.Markdown(label="Urgency Level")
                        signals_out    = gr.Textbox(label="Signals Detected", lines=6)

                        gr.Markdown("### 📝 Auto-Draft Reply")
                        draft_subj_out = gr.Textbox(label="Draft Subject (editable)")
                        draft_body_out = gr.Textbox(label="Draft Body (editable)", lines=12)
                        level_out      = gr.Textbox(label="Level Code", visible=False)

                classify_btn.click(
                    fn=run_urgency,
                    inputs=[sender_input, subj_input, body_input],
                    outputs=[urgency_badge, signals_out, draft_subj_out, draft_body_out, level_out],
                )

                gr.Examples(
                    examples=[
                        [
                            "Alice",
                            "URGENT: Production server crashed – all customers affected",
                            "Our production database has gone offline as of 10 AM. All customers are reporting errors. "
                            "We need this resolved ASAP. The CEO has been notified.",
                        ],
                        [
                            "Bob",
                            "Contract renewal deadline – ACTION REQUIRED by EOD",
                            "Hi, the contract with Vendor X expires tomorrow. We need your sign-off by end of day today "
                            "or we face a compliance violation. Please review and approve immediately.",
                        ],
                        [
                            "Carol",
                            "Team meeting – agenda for Friday",
                            "Hi everyone, just a reminder that we have our weekly team meeting on Friday at 2 PM. "
                            "Please review the attached agenda and confirm attendance.",
                        ],
                        [
                            "Dave",
                            "Monthly Newsletter – November 2026",
                            "Hello! Here is your monthly newsletter with company updates. "
                            "No action required from your side. Have a great weekend!",
                        ],
                    ],
                    inputs=[sender_input, subj_input, body_input],
                    label="Try Example Emails",
                )

    return demo


# Build the interface object at import time (so smoke test can verify it)
demo = build_interface()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=False, server_name="127.0.0.1", server_port=7860)
