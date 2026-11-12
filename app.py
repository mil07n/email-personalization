"""
app.py – Phase 6: Gradio demo for the Email Personalization Model.

Importing this file DOES NOT start a server.
The server only starts under `if __name__ == "__main__":`.
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import gradio as gr
from personalize import personalize


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


def build_interface() -> gr.Blocks:
    """Build and return the Gradio Blocks interface (does NOT launch)."""

    with gr.Blocks(
        title="Email Personalization Model",
    ) as demo:
        gr.Markdown(
            """# 📧 Email Personalization Model
            
Powered by **TF-IDF**, **LSA**, **KMeans segmentation**, and **Word2Vec** embeddings.
NLP concepts: tokenization, stopword removal, lemmatization, TF-IDF, n-grams, LSA, 
POS tagging (HMM/Viterbi), cosine similarity retrieval, segment-based generation.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Recipient Profile")
                name_input = gr.Textbox(
                    label="Recipient Name",
                    placeholder="e.g. Alice Johnson",
                    elem_id="input_name",
                )
                profile_input = gr.Textbox(
                    label="Recipient work or past email",
                    placeholder="Describe the recipient's work or paste one of their past emails",
                    lines=5,
                    elem_id="input_profile",
                )
                purpose_input = gr.Dropdown(
                    label="Email Purpose",
                    choices=["follow_up", "offer", "update", "meeting_request", "introduction"],
                    value="follow_up",
                    elem_id="input_purpose",
                )
                retrieval_input = gr.Radio(
                    label="Retrieval Mode",
                    choices=["tfidf", "word2vec", "random"],
                    value="tfidf",
                    elem_id="input_retrieval",
                )
                generate_btn = gr.Button("✉️ Generate Personalized Email", variant="primary",
                                         elem_id="btn_generate")

            with gr.Column(scale=1):
                gr.Markdown("### 📨 Generated Email")
                subject_out  = gr.Textbox(label="Subject Line", elem_id="output_subject")
                body_out     = gr.Textbox(label="Email Body", lines=12, elem_id="output_body")
                segment_out  = gr.Textbox(label="Predicted Segment", elem_id="output_segment")
                topics_out   = gr.Textbox(label="Extracted Topics (POS-filtered nouns)", elem_id="output_topics")
                similar_out  = gr.Markdown(label="Similar Emails Retrieved", elem_id="output_similar")

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
            elem_id="examples",
        )

        gr.Markdown(
            """---
**Pipeline Overview:** Raw Email CSV → Preprocessing (tokenize, stopwords, lemmatize) 
→ TF-IDF + LSA → KMeans Segmentation → Word2Vec Embeddings → Personalization Engine → Gradio UI
"""
        )

    return demo


# Build the interface object at import time (so smoke test can verify it)
demo = build_interface()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=False, server_name="0.0.0.0", server_port=7860)
