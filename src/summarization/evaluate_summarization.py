"""
Evaluation of Email Summarization against Email Thread Summary and EmailSum datasets.
Computes ROUGE-1, ROUGE-2, and ROUGE-L (Precision, Recall, F1).
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config.config import EMAIL_THREAD_SUMMARY_CSV, EMAILSUM_CSV, EVALUATION_DIR
from src.summarization.summarizer import EmailSummarizer


def evaluate_summarization(sample_size: int = 150):
    print("=" * 60)
    print(" EVALUATING EMAIL SUMMARIZATION (ROUGE-1, ROUGE-2, ROUGE-L)")
    print("=" * 60)

    summarizer = EmailSummarizer()
    results = {}

    # 1. Evaluate on Email Thread Summary Dataset (sidhq/email-thread-summary)
    if EMAIL_THREAD_SUMMARY_CSV.exists():
        print(f"\n[1/2] Evaluating on Email Thread Summary dataset ({EMAIL_THREAD_SUMMARY_CSV})...")
        df_ets = pd.read_csv(EMAIL_THREAD_SUMMARY_CSV).head(sample_size)

        scores = {"r1_p": [], "r1_r": [], "r1_f1": [],
                  "r2_p": [], "r2_r": [], "r2_f1": [],
                  "rl_p": [], "rl_r": [], "rl_f1": []}

        for _, row in df_ets.iterrows():
            thread_data = str(row["thread"])
            ref_summary = str(row["summary"])

            # Generate extractive summary
            gen_summary = summarizer.summarize_single(thread_data, max_sentences=3)

            # Compute ROUGE
            r_metrics = summarizer.compute_rouge(gen_summary, ref_summary)

            scores["r1_p"].append(r_metrics["rouge1"]["precision"])
            scores["r1_r"].append(r_metrics["rouge1"]["recall"])
            scores["r1_f1"].append(r_metrics["rouge1"]["f1"])

            scores["r2_p"].append(r_metrics["rouge2"]["precision"])
            scores["r2_r"].append(r_metrics["rouge2"]["recall"])
            scores["r2_f1"].append(r_metrics["rouge2"]["f1"])

            scores["rl_p"].append(r_metrics["rougeL"]["precision"])
            scores["rl_r"].append(r_metrics["rougeL"]["recall"])
            scores["rl_f1"].append(r_metrics["rougeL"]["f1"])

        results["email_thread_summary"] = {
            "samples_evaluated": len(df_ets),
            "rouge1": {
                "precision": round(float(np.mean(scores["r1_p"])), 4),
                "recall": round(float(np.mean(scores["r1_r"])), 4),
                "f1": round(float(np.mean(scores["r1_f1"])), 4)
            },
            "rouge2": {
                "precision": round(float(np.mean(scores["r2_p"])), 4),
                "recall": round(float(np.mean(scores["r2_r"])), 4),
                "f1": round(float(np.mean(scores["r2_f1"])), 4)
            },
            "rougeL": {
                "precision": round(float(np.mean(scores["rl_p"])), 4),
                "recall": round(float(np.mean(scores["rl_r"])), 4),
                "f1": round(float(np.mean(scores["rl_f1"])), 4)
            }
        }
        print("  ROUGE-1 F1:", results["email_thread_summary"]["rouge1"]["f1"])
        print("  ROUGE-2 F1:", results["email_thread_summary"]["rouge2"]["f1"])
        print("  ROUGE-L F1:", results["email_thread_summary"]["rougeL"]["f1"])

    # 2. Evaluate on EmailSum Dataset (ACL 2021)
    if EMAILSUM_CSV.exists():
        print(f"\n[2/2] Evaluating on EmailSum dataset ({EMAILSUM_CSV})...")
        df_es = pd.read_csv(EMAILSUM_CSV).head(sample_size)

        scores_es = {"r1_p": [], "r1_r": [], "r1_f1": [],
                     "r2_p": [], "r2_r": [], "r2_f1": [],
                     "rl_p": [], "rl_r": [], "rl_f1": []}

        for _, row in df_es.iterrows():
            body = str(row.get("email_body", "") or row.get("thread_id", ""))
            ref = str(row.get("short_summary", "") or row.get("summary", ""))

            gen = summarizer.summarize_single(body, max_sentences=2)
            r_metrics = summarizer.compute_rouge(gen, ref)

            scores_es["r1_p"].append(r_metrics["rouge1"]["precision"])
            scores_es["r1_r"].append(r_metrics["rouge1"]["recall"])
            scores_es["r1_f1"].append(r_metrics["rouge1"]["f1"])

            scores_es["r2_p"].append(r_metrics["rouge2"]["precision"])
            scores_es["r2_r"].append(r_metrics["rouge2"]["recall"])
            scores_es["r2_f1"].append(r_metrics["rouge2"]["f1"])

            scores_es["rl_p"].append(r_metrics["rougeL"]["precision"])
            scores_es["rl_r"].append(r_metrics["rougeL"]["recall"])
            scores_es["rl_f1"].append(r_metrics["rougeL"]["f1"])

        results["emailsum"] = {
            "samples_evaluated": len(df_es),
            "rouge1": {
                "precision": round(float(np.mean(scores_es["r1_p"])), 4),
                "recall": round(float(np.mean(scores_es["r1_r"])), 4),
                "f1": round(float(np.mean(scores_es["r1_f1"])), 4)
            },
            "rouge2": {
                "precision": round(float(np.mean(scores_es["r2_p"])), 4),
                "recall": round(float(np.mean(scores_es["r2_r"])), 4),
                "f1": round(float(np.mean(scores_es["r2_f1"])), 4)
            },
            "rougeL": {
                "precision": round(float(np.mean(scores_es["rl_p"])), 4),
                "recall": round(float(np.mean(scores_es["rl_r"])), 4),
                "f1": round(float(np.mean(scores_es["rl_f1"])), 4)
            }
        }
        print("  ROUGE-1 F1:", results["emailsum"]["rouge1"]["f1"])
        print("  ROUGE-2 F1:", results["emailsum"]["rouge2"]["f1"])
        print("  ROUGE-L F1:", results["emailsum"]["rougeL"]["f1"])

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    out_json = EVALUATION_DIR / "summarization_metrics.json"
    out_json.write_text(json.dumps(results, indent=2))

    out_md = EVALUATION_DIR / "summarization_evaluation.md"
    lines = [
        "# Email Summarization Evaluation Report\n",
        "Evaluated using **ROUGE-1**, **ROUGE-2**, and **ROUGE-L** metrics comparing extractive generated summaries with human reference summaries.\n"
    ]
    for dname, data in results.items():
        lines.append(f"## Dataset: `{dname}` ({data['samples_evaluated']} samples)\n")
        lines.append("| Metric | Precision | Recall | F1-Score |")
        lines.append("| :--- | :---: | :---: | :---: |")
        lines.append(f"| **ROUGE-1 (Unigram Overlap)** | {data['rouge1']['precision']:.4f} | {data['rouge1']['recall']:.4f} | **{data['rouge1']['f1']:.4f}** |")
        lines.append(f"| **ROUGE-2 (Bigram Overlap)** | {data['rouge2']['precision']:.4f} | {data['rouge2']['recall']:.4f} | **{data['rouge2']['f1']:.4f}** |")
        lines.append(f"| **ROUGE-L (Longest Common Subsequence)** | {data['rougeL']['precision']:.4f} | {data['rougeL']['recall']:.4f} | **{data['rougeL']['f1']:.4f}** |\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved metrics to {out_json} and {out_md}")


if __name__ == "__main__":
    evaluate_summarization(sample_size=150)
