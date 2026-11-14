"""
batch_mode.py – Stretch goal: batch personalize a CSV of recipients.

Usage:
    python batch_mode.py recipients.csv outputs/batch_emails.csv

Input CSV must have columns: name, profile_text
Optional columns: purpose, retrieval (defaults to follow_up / tfidf)
"""

import sys
import logging
from pathlib import Path

import pandas as pd

from personalize import personalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def batch_personalize(input_csv: Path, output_csv: Path):
    df = pd.read_csv(input_csv)

    required = ["name", "profile_text"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Input CSV must have column: {col!r}")

    results = []
    for _, row in df.iterrows():
        name         = str(row["name"])
        profile_text = str(row["profile_text"])
        purpose      = str(row.get("purpose", "follow_up")) if "purpose" in df.columns else "follow_up"
        retrieval    = str(row.get("retrieval", "tfidf"))   if "retrieval" in df.columns else "tfidf"

        try:
            res = personalize(name, profile_text, purpose, retrieval)
            results.append({
                "name":          name,
                "segment_label": res["segment_label"],
                "tone":          res["tone"],
                "topics":        ", ".join(res["topics"]),
                "subject":       res["subject"],
                "body":          res["body"],
            })
            log.info("Generated email for %s", name)
        except Exception as e:
            log.error("Failed for %s: %s", name, e)
            results.append({
                "name": name, "segment_label": "error", "tone": "error",
                "topics": "", "subject": "Error", "body": str(e),
            })

    out_df = pd.DataFrame(results)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    log.info("Batch done. %d emails written to %s", len(results), output_csv)
    return out_df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python batch_mode.py <input.csv> <output.csv>")
        sys.exit(1)
    batch_personalize(Path(sys.argv[1]), Path(sys.argv[2]))
