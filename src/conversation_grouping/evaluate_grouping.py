"""
Evaluation of Conversation Grouping against Customer Care ground-truth thread_id.
Computes Adjusted Rand Index (ARI), Normalized Mutual Info, and thread homogeneity.
"""

import sys
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config.config import CUSTOMER_CARE_CSV, EVALUATION_DIR
from src.conversation_grouping.grouper import ConversationGrouper


def evaluate_conversation_grouping(sample_size: int = 500):
    print("=" * 60)
    print(" EVALUATING CONVERSATION GROUPING")
    print("=" * 60)

    if not CUSTOMER_CARE_CSV.exists():
        print(f"Dataset not found at {CUSTOMER_CARE_CSV}")
        return

    df = pd.read_csv(CUSTOMER_CARE_CSV).head(sample_size)
    print(f"Evaluating {len(df)} emails across {df['thread_id'].nunique()} ground-truth threads...")

    emails = []
    for idx, row in df.iterrows():
        emails.append({
            "id": idx,
            "subject": row["subject"],
            "body": row["message_body"],
            "sender": row["sender"],
            "receiver": row["receiver"],
            # DO NOT pass thread_id so heuristic clustering is forced
        })

    grouper = ConversationGrouper(similarity_threshold=0.30)
    conversations = grouper._group_by_heuristics(emails)

    # Map predicted clusters
    pred_labels = [-1] * len(df)
    for c_idx, conv in enumerate(conversations):
        for em in conv["emails"]:
            pred_labels[em["id"]] = c_idx

    true_labels = df["thread_id"].tolist()

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    print("\nConversation Grouping Benchmark Results:")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Info:     {nmi:.4f}")
    print(f"  Predicted Conversations:    {len(conversations)} (vs True: {df['thread_id'].nunique()})")

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = EVALUATION_DIR / "conversation_grouping_metrics.json"
    metrics_data = {
        "dataset": "Customer Care Emails (Ground Truth: thread_id)",
        "sample_size": len(df),
        "true_threads": int(df["thread_id"].nunique()),
        "predicted_conversations": len(conversations),
        "adjusted_rand_index": round(float(ari), 4),
        "normalized_mutual_info": round(float(nmi), 4)
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    md_path = EVALUATION_DIR / "conversation_grouping_evaluation.md"
    md_content = f"""# Conversation Grouping Benchmark Report

**Benchmark Dataset:** Customer Care Emails (`dataset.csv`)  
**Evaluated Sample:** {len(df):,} messages across {df['thread_id'].nunique()} ground truth threads  

## Grouping Performance

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Adjusted Rand Index (ARI)** | **{ari:.4f}** | Measure of similarity between predicted clusters and true threads (chance = 0.0, perfect = 1.0) |
| **Normalized Mutual Information (NMI)** | **{nmi:.4f}** | Information shared between predicted clusters and true conversations |
| **Predicted Conversations** | **{len(conversations)}** | Grouped threads (True threads: {df['thread_id'].nunique()}) |
"""
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Saved metrics to {metrics_path} and {md_path}")


if __name__ == "__main__":
    evaluate_conversation_grouping(sample_size=400)
