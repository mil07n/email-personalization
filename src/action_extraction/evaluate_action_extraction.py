"""
Evaluation of Action-Item Extraction against the EPA Dataset.
Computes Precision, Recall, and F1 on task sentence detection and addressee assignment.
"""

import sys
import json
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

from config.config import EPA_CSV, EVALUATION_DIR
from src.action_extraction.extractor import ActionExtractor


def evaluate_action_extraction(sample_size: int = 500):
    print("=" * 60)
    print(" EVALUATING ACTION-ITEM EXTRACTION ON EPA DATASET")
    print("=" * 60)

    if not EPA_CSV.exists():
        print(f"EPA dataset not found at {EPA_CSV}")
        return

    df = pd.read_csv(EPA_CSV)
    print(f"Loaded {len(df)} EPA dataset records.")

    # Sample subset for rapid evaluation
    sample_df = df.head(sample_size)
    extractor = ActionExtractor()

    tp = 0
    fp = 0
    fn = 0
    person_matches = 0
    total_with_person = 0

    print(f"Evaluating {len(sample_df)} test samples...")

    for idx, row in sample_df.iterrows():
        try:
            data = json.loads(row["Json"])
            message = data.get("Message", "")
            gt_task = data.get("TaskSentence", "").strip()

            # Ground truth assigned person
            gt_person = ""
            judgements = data.get("Judgements", [])
            for j in judgements:
                for k, v in j.items():
                    if v and isinstance(v, list) and v[0]:
                        gt_person = v[0].split("@")[0].split(".")[0].title()
                        break
                if gt_person:
                    break

            # Run extractor
            extracted = extractor.extract(message)

            if extracted:
                matched_gt = False
                for item in extracted:
                    src = item["source_sentence"].lower()
                    if gt_task.lower() in src or src in gt_task.lower():
                        matched_gt = True
                        if gt_person and item["responsible_person"] != "Unassigned":
                            total_with_person += 1
                            if gt_person.lower() in item["responsible_person"].lower() or item["responsible_person"].lower() in gt_person.lower():
                                person_matches += 1
                        break

                if matched_gt:
                    tp += 1
                else:
                    fp += 1
            else:
                fn += 1
        except Exception:
            continue

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    person_acc = person_matches / total_with_person if total_with_person > 0 else 0.0

    print("\nResults on EPA Benchmark:")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall:.4f}")
    print(f"  F1-Score:        {f1:.4f}")
    print(f"  Addressee Match: {person_acc:.4f} ({person_matches}/{total_with_person})")

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = EVALUATION_DIR / "action_extraction_metrics.json"
    metrics_data = {
        "dataset": "EPA Dataset (Enron People Assignment for Task Intent)",
        "sample_size": len(sample_df),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "addressee_accuracy": round(person_acc, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    md_path = EVALUATION_DIR / "action_extraction_evaluation.md"
    md_content = f"""# Action-Item Extraction & Addressee Tagging Benchmark

**Benchmark Dataset:** EPA Dataset (`EPADataset.tsv`, Microsoft Research / EMNLP W-NUT)  
**Evaluated Samples:** {len(sample_df):,} annotated task sentences  

## Performance Metrics

| Metric | Score | Notes |
| :--- | :---: | :--- |
| **Precision** | **{precision:.4f}** | Proportion of extracted action items that match ground-truth intent |
| **Recall** | **{recall:.4f}** | Proportion of actual task sentences successfully identified |
| **F1-Score** | **{f1:.4f}** | Harmonic mean of task extraction precision and recall |
| **Addressee Accuracy** | **{person_acc:.4f}** | Correct assignment of the responsible individual |
"""
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Saved metrics to {metrics_path} and {md_path}")


if __name__ == "__main__":
    evaluate_action_extraction(sample_size=300)
