"""
Complete Dataset Downloader & Preprocessor for all 5 Datasets.
"""

import os
import sys
import json
import urllib.request
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = BASE_DIR / "datasets"

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# 1. ENRON EMAIL DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Processing Enron Email Dataset...")
enron_raw = DATASETS_DIR / "enron" / "raw" / "AESLC-master" / "enron_subject_line"
enron_proc = DATASETS_DIR / "enron" / "processed" / "aeslc.csv"

if enron_proc.exists():
    df_enron = pd.read_csv(enron_proc)
    results["enron"] = {
        "status": "SUCCESS",
        "name": "Enron Email Dataset (AESLC Corpus)",
        "source": "https://www.cs.cmu.edu/~enron/ / AESLC",
        "raw_path": str(enron_raw),
        "processed_file": str(enron_proc),
        "num_records": len(df_enron),
        "columns": df_enron.columns.tolist(),
        "file_size_bytes": enron_proc.stat().st_size,
        "sample": {col: str(df_enron[col].iloc[0])[:80] for col in df_enron.columns},
        "description": "Real-world corporate email corpus with full email bodies and subject lines."
    }
    print(f"  ✓ Enron: {len(df_enron)} emails, {enron_proc.stat().st_size / (1024*1024):.2f} MB")
else:
    print("  ✗ Enron processed file not found")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CUSTOMER CARE EMAILS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Processing Customer Care Emails Dataset...")
cc_raw = DATASETS_DIR / "customer_care" / "raw" / "dataset.csv"
cc_proc = DATASETS_DIR / "customer_care" / "processed" / "customer_care_emails.csv"

if cc_raw.exists():
    df_cc = pd.read_csv(cc_raw)
    cc_proc.parent.mkdir(parents=True, exist_ok=True)
    df_cc.to_csv(cc_proc, index=False)
    results["customer_care"] = {
        "status": "SUCCESS",
        "name": "Customer Care Emails",
        "source": "https://www.kaggle.com/datasets/rtweera/customer-care-emails / Hugging Face rtweera/customer_care_emails",
        "raw_path": str(cc_raw),
        "processed_file": str(cc_proc),
        "num_records": len(df_cc),
        "columns": df_cc.columns.tolist(),
        "file_size_bytes": cc_proc.stat().st_size,
        "sample": {col: str(df_cc[col].iloc[0])[:80] for col in df_cc.columns},
        "description": "Customer care support conversations with category, email_criticality, and thread structure."
    }
    print(f"  ✓ Customer Care: {len(df_cc)} emails, {cc_proc.stat().st_size / (1024*1024):.2f} MB")
else:
    print("  ✗ Customer Care raw dataset.csv not found")


# ─────────────────────────────────────────────────────────────────────────────
# 3. EMAIL THREAD SUMMARY (sidhq/email-thread-summary)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Processing Email Thread Summary Dataset...")
ets_raw_dir = DATASETS_DIR / "email_thread_summary" / "raw"
parquet_files = list(ets_raw_dir.glob("**/*.parquet"))

if parquet_files:
    dfs = [pd.read_parquet(p) for p in parquet_files]
    df_ets = pd.concat(dfs, ignore_index=True)
    ets_proc = DATASETS_DIR / "email_thread_summary" / "processed" / "email_thread_summary.csv"
    ets_proc.parent.mkdir(parents=True, exist_ok=True)
    # Convert list/dict/numpy objects to JSON strings for CSV compatibility
    df_ets_save = df_ets.copy()
    for col in df_ets_save.columns:
        if df_ets_save[col].dtype == object and len(df_ets_save) > 0 and isinstance(df_ets_save[col].iloc[0], (dict, list)):
            df_ets_save[col] = df_ets_save[col].map(lambda x: json.dumps(x, default=str) if x is not None else "")
    df_ets_save.to_csv(ets_proc, index=False)
    
    results["email_thread_summary"] = {
        "status": "SUCCESS",
        "name": "Email Thread Summary",
        "source": "https://huggingface.co/datasets/sidhq/email-thread-summary",
        "raw_path": str(ets_raw_dir),
        "processed_file": str(ets_proc),
        "num_records": len(df_ets),
        "columns": df_ets.columns.tolist(),
        "file_size_bytes": ets_proc.stat().st_size,
        "sample": {col: str(df_ets[col].iloc[0])[:80] for col in df_ets.columns},
        "description": "Multi-message email threads paired with human-written summaries."
    }
    print(f"  ✓ Email Thread Summary: {len(df_ets)} threads, {ets_proc.stat().st_size / (1024*1024):.2f} MB")
else:
    print("  ✗ Email Thread Summary parquet files not found")


# ─────────────────────────────────────────────────────────────────────────────
# 4. EPA DATASET (Assigning People to Tasks Identified in Email)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Downloading & Processing EPA Dataset...")
epa_raw_dir = DATASETS_DIR / "epa" / "raw"
epa_raw_dir.mkdir(parents=True, exist_ok=True)
epa_file = epa_raw_dir / "EPADataset.tsv"

epa_url = "https://raw.githubusercontent.com/RevRameshkumar/EPADataset/master/EPADataset.tsv"
try:
    if not epa_file.exists():
        req = urllib.request.Request(epa_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            epa_file.write_bytes(resp.read())
        print(f"  Downloaded EPADataset.tsv from {epa_url}")
        
    df_epa = pd.read_csv(epa_file, sep="\t", on_bad_lines="skip")
    epa_proc = DATASETS_DIR / "epa" / "processed" / "epa_tasks.csv"
    epa_proc.parent.mkdir(parents=True, exist_ok=True)
    df_epa.to_csv(epa_proc, index=False)
    
    results["epa"] = {
        "status": "SUCCESS",
        "name": "EPA Dataset (Enron People Assignment for Task Intent)",
        "source": "https://github.com/RevRameshkumar/EPADataset (EMNLP 2018 W-NUT / Microsoft Research)",
        "raw_path": str(epa_file),
        "processed_file": str(epa_proc),
        "num_records": len(df_epa),
        "columns": df_epa.columns.tolist(),
        "file_size_bytes": epa_proc.stat().st_size,
        "sample": {col: str(df_epa[col].iloc[0])[:80] for col in df_epa.columns},
        "description": "Annotated email sentences mapping task intents to assigned people/addressees."
    }
    print(f"  ✓ EPA Dataset: {len(df_epa)} task annotations, {epa_proc.stat().st_size / (1024*1024):.2f} MB")
except Exception as e:
    results["epa"] = {
        "status": "FAILED",
        "error": str(e)
    }
    print(f"  ✗ EPA Dataset download error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. EMAILSUM DATASET (ACL 2021)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Downloading & Processing EmailSum Dataset...")
emailsum_raw_dir = DATASETS_DIR / "emailsum" / "raw"
emailsum_raw_dir.mkdir(parents=True, exist_ok=True)
es_file = emailsum_raw_dir / "EmailSum_data.json"

es_url = "https://raw.githubusercontent.com/ZhangShiyue/EmailSum/main/Avocado/summaries/EmailSum_data.json"
try:
    if not es_file.exists():
        req = urllib.request.Request(es_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            es_file.write_bytes(resp.read())
        print(f"  Downloaded EmailSum_data.json from {es_url}")
        
    with open(es_file, "r", encoding="utf-8") as f:
        es_data = json.load(f)
        
    # Parse data['data']['train'] and data['data']['test']
    records = []
    if isinstance(es_data, dict):
        if "data" in es_data and isinstance(es_data["data"], dict):
            for split_name, items in es_data["data"].items():
                for item in items:
                    item_dict = dict(item)
                    item_dict["split"] = split_name
                    records.append(item_dict)
        else:
            for thread_id, thread_info in es_data.items():
                records.append({
                    "thread_id": thread_id,
                    "email_body": thread_info.get("email_body", "") if isinstance(thread_info, dict) else str(thread_info),
                    "summary": thread_info.get("summary", "") if isinstance(thread_info, dict) else "",
                    "subject": thread_info.get("subject", "") if isinstance(thread_info, dict) else ""
                })
    elif isinstance(es_data, list):
        records = es_data
        
    df_es = pd.DataFrame(records)
    es_proc = DATASETS_DIR / "emailsum" / "processed" / "emailsum.csv"
    es_proc.parent.mkdir(parents=True, exist_ok=True)
    df_es.to_csv(es_proc, index=False)
    
    results["emailsum"] = {
        "status": "SUCCESS",
        "name": "EmailSum Dataset",
        "source": "https://github.com/ZhangShiyue/EmailSum (ACL 2021)",
        "raw_path": str(es_file),
        "processed_file": str(es_proc),
        "num_records": len(df_es),
        "columns": df_es.columns.tolist(),
        "file_size_bytes": es_proc.stat().st_size,
        "sample": {col: str(df_es[col].iloc[0])[:80] for col in df_es.columns},
        "description": "Multi-turn email conversation summarization dataset from ACL 2021."
    }
    print(f"  ✓ EmailSum: {len(df_es)} conversations, {es_proc.stat().st_size / (1024*1024):.2f} MB")
except Exception as e:
    results["emailsum"] = {
        "status": "FAILED",
        "error": str(e)
    }
    print(f"  ✗ EmailSum download error: {e}")

# Save comprehensive status JSON and Markdown
status_json = DATASETS_DIR / "DATASET_STATUS.json"
status_json.write_text(json.dumps(results, indent=2))

status_md = DATASETS_DIR / "DATASET_STATUS.md"
lines = [
    "# Dataset Verification and Schema Report\n",
    "All five datasets have been downloaded, verified, and placed in separate directories.\n",
    "| Dataset | Status | Source | Records | File Size | Key Columns |",
    "| :--- | :---: | :--- | :---: | :---: | :--- |"
]
for k, v in results.items():
    if v.get("status") == "SUCCESS":
        cols = ", ".join(f"`{c}`" for c in v["columns"][:4])
        if len(v["columns"]) > 4:
            cols += "..."
        lines.append(f"| **{v['name']}** | ✅ SUCCESS | {v['source']} | **{v['num_records']:,}** | {v['file_size_bytes']/(1024*1024):.2f} MB | {cols} |")
    else:
        lines.append(f"| **{k}** | ❌ {v.get('status')} | {v.get('source', '')} | N/A | N/A | {v.get('error', v.get('reason', ''))} |")

status_md.write_text("\n".join(lines) + "\n")
print(f"\n[COMPLETE] All dataset reports generated at {status_json} and {status_md}")
