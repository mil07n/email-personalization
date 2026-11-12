"""Parse and clean the AESLC Enron subject-line corpus."""

import re
import logging
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STOPWORDS = set(ENGLISH_STOP_WORDS) | {"enron", "ect", "hou", "forwarded"}
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_RE_PHONE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-])\d{3}[\s.-]\d{4}(?!\w)")
_RE_FORWARD = re.compile(
    r"(?im)^(?:[- ]*(?:forwarded message|original message|begin forwarded message)"
    r"|from:.*|sent:.*|to:.*|cc:.*|subject:.*)\s*$"
)
_RE_QUOTE = re.compile(r"(?m)^\s*>.*(?:\n|$)")
_RE_DISCLAIMER = re.compile(
    r"(?is)(?:this (?:e-?mail|message).{0,80}(?:confidential|intended recipient).*)$"
)
_RE_NONALPHA = re.compile(r"[^a-z\s]")


def parse_record(path: Path, split: str) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if "\n@subject" in raw:
        body, subject = raw.rsplit("\n@subject", 1)
    elif "@subject" in raw:
        body, subject = raw.rsplit("@subject", 1)
    else:
        body, subject = raw, ""
    return {
        "email_id": f"{split}/{path.stem}",
        "body": body.strip(),
        "subject_line": subject.strip().splitlines()[0].strip() if subject.strip() else "",
        "split": split,
    }


def clean_text(text: str) -> str:
    text = str(text)
    text = _RE_QUOTE.sub(" ", text)
    text = _RE_FORWARD.sub(" ", text)
    text = _RE_DISCLAIMER.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_PHONE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    words = re.findall(r"[a-z]{2,}", text.lower())
    return " ".join(w for w in words if w not in STOPWORDS)


def load_and_preprocess(csv_path=config.RAW_CSV) -> pd.DataFrame:
    root = Path(csv_path)
    records = []
    for split in ("train", "dev", "test"):
        for path in sorted((root / split).glob("*.subject")):
            records.append(parse_record(path, split))
    if not records:
        raise FileNotFoundError(f"No AESLC .subject records found under {root}")
    df = pd.DataFrame(records)
    df["clean"] = df["body"].map(clean_text)
    before = len(df)
    df = df[df["clean"].str.len() > 0].drop_duplicates("clean", keep="first").reset_index(drop=True)
    log.info("AESLC parsed=%d, nonempty/deduplicated=%d", before, len(df))
    df["tokens"] = df["clean"].str.split()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.AESLC_CSV, index=False)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    counts = df.groupby("split").size()
    report = [
        "# AESLC Data Report\n\n",
        "## Parsing\n\n",
        f"- Parsed `.subject` files: **{before}**\n",
        f"- Deduplicated cleaned emails: **{len(df)}**\n",
        "- Each record is parsed as body text before the final `@subject` marker and a subject line after it.\n",
        "- Original directory split is preserved; no random split is used.\n\n",
        "## Split Counts\n\n| Split | Emails |\n| --- | ---: |\n",
    ]
    report.extend(f"| {split} | {int(counts.get(split, 0))} |\n" for split in ("train", "dev", "test"))
    report.extend([
        "\n## Cleaning\n\n",
        "- Removed forwarded/original-message headers, quoted replies, disclaimers, URLs, email addresses, and phone numbers.\n",
        "- Added `enron`, `ect`, `hou`, and `forwarded` to stopwords.\n",
        "- Deduplication occurs before model fitting and preserves the first original split assignment.\n",
    ])
    (config.OUTPUTS_DIR / "data_report.md").write_text("".join(report))
    return df


if __name__ == "__main__":
    result = load_and_preprocess()
    print(f"Preprocessing complete: {len(result)} AESLC emails")
