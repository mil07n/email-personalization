# AESLC Email Personalization Report

## Dataset and protocol

The project uses the AESLC Enron subject-line corpus in
`AESLC-master/enron_subject_line`. Each `.subject` file is parsed at its final
`@subject` marker into a body and subject line while preserving the original
train/dev/test split. Forwarded headers, quoted replies, disclaimers, URLs,
email addresses, phone numbers, and Enron boilerplate are cleaned. Exact
duplicates are removed before fitting.

TF-IDF, LSA, KMeans, and Word2Vec are fitted on train emails only. Evaluation
uses held-out test emails only. Counts are in
[outputs/data_report.md](outputs/data_report.md).

## Main metric: held-out subject retrieval

The train corpus is the retrieval index. For every test email, the system
retrieves three train emails using TF-IDF/LSA, Word2Vec, and a random baseline.
The primary metric is token-overlap F1 between each retrieved subject and the
true held-out subject:

| Method | Top-1 subject F1 | Top-3 subject F1 |
|---|---:|---:|
| TF-IDF + LSA | 0.0981 | 0.0772 |
| Word2Vec | 0.0621 | 0.0508 |
| Random | 0.0057 | 0.0055 |

These are held-out numbers, generated in
[outputs/retrieval_comparison.csv](outputs/retrieval_comparison.csv), with
per-email details in [outputs/retrieval_results.csv](outputs/retrieval_results.csv).

## Unsupervised segmentation

There is no supervised classifier. KMeans is selected from k=5..12 by train
silhouette, subject to no cluster containing fewer than 2% of train emails.
The selected model currently uses **k=5** with train silhouette **0.0143** and
held-out test silhouette **0.0142**. The low score is reported honestly for
noisy Enron text.

Each row in [outputs/segments.csv](outputs/segments.csv) contains an editable
human-readable label, keywords, three sample emails, common greeting,
sign-off, and formality. The held-out 2D plot is
[outputs/cluster_scatter.png](outputs/cluster_scatter.png).

## Profile-to-segment sanity check

The evaluation uses 12 profiles written in AESLC's domain language: energy
trading, gas pipelines, legal contracts, HR, meetings, travel, finance,
settlements, procurement, and plant operations. For every profile the report
records the segment it reaches and the top similarity score among its three
retrieved train emails:

[outputs/profile_baseline.csv](outputs/profile_baseline.csv)

The table also includes the number of distinct segments reached. This is a
qualitative segmentation sanity check, not a held-out accuracy claim.

Topic coverage is intentionally not a headline metric. It remains only as a
sanity check in the same CSV and evaluation report.

## Generation

Recipient topics combine POS-tagged noun phrases, collocations, acronyms, and
train TF-IDF weights. Generic and overlapping topics are removed and the top
three are returned. Generated subjects use retrieved subject patterns plus a
topic. Bodies use first name only, the selected segment's common greeting and
sign-off style, the real sender name `Milan`, and each topic at most once.

## Limitations

AESLC is noisy Enron mail and subject-line token overlap is a strict lexical
metric. Unsupervised clusters can have low silhouette even when useful for
generation. Human-readable segment labels are editable descriptions, not
ground-truth categories. The profile table is a sanity check rather than a
test-set performance estimate.
