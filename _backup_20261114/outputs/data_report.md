# AESLC Data Report

## Parsing

- Parsed `.subject` files: **18302**
- Deduplicated cleaned emails: **16746**
- Each record is parsed as body text before the final `@subject` marker and a subject line after it.
- Original directory split is preserved; no random split is used.

## Split Counts

| Split | Emails |
| --- | ---: |
| train | 13330 |
| dev | 1787 |
| test | 1629 |

## Cleaning

- Removed forwarded/original-message headers, quoted replies, disclaimers, URLs, email addresses, and phone numbers.
- Added `enron`, `ect`, `hou`, and `forwarded` to stopwords.
- Deduplication occurs before model fitting and preserves the first original split assignment.
