# AESLC Evaluation Report

## Held-out evaluation

- Train index: **13330** emails.
- Test index: **1629** emails.
- KMeans silhouette on held-out test assignments: **0.01417388268305531**.
- The clustering model and all vectorizers/embeddings were fit on train only.

## Retrieval subject-line token-overlap F1

| method | top1_subject_f1 | top3_subject_f1 |
| --- | --- | --- |
| random | 0.005662983425414364 | 0.0054689993861264585 |
| tfidf | 0.0980853898096992 | 0.07722516881522407 |
| word2vec | 0.06211639042357275 | 0.05075174953959484 |


The random row is a retrieval baseline. `top3_subject_f1` averages F1 against each of the three retrieved subjects.

## Profile-to-segment sanity check

The 12 Enron-style profiles below are a varied qualitative sanity check. Each
row reports the predicted segment and the highest similarity score among the
three retrieved train emails. The full table is in `outputs/profile_baseline.csv`.

| name | segment_id | segment_label | top_similarity_score |
| --- | --- | --- | --- |
| Ava | 3 | Segment 3: gas, deal, power | 0.8178 |
| Ben | 3 | Segment 3: gas, deal, power | 0.8423 |
| Cara | 2 | Segment 2: agreement, master, attached | 0.6373 |
| Dev | 2 | Segment 2: agreement, master, attached | 0.5796 |
| Eli | 4 | Segment 4: thanks, meeting, information | 0.671 |
| Faye | 4 | Segment 4: thanks, meeting, information | 0.668 |
| Gus | 4 | Segment 4: thanks, meeting, information | 0.6423 |
| Hana | 4 | Segment 4: thanks, meeting, information | 0.7458 |
| Ivan | 3 | Segment 3: gas, deal, power | 0.787 |
| Jules | 4 | Segment 4: thanks, meeting, information | 0.6956 |
| Kira | 4 | Segment 4: thanks, meeting, information | 0.6128 |
| Luca | 4 | Segment 4: thanks, meeting, information | 0.6038 |


Distinct segments reached: **3** of **5**.

Topic coverage is retained only as a sanity check: personalized topic mentions
mean **3.00** per profile versus
**0.00** for the segment-only baseline.

## Limitations

AESLC is noisy Enron mail and subject-line token overlap is a strict lexical metric. Clusters are unsupervised and can have low silhouette even when useful to generation. Human-readable segment labels in `outputs/segments.csv` are editable and are not ground-truth categories.
