# Amazon ML Challenge 2026: Business Entity Resolution (Kernel Squad)

For every Source 1 business record, find the matching records in Source 2 and Source 3. Training data covers the US and India. The test set also contains France, which never appears in training. Submissions are scored by macro F0.5.

**Final submission (v4): out-of-fold macro F0.5 0.9863, leaderboard 0.983.**

## Pipeline

```
raw TSV → normalisation + transliteration map learned from train pairs
        → candidates: IDF-weighted rare-key inverted index ∪ character 3-gram BM25
        → 43 pair features → LightGBM re-ranker → top 20 per S1        (candidate_pairs.tsv)
        → context features (competition between S1s, clusters, twins)
        → fine-tuned cross-encoders: mDeBERTa-v3-base (1 + 2 epochs), CamemBERT-base
        → stage-1 LightGBM matcher (out-of-fold) → stage-2 LightGBM stacker
        → expected-F0.5 set selection + one-to-one assignment → matching_results.tsv
```

## Results

| Version | Change | OOF F0.5 | Leaderboard |
|---|---|---|---|
| v1 | blocking + features + LightGBM matcher | 0.9731 | 0.950 |
| v2 | + mDeBERTa cross-encoder | 0.9854 | 0.9803 |
| v3full | + mDeBERTa epoch 2 + CamemBERT (France pseudo-labels) | 0.9858 | 0.9812 |
| **v4** | + stage-2 stacker (confirmed-match clusters, competition) | **0.9863** | **0.983** |

## Repository layout

```
business_entity_resolution/   the pipeline: code, requirements and full reproduction steps
  README.md                   how to reproduce the submission end to end
  src/ber/                    pipeline package (python -m ber.<module>)
  kaggle/                     tooling that ran every stage as Kaggle jobs
PROGRESS.md                   development log and design notes
KAGGLE_RUNS.md                record of the Kaggle jobs and their results
Documents/                    links (dataset, submissions)
run.sh                        runs local jobs under a memory cap
```

Data, intermediate files (`work/`), outputs and submission packages are not in the repository. See [business_entity_resolution/README.md](business_entity_resolution/README.md) for setup and reproduction.

## Constraints we followed

- No external data. All statistics, maps and labels come from the provided data.
- Pretrained models are MIT licensed and have at most 8B parameters: `microsoft/mdeberta-v3-base` and `almanach/camembert-base`.

## Team

Kernel Squad: Shobhan Parida (leader), Shubham Paliwal, Peeyush Prashant, Rahul Saha.
