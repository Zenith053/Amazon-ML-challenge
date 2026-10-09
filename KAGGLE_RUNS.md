# Kaggle Runs Log: Kernel Squad

_Status as of 2026-09-26, 10:25 IST. All times are IST (Kaggle shows UTC; add 5 h 30 min)._

This file lists **every job we have run on Kaggle**: what code it ran, why, what it produced, and how that output is used later. Every job is a **private** script job in the `peeyushprashant` Kaggle account. They were launched from the local terminal with the Kaggle CLI (`business_entity_resolution/kaggle/push_job.py`); none were run by hand in the Kaggle editor.

---

## 1. Why Kaggle at all

| Problem on the local laptop | What Kaggle gives us |
|---|---|
| 14 GB RAM (about 7 GB usable). The 10M-record pool plus features don't fit; one early run was killed by the memory cap | about 30 GB RAM per session |
| GTX 1650 Ti with 4 GB. It can't fine-tune a transformer cross-encoder | **2× Tesla T4** (16 GB each) GPU sessions |
| Internet at about 70–380 KB/s. Moving GBs of data or scores is impractical | the dataset downloads to Kaggle at about 35 MB/s; only code and small logs cross our link |
| One machine | **5 CPU sessions in parallel**, plus GPU sessions on a separate quota |

Constraints we learned the hard way, and how the tooling handles each:

| Kaggle constraint | How we handle it |
|---|---|
| Uploaded zips are **flattened**: our `ber/` folder arrives as loose files | the job runner rebuilds the `ber` package with a symlink |
| At most **5 CPU sessions at once** | extra jobs are queued and pushed automatically when a slot frees |
| **12-hour limit** per session | long work is split per country and per shard |
| The CLI can't show a running job's log | every job prints timestamped progress with ETA, visible live on its kaggle.com page (from 26 Sep 00:14 onward) |
| Attached inputs are **read-only** | every model version writes its own tagged files (`BER_TAG=v2`, giving `*_v2.parquet`) |

## 2. How a job works (shared by all jobs)

- **Code:** `src/ber/*.py` is uploaded as the private dataset **`ber-code`** by `kaggle/sync_code.sh`. There's a new version per upload; the dataset currently has 13 versions. A job always mounts the latest version at the moment it **starts**.
- **Data:** the organizers' zip is fetched once by the `ber-data` job (below). Every other job attaches its output.
- **Chaining:** a job lists earlier jobs in `--after`. Their `work/` files are linked into its own work directory, so the pipeline's intermediate files flow from job to job without ever being downloaded.
- **Steps:** a job runs `python -m ber.<module> ...` steps in order. On GPU jobs, `$ACCEL` expands to `accelerate launch` with one process per visible GPU.
- **Reading results:** `python kaggle/push_job.py <job> --log` downloads the finished log.

---

## 3. Jobs run so far

### Overview

| # | Job | Machine | Started (IST) | Status | Runtime | Main output |
|---|---|---|---|---|---|---|
| 1 | `ber-data` | CPU | 25 Sep 20:25 | ✅ complete | ~1 min | raw dataset (TSVs, validator, doc template) |
| 2 | `ber-prep` | CPU | 25 Sep 21:08 (4th attempt) | ✅ complete | 14 min | normalized train/test parquet, transliteration map |
| 3 | `ber-rerank` | CPU | 25 Sep 21:24 | ✅ complete | 30 min | `reranker.txt` (BM25 version) + recall report |
| 4 | `ber-rerank-tfidf` | CPU | 25 Sep 21:24 | ✅ complete | 43 min | TF-IDF comparison (design check) |
| 5–8 | `ber-cand-train-us0/us1/in0/in1` | CPU ×4 | 25 Sep 22:13 | ✅ complete | 2.0–4.2 h | train candidates (8 part files) |
| 9–11 | `ber-cand-test-us/in/fr` | CPU ×3 | 22:13 / 00:14 / 01:54 | ✅ complete | 4.8 / 5.5 / 1.1 h | test candidates (7 part files) |
| 12 | `ber-ce-smoke` | **GPU 2× T4** | 26 Sep 10:02 (2nd attempt) | ✅ complete | ~10 min | merged candidate files; cross-encoder verified |
| 13 | `ber-v1` | CPU | 26 Sep 10:04 | ⏳ running | est. 2–3 h | **v1 predictions** (first submission) |
| 14 | `ber-ce-train` | **GPU 2× T4** | 26 Sep 10:14 | ⏳ running | est. 1–1.5 h | fine-tuned cross-encoder model |

---

### 1. `ber-data`: fetch the dataset
- **Ran:** `kaggle/kernels/ber-data/ber_data.py`. It downloads the organizers' zip from the Google Drive link with `gdown` (1.09 GB in ~30 s) and extracts `dataset/`, `utils/validate_submission.py` and `Documentation_template.md`.
- **Why:** our local connection couldn't upload 2.4 GB of TSVs in reasonable time, so Kaggle fetches the data itself.
- **Future use:** attached as the data input of **every** other job, including the final packaging job. It never needs to run again.

### 2. `ber-prep`: normalize all data
- **Ran:** `ber.prepare --splits train test`, which uses `io.py`, `normalize.py`, `translit.py` and `prepare.py`.
- **What it does:**
  - normalizes all 22M names and addresses: transliteration to ASCII, legal suffixes, abbreviations, numbers, city/region names
  - learns the **transliteration map** from train pairs
  - re-normalizes everything with that map
  - maps the ground truth to record indexes
- **Why:** every later stage (blocking, n-grams, features, matcher) works on these normalized fields.
- **Result:** 14 min. It learned **14,594** transliteration mappings, identical to the local run, so the step is reproducible. Output is 4.6 GB.
- **Attempts:** versions 1–3 failed at startup because our code wasn't found. The cause was that Kaggle flattens zipped datasets. Fixing that is what made every later job work.
- **Future use:** input to every pipeline job, and to the final package job.

### 3. `ber-rerank`: train the candidate re-ranker (BM25)
- **Ran:** `ber.rerank_dev --weighting bm25`, which uses `blocking.py`, `ngram.py`, `features.py` and `candidates.py`.
- **What it does:**
  - builds the rare-key index and the **BM25 character n-gram index**
  - takes union candidates for 60k train S1 records (40k to train, 20k to evaluate)
  - computes 43 features and trains the LightGBM re-ranker
- **Why:** blocking alone ranks true matches poorly. The re-ranker lets us keep only the **top 20** candidates per S1 while keeping almost all true matches.
- **Result:**

  | | Recall |
  |---|---|
  | Key index alone (top 200) | 97.0% |
  | Union with n-gram BM25 | **98.5%** (ceiling) |
  | After re-ranking, top 10 / top 20 | **97.8% / 97.9%** |
- **Future use:** its `reranker.txt` was used by all 7 candidate jobs. Its train keys and IDF tables were reused. Its dev feature files are available for analysis.

### 4. `ber-rerank-tfidf`: design check, TF-IDF vs BM25
- **Ran:** the same as job 3, with `--weighting tfidf`.
- **Why:** an open design question: is BM25 better than TF-IDF cosine for n-gram retrieval?
- **Result:** BM25 won. Its n-gram-only recall was 96.4% vs 95.3%, its own score ranks better, and it was **about 30% faster**. After re-ranking the two tie. **Decision: BM25.** Details are in PROGRESS.md §3a.
- **Future use:** evidence for the methodology document; no later job depends on it.

### 5–11. `ber-cand-*`: full-scale candidate generation (7 jobs)
- **Ran:** `ber.candidates --splits <train|test> --countries <US|India|France> [--shard i/n]`
- **What it does:** for **every** S1 record, it takes the union of key-index and n-gram candidates (~241 per S1), computes 43 features, scores them with the re-ranker, and keeps the **top 20**.

  | Job | S1 records | Part files | Runtime |
  |---|---|---|---|
  | train-us0 (shard 0/2) | 724k | `train_cand_part_US_000, 002, 004` | 4.2 h |
  | train-us1 (shard 1/2) | 600k | `US_001, 003` | 3.8 h |
  | train-in0 | 583k | `India_000, 002` | 3.6 h |
  | train-in1 | 300k | `India_001` | 2.0 h |
  | test-us | 663k | `test_cand_part_US_000–002` | 4.8 h |
  | test-in | 810k | `India_000–002` | 5.5 h |
  | test-fr | 259k | `France_000` | 1.1 h |
- **Why split into 7:** Kaggle's 12 h session limit and 5-session concurrency. Test India and France were queued and pushed automatically when slots freed.
- **Why train *and* test:**
  - The **train** candidates are what the matcher learns from, and where we measure F0.5 at full scale. The competition between S1 records only appears when all 2.2M are present.
  - The **test** candidates become `candidate_pairs.tsv` and are what the matcher predicts on.
- **Result:** all exited 0. After merging: **train 44.1M pairs, test 34.65M pairs** (20 per S1).
- **Measured speed:** ~44 S1 records/s per job. Most of the time goes into computing features for ~251 candidates per S1, of which only 20 are kept (see PROGRESS.md for a possible two-stage speed-up).
- **Future use:** the input of everything downstream (context features, matcher, cross-encoder, final outputs).

### 12. `ber-ce-smoke`: cross-encoder smoke test (GPU)
- **Ran:**
  1. `ber.candidates --merge`: merges the 15 part files into `train_cand.parquet` and `test_cand.parquet`
  2. `ber.cross_encoder train --limit-q 3000`
  3. `ber.cross_encoder score --split test --limit-q 3000`
- **Why:** the cross-encoder code had never run. A tiny run on the real data catches bugs cheaply and **measures speed**, so the full run can be sized.
- **Attempts:**
  - v1 failed: `transformers` loads mDeBERTa's weights in fp16, but mixed-precision training needs fp32 master weights. Fixed by casting the model to fp32.
  - v2 passed.
- **Result:**
  - 2× T4 detected
  - training loss 0.68 → 0.09 in 563 steps
  - **training 136 pairs/s, scoring ~1,230 pairs/s**
  - scores written correctly
  - The model-loading report (UNEXPECTED / MISSING keys) is normal: pretraining heads dropped, a new classification head created.
- **What we learned:** at those speeds a naive full run would take ~10.6 GPU-hours. This led to two optimizations: **frozen word embeddings** (training) and **gated scoring**, where only pairs v1 is unsure about get scored.
- **Future use:** its output (`train_cand.parquet`, `test_cand.parquet`) is attached by `ber-ce-train` and later jobs, so the merge isn't repeated.

### 13. `ber-v1`: first full predictions (running)
- **Runs:**
  1. merge
  2. `ber.matcher ctx --split train` and `--split test`: context features (query context, **competition** between S1 records, **cluster consistency**)
  3. `ber.matcher oof`: 2-fold out-of-fold LightGBM matcher on all train S1 records; picks the best decision rule by macro F0.5
  4. `ber.matcher final`: trains on 300k S1, predicts all test S1, writes `matching_results.tsv` + `candidate_pairs.tsv`, runs the **official validator with ID checks**
  5. zips the outputs
- **Why:** gives the **first leaderboard submission** and the **real full-scale F0.5**, and it's the baseline every later version must beat.
- **Future use:**
  - `train_oof.parquet` and `test_probs.parquet` (v1 probabilities) are the **gate** that decides which pairs the cross-encoder scores
  - `train_ctx.parquet` and `test_ctx.parquet` are reused by v2
  - its `decision.json` provides the v1 comparison
  - the zipped outputs are **submission v1**

### 14. `ber-ce-train`: full cross-encoder training (running, GPU)
- **Runs:** `ber.cross_encoder train --limit-q 150000 --neg-per-q 8 --bs 64` on 2× T4, with frozen word embeddings.
- **Training data:** 150k S1 records from the reserved 10% cross-encoder sample (never used to train or evaluate the matcher), with all their true matches plus the hardest non-matches from our own candidates.
- **Why:** a cross-encoder reads both records *together*, using the raw text (Devanagari, Tamil, French accents included), so it can judge matches that string-similarity features can't. We expect it to be the biggest accuracy gain, especially for native-script names and France.
- **Future use:** the fine-tuned model (`ce_model/`) is used by `ber-ce-score`.

---

## 4. Planned next jobs

| Job | Machine | Runs | Needs | Why |
|---|---|---|---|---|
| `ber-ce-score` | GPU | `ber.cross_encoder score --gate <v1 probs> --band 0.02 0.98` for all test S1 (top 10) and a fixed 30% train sample | `ber-v1`, `ber-ce-train` | scores only the pairs v1 is unsure about, so GPU time goes where decisions are actually made |
| `ber-v2` | CPU | `BER_TAG=v2 ber.matcher oof` + `final` | `ber-ce-score`, `ber-v1` | matcher with cross-encoder features. Prints **v1 vs v2 F0.5 on the same S1 records**; if v2 wins it becomes the submission |
| `ber-package` | CPU | builds `Kernel_Squad_submission.zip` (outputs + code + README + requirements + filled documentation), validates, compresses | final version | the final deliverable, downloaded as one compressed file |

## 5. How this helps us going forward

- **Reproducibility for the judges:** every output can be regenerated from the organizers' data by re-running this same chain of jobs. It's the exact procedure the README documents.
- **No redundant computation:** normalization, candidates and context features are computed once and reused by every later model version. A new matcher version costs minutes, not the ~5 h of candidate generation.
- **Fair comparisons:** each version writes tagged outputs and reports its F0.5 against v1 on identical S1 records, so we only submit real improvements.
- **Safety margin before the deadline (27 Sep, 23:59 IST):** the v1 outputs are a validated fallback submission that exists whatever happens to the cross-encoder.
