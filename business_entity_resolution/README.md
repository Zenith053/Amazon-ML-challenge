# Business Entity Resolution: Kernel Squad

This pipeline produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the challenge data: for every Source 1 (S1) record, the matching Source 2 / Source 3 records.

Final submission = **v4**: out-of-fold macro F0.5 0.9863, leaderboard 0.983.

```
raw TSV → prepare (normalise + transliteration map learned from train)
        → candidates: rare-key inverted index (top 200) ∪ character 3-gram BM25 (top 50 per field)
        → 43 pair features → LightGBM re-ranker → top 20 per S1        (= candidate_pairs.tsv)
        → context features (query context, competition between S1s, cluster, twin)
        → fine-tuned cross-encoders: mDeBERTa-v3-base (1 + 2 epochs), CamemBERT-base
        → stage-1 LightGBM matcher (2-fold out-of-fold)
        → stage-2 LightGBM stacker (stage-1 probability + confirmed-match cluster + competition)
        → expected-F0.5 set selection → matching_results.tsv → official validator
```

The methodology is described in `Documentation_template.md` at the root of the submission zip.

## Layout

```
src/ber/              pipeline package (run with: python -m ber.<module>)
  io.py               TSV loading/caching; paths via BER_DATA / BER_WORK / BER_OUTPUT; version tags (BER_TAG)
  normalize.py        name / address normalisation (vectorised polars)
  translit.py         transliteration dictionary learned from train pairs only
  prepare.py          normalise all splits → work/*.parquet (+ ground-truth pairs)
  blocking.py         candidate channel 1: IDF-weighted rare-key inverted index
  ngram.py            candidate channel 2: BM25 over character n-grams (TF-IDF option)
  features.py         43 pair features
  candidates.py       union → features → re-ranker → top-20 per S1, per country / shard, --merge
  rerank_dev.py       trains the re-ranker (work/reranker.txt), reports channel recall
  context.py          context, competition, cluster and twin features
  cross_encoder.py    train / validate / score cross-encoders (accelerate, multi-GPU)
  matcher.py          ctx | oof | final: stage-1 LightGBM matcher and decision rule
  stage2.py           oof | fit | decide: stage-2 stacker → final outputs + validation
  decide.py           one-to-one assignment, expected-F0.5 set selection
  metric.py           macro F0.5, identical to the official definition
  submit.py           writes both TSVs, runs utils/validate_submission.py --check-ids
  log.py              timestamped progress logging
  analysis.py, diag.py, france_audit.py, variants.py, eval_blocking.py, f05_dev.py
                      diagnostics used during development (not needed for the outputs)
kaggle/               tooling that ran every stage as Kaggle jobs from a local terminal
requirements.txt
```

## Environment

- **Python 3.12:** `pip install -r requirements.txt`. torch must be a CUDA build for the GPU stages.
- **Hardware used:**
  - Kaggle CPU sessions (4 vCPU, ~30 GB RAM) for all CPU stages.
  - Kaggle GPU sessions (2× NVIDIA T4, 16 GB each) for the cross-encoders.
- **No external data.** The inputs are the challenge TSVs and the public pretrained weights of `microsoft/mdeberta-v3-base` and `almanach/camembert-base` (both MIT, downloaded from Hugging Face). All statistics, maps and labels come from the provided data.

## Reproduce end to end

```bash
export PYTHONPATH=src
export BER_DATA=/path/to/student_resource/dataset   # contains train/ and test/ (utils/ next to it)
export BER_WORK=/path/to/work                        # intermediate files (~25 GB)
export BER_OUTPUT=/path/to/output
ACCEL="accelerate launch --mixed_precision fp16 --multi_gpu --num_processes 2 -m"   # or --num_processes 1

# 1. normalisation (learns, then applies, the transliteration map)
python -m ber.prepare --splits train test

# 2. re-ranker (40k train S1; writes work/reranker.txt)
python -m ber.rerank_dev --weighting bm25

# 3. candidates: key index ∪ BM25 → re-ranker → top 20 per S1 (resumable; --countries / --shard to split)
python -m ber.candidates --splits train test

# 4. cross-encoders (GPU). The first model trains on a fixed 10% of train S1 that the matcher never uses.
$ACCEL ber.cross_encoder train --limit-q 150000 --keep-k 20 --neg-per-q 8 --bs 64 --epochs 1
python -m ber.cross_encoder validate --keep-k 10 --bs 256
$ACCEL ber.cross_encoder score --split train --keep-k 10 --bs 256 --share 0.3
$ACCEL ber.cross_encoder score --split test  --keep-k 10 --bs 256
#    epoch 2 (resume, recall-weighted)
$ACCEL ber.cross_encoder train --name m2 --base ce_model --train-set all-but-val --neg-per-q 12 \
       --hard-pos-weight 2 --lr 1.5e-5 --bs 64 --epochs 1
$ACCEL ber.cross_encoder score --name m2 --split train --keep-k 10 --bs 256 --share 0.3
$ACCEL ber.cross_encoder score --name m2 --split test  --keep-k 10 --bs 256

# 5. stage-1 matcher v2 (one cross-encoder); its test probabilities supply CamemBERT's France pseudo-labels
python -m ber.matcher ctx --split train          # v1 context files (reference model)
BER_CE_NAMES=none python -m ber.matcher oof      # v1 baseline without cross-encoders, needed as reference
BER_TAG=v2 python -m ber.matcher ctx --split train
BER_TAG=v2 python -m ber.matcher ctx --split test
BER_TAG=v2 python -m ber.matcher oof --n-train-q 600000 --rounds 600
BER_TAG=v2 python -m ber.matcher final --n-train-q 600000 --rounds 600

# 6. CamemBERT: Latin-script train pairs + France pseudo-labels from v2's test probabilities
$ACCEL ber.cross_encoder train --name cb --base almanach/camembert-base --train-set all-but-val --latin-only \
       --neg-per-q 12 --no-freeze-emb --lr 3e-5 --bs 64 --epochs 1 \
       --pseudo test_probs_v2.parquet --pseudo-country France --pseudo-q 50000
$ACCEL ber.cross_encoder score --name cb --split train --keep-k 10 --bs 256 --share 0.3
$ACCEL ber.cross_encoder score --name cb --split test  --keep-k 10 --bs 256

# 7. stage-1 matcher v3full (three cross-encoders)
export BER_CTX_TAG=v2 BER_CE_NAMES=,m2,cb
BER_TAG=v3full BER_BASELINE=v2 python -m ber.matcher oof --n-train-q 600000 --rounds 600
BER_TAG=v3full python -m ber.matcher final --n-train-q 600000 --rounds 600

# 8. stage-2 stacker v4 = final submission
export BER_TAG=v4 BER_STAGE1=v3full
python -m ber.stage2 oof --seeds 1 2            # out-of-fold F0.5, decision rule → work/decision_v4.json
python -m ber.stage2 fit --seeds 1 2            # test probabilities → work/test_probs_v4.parquet
python -m ber.stage2 decide                     # → $BER_OUTPUT/v4/{matching_results,candidate_pairs}.tsv + validation
```

`python -m ber.matcher oof` (step 5) needs the v1 context files from `python -m ber.matcher ctx` without a tag. The v1 out-of-fold probabilities supply the competition context for train queries outside the cross-encoder-scored sample.

## How our outputs were actually produced (Kaggle)

Every stage ran as a private Kaggle script job, pushed from a local terminal with the Kaggle CLI. The code was uploaded as a private dataset and outputs flowed between jobs:

```bash
kaggle/sync_code.sh "msg"                                  # upload src/ber as dataset <user>/ber-code
kaggle kernels push -p kaggle/kernels/ber-data             # downloads the challenge zip once
python kaggle/push_job.py <job> --after <earlier jobs> --steps "<module args>" [--gpu]
python kaggle/push_job.py <job> --log                      # read a finished job's log
```

The full job log (what ran, why, results) is in the methodology document.

Kaggle constraints the tooling handles:
- 5 concurrent CPU sessions
- a 12 h session limit (candidates were split per country and shard)
- flattened dataset zips
- read-only inputs (versioned output names via `BER_TAG`)

## Runtime (Kaggle)

| Stage | Hardware | Time |
|---|---|---|
| prepare (train + test) | CPU | 14 min |
| re-ranker | CPU | 30 min |
| candidates (7 jobs, per country / shard) | CPU ×5 | ~2–5.5 h each |
| cross-encoder training (mDeBERTa / epoch 2 / CamemBERT) | 2× T4 | 1.8 h / 3.0 h / 2.6 h |
| cross-encoder scoring, test top-10 (17.3M pairs) | 2× T4 | ~2.2–3.4 h per model |
| context features (train + test) | CPU | ~30 min |
| matcher out-of-fold + final | CPU | ~1.2 h |
| stage-2 out-of-fold / fit / decide | CPU | 30 / 55 / 5 min |
