"""Cross-encoder pair scorer (GPU). Fine-tunes microsoft/mdeberta-v3-base (MIT) on
(S1 record, candidate record) pairs built from our own blocking candidates, so the
negatives are exactly the hard cases the matcher faces. Raw (un-normalised) text is
used: the multilingual tokenizer reads Devanagari/Tamil/... and accented French directly.

  train : fine-tune on candidate pairs of the `ce` query subset of train
  score : add `ce_logit` to work/{split}_cand.parquet pairs (all queries not used for
          training get honest scores; training queries are flagged)

Multi-GPU: launch with `accelerate launch --multi_gpu --num_processes 2 -m ber.cross_encoder ...`.
"""
import argparse
import math
import os
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

import json

from ber.io import WORK, in_score_share, is_ce_query
from ber.log import Progress, get

log = get("ce")

MODEL = os.environ.get("BER_CE_MODEL", "microsoft/mdeberta-v3-base")
MAX_LEN = 96
VAL_Q, BASE_TRAIN_Q = 20_000, 150_000  # fixed held-out validation set / first model's training sample


def ce_dir(name: str):
    return WORK / (f"ce_model_{name}" if name else "ce_model")


def sfx(name: str) -> str:
    return f"_{name}" if name else ""


def reserved_split(reserved_q: pl.Series) -> tuple[pl.Series, pl.Series]:
    """(training pool, validation set) of the reserved cross-encoder S1 records. The validation
    set is the one the first model was validated on: 20k drawn (seed 2) from the reserved
    records outside its 150k training sample (seed 1), so every model is compared on it."""
    q = reserved_q.unique().sort()
    used = q.sample(min(BASE_TRAIN_Q, len(q)), seed=1)
    val = q.filter(~q.is_in(used.implode())).sort().sample(VAL_Q, seed=2)
    return q.filter(~q.is_in(val.implode())), val


def record_text(split: str, which: str) -> pl.DataFrame:
    path = WORK / (f"{split}_s1.parquet" if which == "s1" else f"{split}_pool.parquet")
    return pl.scan_parquet(path).select(
        "idx",
        "entity_id",
        text=pl.concat_str(
            [pl.col("business_name").fill_null(""), pl.col("business_address").fill_null("")], separator=" | "
        ),
    ).collect()


def pair_frame(split: str, keep_k: int, only_ce: bool | None, limit_q: int = 0) -> pl.DataFrame:
    c = pl.scan_parquet(WORK / f"{split}_cand.parquet").filter(pl.col("rr_rank") <= keep_k).select(
        "q_idx", "p_idx", "rr_rank"
    ).collect()
    q = record_text(split, "s1")
    c = c.join(q.rename({"idx": "q_idx", "entity_id": "s1_id", "text": "q_text"}), on="q_idx")
    if only_ce is not None:
        c = c.filter(is_ce_query(pl.col("s1_id")) if only_ce else ~is_ce_query(pl.col("s1_id")))
    if limit_q:
        keep = c["q_idx"].unique().sort().sample(min(limit_q, c["q_idx"].n_unique()), seed=1)
        c = c.filter(pl.col("q_idx").is_in(keep.implode()))
    p = record_text(split, "pool")
    return c.join(p.rename({"idx": "p_idx", "entity_id": "cand_id", "text": "p_text"}), on="p_idx")


class Pairs(Dataset):
    def __init__(self, a: list, b: list, y: np.ndarray | None, w: np.ndarray | None = None):
        self.a, self.b, self.y, self.w = a, b, y, w

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return (self.a[i], self.b[i], (self.y[i] if self.y is not None else 0.0),
                (self.w[i] if self.w is not None else 1.0))


def _collate(tok):
    def f(batch):
        a, b, y, w = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        enc["weights"] = torch.tensor(w, dtype=torch.float32)
        return enc
    return f


INDIC = r"[\u0900-\u0DFF]"  # Devanagari ... Sinhala: scripts a French/Latin tokenizer cannot read


def pseudo_pairs(probs_file: str, country: str, max_q: int, hi: float, lo: float, keep_k: int) -> pl.DataFrame:
    """Confident predictions of an earlier model on TEST pairs of one country, as labels
    (self-training for a country without training labels). Only p > hi (1) and p < lo (0)."""
    s1 = pl.read_parquet(WORK / "test_s1.parquet", columns=["idx", "country"]).filter(pl.col("country") == country)
    pr = (pl.read_parquet(WORK / probs_file, columns=["q_idx", "p_idx", "prob"])
          .filter(pl.col("q_idx").is_in(s1["idx"].implode())))
    rk = pl.scan_parquet(WORK / "test_cand.parquet").select("q_idx", "p_idx", "rr_rank").filter(
        pl.col("rr_rank") <= keep_k).collect()
    pr = pr.join(rk, on=["q_idx", "p_idx"])
    qs = pr["q_idx"].unique().sort()
    qs = qs.sample(min(max_q, len(qs)), seed=3)
    pr = pr.filter(pl.col("q_idx").is_in(qs.implode()) & ((pl.col("prob") > hi) | (pl.col("prob") < lo)))
    q = record_text("test", "s1").rename({"idx": "q_idx", "entity_id": "s1_id", "text": "q_text"})
    p = record_text("test", "pool").rename({"idx": "p_idx", "entity_id": "cand_id", "text": "p_text"})
    return (pr.join(q, on="q_idx").join(p, on="p_idx")
            .with_columns(label=(pl.col("prob") > hi).cast(pl.Float32)).drop("prob"))


def train(keep_k: int, neg_per_q: int, epochs: float, lr: float, bs: int, limit_q: int, freeze_emb: bool,
          name: str, base: str, train_set: str, latin_only: bool = False, hard_pos_weight: float = 1.0,
          pseudo: str = "", pseudo_country: str = "France", pseudo_q: int = 50_000,
          pseudo_hi: float = 0.99, pseudo_lo: float = 0.01) -> None:
    from accelerate import Accelerator
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    acc = Accelerator(mixed_precision="fp16")
    if train_set == "all-but-val":
        df = pair_frame("train", keep_k, only_ce=True)
        pool, _ = reserved_split(df["q_idx"])
        df = df.filter(pl.col("q_idx").is_in(pool.implode()))
    else:
        df = pair_frame("train", keep_k, only_ce=True, limit_q=limit_q)
    base_path = WORK / base if (WORK / base).exists() else None
    src = str(base_path) if base_path else base
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet").with_columns(label=pl.lit(1.0, pl.Float32))
    df = df.join(gt, on=["q_idx", "p_idx"], how="left").with_columns(pl.col("label").fill_null(0.0))
    # all positives + the hardest negatives (by re-ranker rank) per query
    df = df.filter((pl.col("label") == 1) | (pl.col("rr_rank") <= neg_per_q + 4))
    if latin_only:
        n0 = df.height
        df = df.filter(~pl.col("q_text").str.contains(INDIC) & ~pl.col("p_text").str.contains(INDIC))
        if acc.is_main_process:
            log.info(f"latin-only: kept {df.height:,} of {n0:,} pairs")
    # recall focus: up-weight true pairs whose candidate has no address or a native-script name
    hard = (pl.col("label") == 1) & (pl.col("p_text").str.ends_with("| ") | pl.col("p_text").str.contains(INDIC))
    df = df.with_columns(weight=pl.when(hard).then(pl.lit(hard_pos_weight)).otherwise(pl.lit(1.0)).cast(pl.Float32))
    df = df.select("q_idx", "p_idx", "q_text", "p_text", "label", "weight")
    if pseudo:
        ps = pseudo_pairs(pseudo, pseudo_country, pseudo_q, pseudo_hi, pseudo_lo, keep_k).with_columns(
            weight=pl.lit(1.0, pl.Float32)).select(df.columns)
        if acc.is_main_process:
            log.info(f"pseudo-labels ({pseudo_country}, from {pseudo}): {ps.height:,} pairs from "
                     f"{ps['q_idx'].n_unique():,} S1, positives {ps['label'].mean():.3f}")
        df = pl.concat([df, ps])
    df = df.sample(fraction=1.0, shuffle=True, seed=0)
    if acc.is_main_process:
        log.info(f"train pairs {df.height:,} from {df['q_idx'].n_unique():,} S1; positives {df['label'].mean():.3f}; "
                 f"up-weighted hard positives {(df['weight'] > 1).sum():,}; "
                 f"GPUs {acc.num_processes}; init {src}; saving as '{name or 'ce_model'}'")

    tok = AutoTokenizer.from_pretrained(src)
    # checkpoints may be stored in fp16; AMP needs fp32 master weights
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1).float()
    if freeze_emb:
        # the 250k-token multilingual embedding matrix is ~2/3 of all parameters; freezing it
        # makes each optimizer step far cheaper with little effect on fine-tuning quality
        model.get_input_embeddings().weight.requires_grad_(False)
    if acc.is_main_process:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        smp = df.sample(min(2000, df.height), seed=0)
        lens = [len(x) for x in tok(smp["q_text"].to_list(), smp["p_text"].to_list())["input_ids"]]
        log.info(f"trainable params {n_tr / 1e6:.0f}M of {n_all / 1e6:.0f}M; pair length median "
                 f"{int(np.median(lens))} tokens, {100 * np.mean(np.array(lens) > MAX_LEN):.1f}% longer than {MAX_LEN}")
    dl = DataLoader(Pairs(df["q_text"].to_list(), df["p_text"].to_list(), df["label"].to_numpy(),
                          df["weight"].to_numpy()),
                    batch_size=bs, shuffle=True, collate_fn=_collate(tok), num_workers=2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = math.ceil(len(dl) * epochs / acc.num_processes)
    # accelerate steps a prepared scheduler once per process, so size it accordingly
    total = steps * acc.num_processes
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    model, opt, dl, sched = acc.prepare(model, opt, dl, sched)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    step, t = 0, time.time()
    prog = Progress(log, "ce train (steps)", steps) if acc.is_main_process else None
    run_loss = 0.0
    while step < steps:
        for batch in dl:
            y, w = batch.pop("labels"), batch.pop("weights")
            loss = (lossf(model(**batch).logits.squeeze(-1), y) * w).sum() / w.sum()
            acc.backward(loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            run_loss = 0.98 * run_loss + 0.02 * loss.item() if step > 1 else loss.item()
            if prog:
                prog.step(1, loss=f"{run_loss:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}",
                          pairs_per_s=f"{step * bs * acc.num_processes / (time.time() - t):.0f}")
            if step >= steps:
                break
    acc.wait_for_everyone()
    if acc.is_main_process:
        out = ce_dir(name)
        acc.unwrap_model(model).save_pretrained(out)
        tok.save_pretrained(out)
        log.info(f"saved model to {out} after {step} steps ({time.time() - t:.0f}s)")


@torch.no_grad()
def validate(keep_k: int, bs: int, name: str) -> None:
    """Held-out check on reserved cross-encoder S1 records that were NOT used for training
    (single GPU): log loss, AUC overall and on the pairs v1 was unsure about, P/R at 0.5."""
    from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    reserved = pair_frame("train", 20, only_ce=True)
    _, val = reserved_split(reserved["q_idx"])
    df = reserved.filter(pl.col("q_idx").is_in(val.implode()) & (pl.col("rr_rank") <= keep_k))
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet").with_columns(label=pl.lit(1, pl.Int8))
    df = df.join(gt, on=["q_idx", "p_idx"], how="left").with_columns(pl.col("label").fill_null(0))
    log.info(f"held-out: {df.height:,} pairs from {df['q_idx'].n_unique():,} reserved S1 never used in training "
             f"(positives {df['label'].mean():.3f})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ce_dir(name))
    model = AutoModelForSequenceClassification.from_pretrained(ce_dir(name)).to(device).eval().half()
    df = df.with_columns(L=pl.col("q_text").str.len_chars() + pl.col("p_text").str.len_chars()).sort("L")
    dl = DataLoader(Pairs(df["q_text"].to_list(), df["p_text"].to_list(), None),
                    batch_size=bs, shuffle=False, collate_fn=_collate(tok), num_workers=2)
    out = []
    for batch in dl:
        batch.pop("labels"), batch.pop("weights")
        out.append(model(**{k: v.to(device) for k, v in batch.items()}).logits.squeeze(-1).float().cpu().numpy())
    logit = np.nan_to_num(np.concatenate(out), nan=-20.0, posinf=20.0, neginf=-20.0)
    p = 1 / (1 + np.exp(-logit))
    y = df["label"].to_numpy()
    res = {
        "pairs": int(len(y)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "auc": float(roc_auc_score(y, p)),
        "precision@0.5": float(precision_score(y, p >= 0.5)),
        "recall@0.5": float(recall_score(y, p >= 0.5)),
    }
    oof = WORK / "train_oof.parquet"
    if oof.exists():
        v1 = df.select("q_idx", "p_idx").with_columns(ce=pl.Series(p), y=pl.Series(y)).join(
            pl.read_parquet(oof), on=["q_idx", "p_idx"], how="inner")
        band = v1.filter(pl.col("prob").is_between(0.02, 0.98))
        if band["y"].n_unique() == 2:
            res["band_pairs"] = band.height
            res["band_auc_ce"] = float(roc_auc_score(band["y"], band["ce"]))
            res["band_auc_v1"] = float(roc_auc_score(band["y"], band["prob"]))
        res["auc_v1_all"] = float(roc_auc_score(v1["y"], v1["prob"]))
    for k, v in res.items():
        log.info(f"VALIDATION[{name or 'ce'}] {k}: {v:.4f}" if isinstance(v, float) else f"VALIDATION[{name or 'ce'}] {k}: {v:,}")
    (WORK / f"ce_validation{sfx(name)}.json").write_text(json.dumps(res, indent=2))
    df.select("q_idx", "p_idx").with_columns(ce_logit=pl.Series(logit.astype(np.float32))).write_parquet(
        WORK / f"ce_val_scores{sfx(name)}.parquet")


@torch.no_grad()
def score(split: str, keep_k: int, bs: int, share: float, limit_q: int, gate: str, band: tuple[float, float],
          name: str) -> None:
    from accelerate import Accelerator
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    acc = Accelerator(mixed_precision="fp16")
    df = pair_frame(split, keep_k, only_ce=False if split == "train" else None, limit_q=limit_q)
    if share < 1.0:
        df = df.filter(in_score_share(pl.col("s1_id"), share))
    n_before = df.height
    if gate:
        # only pairs the v1 matcher is unsure about; confident pairs gain nothing from the cross-encoder
        g = pl.read_parquet(WORK / gate, columns=["q_idx", "p_idx", "prob"])
        df = df.join(g, on=["q_idx", "p_idx"], how="inner").filter(pl.col("prob").is_between(*band)).drop("prob")
        if acc.is_main_process:
            log.info(f"gate {gate} band {band}: {df.height:,} of {n_before:,} pairs kept ({df.height / max(n_before, 1):.1%})")
    df = df.sort(["q_idx", "p_idx"])
    if acc.is_main_process:
        log.info(f"scoring {split}: {df.height:,} pairs from {df['q_idx'].n_unique():,} S1 on {acc.num_processes} GPU(s)")
    shard = df.with_row_index("r").filter(pl.col("r") % acc.num_processes == acc.process_index)
    # sort by length within the shard to minimise padding
    shard = shard.with_columns(L=pl.col("q_text").str.len_chars() + pl.col("p_text").str.len_chars()).sort("L")
    tok = AutoTokenizer.from_pretrained(ce_dir(name))
    model = AutoModelForSequenceClassification.from_pretrained(ce_dir(name)).to(acc.device).eval().half()
    dl = DataLoader(Pairs(shard["q_text"].to_list(), shard["p_text"].to_list(), None),
                    batch_size=bs, shuffle=False, collate_fn=_collate(tok), num_workers=2)
    out, t = [], time.time()
    prog = Progress(log, f"ce score {split} (batches)", len(dl)) if acc.is_main_process else None
    for i, batch in enumerate(dl):
        batch.pop("labels"), batch.pop("weights")
        batch = {k: v.to(acc.device) for k, v in batch.items()}
        out.append(model(**batch).logits.squeeze(-1).float().cpu().numpy())
        if prog:
            prog.step(1, pairs_per_s=f"{(i + 1) * bs * acc.num_processes / (time.time() - t):.0f}")
    scores = np.concatenate(out) if out else np.zeros(0, np.float32)
    n_bad = int((~np.isfinite(scores)).sum())
    if n_bad:
        log.warning(f"process {acc.process_index}: {n_bad:,} non-finite scores (fp16 overflow?) -> set to -20")
        scores = np.where(np.isfinite(scores), scores, -20.0).astype(np.float32)
    shard.select("q_idx", "p_idx").with_columns(ce_logit=pl.Series(scores)).write_parquet(
        WORK / f"{split}_ce{sfx(name)}.part{acc.process_index}.parquet"
    )
    acc.wait_for_everyone()
    if acc.is_main_process:
        parts = [WORK / f"{split}_ce{sfx(name)}.part{i}.parquet" for i in range(acc.num_processes)]
        res = pl.concat([pl.read_parquet(p) for p in parts])
        res.write_parquet(WORK / f"{split}_ce{sfx(name)}.parquet")
        for p in parts:
            p.unlink()
        log.info(f"wrote {split}_ce{sfx(name)}.parquet: {res.height:,} pairs, logit mean {res['ce_logit'].mean():.2f} "
                 f"({time.time() - t:.0f}s)")
        meta = {"share": share, "keep_k": keep_k, "gate": gate, "band": list(band)}
        (WORK / f"{split}_ce{sfx(name)}_meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "score", "validate"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--keep-k", type=int, default=20)
    ap.add_argument("--neg-per-q", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--share", type=float, default=1.0, help="fraction of queries to score")
    ap.add_argument("--limit-q", type=int, default=0, help="only this many S1 records (smoke tests)")
    ap.add_argument("--no-freeze-emb", action="store_true", help="also fine-tune the word embeddings")
    ap.add_argument("--gate", default="", help="parquet (q_idx, p_idx, prob) of v1 probabilities in WORK")
    ap.add_argument("--band", nargs=2, type=float, default=[0.02, 0.98], help="score pairs with lo <= prob <= hi")
    ap.add_argument("--name", default="", help="model tag: ce_model_<name>/, *_ce_<name>.parquet ('' = first model)")
    ap.add_argument("--base", default=MODEL, help="HF model id, or a checkpoint dir in WORK to resume from")
    ap.add_argument("--train-set", default="limit", choices=["limit", "all-but-val"],
                    help="all-but-val: every reserved S1 except the fixed 20k validation set")
    ap.add_argument("--latin-only", action="store_true", help="drop pairs with Indic scripts (French/Latin models)")
    ap.add_argument("--hard-pos-weight", type=float, default=1.0,
                    help="loss weight for true pairs whose candidate has no address or a native-script name")
    ap.add_argument("--pseudo", default="", help="probability parquet for self-training labels, e.g. test_probs_v2.parquet")
    ap.add_argument("--pseudo-country", default="France")
    ap.add_argument("--pseudo-q", type=int, default=50_000)
    a = ap.parse_args()
    if a.step == "train":
        train(a.keep_k, a.neg_per_q, a.epochs, a.lr, a.bs, a.limit_q, not a.no_freeze_emb, a.name, a.base,
              a.train_set, a.latin_only, a.hard_pos_weight, a.pseudo, a.pseudo_country, a.pseudo_q)
    elif a.step == "validate":
        validate(a.keep_k, a.bs, a.name)
    else:
        score(a.split, a.keep_k, a.bs, a.share, a.limit_q, a.gate, tuple(a.band), a.name)
