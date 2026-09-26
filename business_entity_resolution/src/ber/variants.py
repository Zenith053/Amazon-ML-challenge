"""Stricter decision variants from saved probabilities (no retraining).

Test differs from train: France is unseen and over-matched, and test has ~25% more
pool records per S1 (more distractors). Each variant shifts probabilities in log-odds
per country (negative = stricter) and/or uses a fixed threshold, then applies the
decision rule. Reports train-OOF cost and per-country test statistics, and writes a
zipped matching_results.tsv per variant for leaderboard checks.

  python -m ber.variants --tag v2 --spec 'A:France=-1.5' 'B:*=-1.0' 'C:thr=0.9'
"""
import argparse
import json
import zipfile

import polars as pl

from ber import submit
from ber.decide import expected_f05_sets, one_to_one
from ber.io import OUTPUT, WORK
from ber.log import get
from ber.matcher import apply_rule, eligible_train_queries, eval_cfg
from ber.metric import macro_f05

log = get("variants")


def parse(spec: str) -> tuple[str, dict, float | None]:
    name, _, body = spec.partition(":")
    shifts, thr = {}, None
    for part in filter(None, body.split(",")):
        k, v = part.split("=")
        if k == "thr":
            thr = float(v)
        else:
            shifts[k] = float(v)
    return name, shifts, thr


def shifted(pr: pl.DataFrame, split: str, shifts: dict) -> pl.DataFrame:
    if not shifts:
        return pr
    s1 = pl.read_parquet(WORK / f"{split}_s1.parquet", columns=["idx", "country"]).rename({"idx": "q_idx"})
    x = pr.join(s1, on="q_idx", how="left")
    delta = pl.lit(shifts.get("*", 0.0))
    for c, d in shifts.items():
        if c != "*":
            delta = pl.when(pl.col("country") == c).then(pl.lit(d)).otherwise(delta)
    p = pl.col("prob").clip(1e-6, 1 - 1e-6)
    logit = (p / (1 - p)).log() + delta
    return x.with_columns(prob=(1 / (1 + (-logit).exp())).cast(pl.Float32)).drop("country")


def decide(pr: pl.DataFrame, base: dict, thr: float | None) -> pl.DataFrame:
    if thr is None:
        return apply_rule(pr, base)
    return one_to_one(pr).filter(pl.col("prob") >= thr).select("q_idx", "p_idx")


def country_stats(sel: pl.DataFrame, split: str) -> pl.DataFrame:
    s1 = pl.read_parquet(WORK / f"{split}_s1.parquet", columns=["idx", "country"]).rename({"idx": "q_idx"})
    n = sel.group_by("q_idx").len("n")
    x = s1.join(n, on="q_idx", how="left").fill_null(0)
    return x.group_by("country").agg(nonempty=(pl.col("n") > 0).mean(),
                                     per_matched=pl.col("n").filter(pl.col("n") > 0).mean(),
                                     total=pl.col("n").sum()).sort("country")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--spec", nargs="+", required=True)
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    base = json.loads((WORK / f"decision{sfx}.json").read_text())
    oof = pl.read_parquet(WORK / f"train_oof{sfx}.parquet")
    test = pl.read_parquet(WORK / f"test_probs{sfx}.parquet")
    _, good = eligible_train_queries()
    out_dir = OUTPUT / "variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = test.select("q_idx", "p_idx")
    log.info(f"base rule {base}; train OOF F0.5 {eval_cfg(oof, good, base):.4f}")
    for spec in ["base:"] + a.spec:
        name, shifts, thr = parse(spec)
        tr_sel = decide(shifted(oof, "train", shifts), base, thr)
        gt = pl.read_parquet(WORK / "train_gt_pairs.parquet").filter(pl.col("q_idx").is_in(good.implode()))
        f_tr = macro_f05(tr_sel.filter(pl.col("q_idx").is_in(good.implode())).select(s1="q_idx", match="p_idx"),
                         gt.rename({"q_idx": "s1", "p_idx": "match"}), good)
        sel = decide(shifted(test, "test", shifts), base, thr)
        with pl.Config(float_precision=3):
            log.info(f"[{name}] shifts={shifts} thr={thr}  train OOF F0.5 {f_tr:.4f}  test matches {sel.height:,}\n"
                     f"{country_stats(sel, 'test')}")
        vdir = out_dir / name
        submit.write(sel, cands, out=vdir)
        (vdir / "candidate_pairs.tsv").unlink()
        with zipfile.ZipFile(out_dir / f"{name}_matching_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(vdir / "matching_results.tsv", "matching_results.tsv")
    log.info(f"wrote {sorted(p.name for p in out_dir.glob('*.zip'))}")


if __name__ == "__main__":
    main()
