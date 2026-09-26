"""Unlabeled diagnostic: compare prediction statistics per country on test with train OOF.

The data generator is shared across countries, so on a healthy model every country
should show a similar share of S1 with a match (~94%) and matches per S1 (~3.5). A
country far off (e.g. France, unseen in training) points at a recall/precision issue.
"""
import argparse
import json

import polars as pl

from ber.io import WORK
from ber.log import get
from ber.matcher import apply_rule

log = get("diag")


def stats(pr: pl.DataFrame, split: str, cfg: dict, label: str) -> pl.DataFrame:
    s1 = pl.read_parquet(WORK / f"{split}_s1.parquet", columns=["idx", "country"]).rename({"idx": "q_idx"})
    sel = apply_rule(pr, cfg)
    n_match = sel.group_by("q_idx").len("n")
    top = pr.group_by("q_idx").agg(top1=pl.col("prob").max(), n_over_05=(pl.col("prob") >= 0.5).sum(),
                                   n_over_02=(pl.col("prob") >= 0.2).sum(), mass=pl.col("prob").sum())
    q = s1.filter(pl.col("q_idx").is_in(pr["q_idx"].unique().implode()))
    x = q.join(n_match, on="q_idx", how="left").join(top, on="q_idx", how="left").fill_null(0)
    return x.group_by("country").agg(
        S1=pl.len(),
        nonempty=(pl.col("n") > 0).mean(),
        matches_per_matched=pl.col("n").filter(pl.col("n") > 0).mean(),
        top1_mean=pl.col("top1").mean(),
        top1_p10=pl.col("top1").quantile(0.1),
        top1_median=pl.col("top1").median(),
        n_prob_ge_05=pl.col("n_over_05").mean(),
        n_prob_ge_02=pl.col("n_over_02").mean(),
        prob_mass=pl.col("mass").mean(),
    ).with_columns(set=pl.lit(label)).sort("country")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="'' for v1, 'v2' for v2 files")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    cfg = json.loads((WORK / f"decision{sfx}.json").read_text())
    log.info(f"decision rule {cfg}")
    rows = [stats(pl.read_parquet(WORK / f"train_oof{sfx}.parquet"), "train", cfg, f"train OOF{sfx}"),
            stats(pl.read_parquet(WORK / f"test_probs{sfx}.parquet"), "test", cfg, f"test{sfx}")]
    with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250, float_precision=3):
        print(pl.concat(rows), flush=True)

    # candidate-generation health per country (independent of the matcher)
    for split in ("train", "test"):
        s1 = pl.read_parquet(WORK / f"{split}_s1.parquet", columns=["idx", "country"]).rename({"idx": "q_idx"})
        c = pl.scan_parquet(WORK / f"{split}_cand.parquet").select("q_idx", "rr_prob", "rr_rank", "block_score",
                                                                   "ng_name", "ng_name_addr").collect()
        t = c.group_by("q_idx").agg(rr_top1=pl.col("rr_prob").max(), rr_mass=pl.col("rr_prob").sum(),
                                   ng_name_top=pl.col("ng_name").max(), ng_na_top=pl.col("ng_name_addr").max(),
                                   block_top=pl.col("block_score").max())
        with pl.Config(tbl_rows=20, tbl_cols=20, tbl_width_chars=250, float_precision=3):
            print(f"\n{split} candidates:\n" + str(t.join(s1, on="q_idx").group_by("country").agg(
                pl.len().alias("S1"), pl.col("rr_top1").mean(), pl.col("rr_mass").mean(), pl.col("ng_name_top").mean(),
                pl.col("ng_na_top").mean(), pl.col("block_top").mean()).sort("country")), flush=True)
    ce = WORK / "test_ce.parquet"
    if ce.exists():
        s1 = pl.read_parquet(WORK / "test_s1.parquet", columns=["idx", "country"]).rename({"idx": "q_idx"})
        t = pl.read_parquet(ce).group_by("q_idx").agg(ce_top=pl.col("ce_logit").max(),
                                                    ce_n_pos=(pl.col("ce_logit") > 0).sum())
        with pl.Config(float_precision=3):
            print("\ntest cross-encoder:\n" + str(t.join(s1, on="q_idx").group_by("country").agg(
                pl.len().alias("S1"), pl.col("ce_top").mean(), pl.col("ce_n_pos").mean(),
                (pl.col("ce_n_pos") > 0).mean().alias("any_pos")).sort("country")), flush=True)


if __name__ == "__main__":
    main()
