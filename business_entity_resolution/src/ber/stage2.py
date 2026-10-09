"""Stage-2 stacker on top of a stage-1 matcher (v3full): re-scores every candidate with the
stage-1 features, the stage-1 probability, and features that only exist once stage-1
probabilities exist:

  - confirmed-match cluster: each S1's confirmed matches (p1 >= CONF); for every other
    candidate its best name / address similarity, exact name / compact-name equality and
    shared house numbers with a confirmed match. S2/S3 hold several copies of the same
    business, so a candidate that looks like a confirmed copy (e.g. same name, empty
    address) is likely a match too; one that looks different from all of them is not.
  - same-name ambiguity: other candidates of the S1 with the identical core name, and how
    many of them are confirmed (near-identical twins vs genuine duplicates).
  - competition recomputed with p1 across all S1s, plus query context (rank, gap, mass).

  oof     2-fold OOF on the scored train sample, decision rule, comparison with stage 1
  fit     train on all eligible queries, predict test probabilities
  decide  apply the OOF decision rule to the test probabilities, write + validate outputs

Env: BER_STAGE1 (stage-1 tag, default v3full), BER_TAG (this version, default v4), plus the
stage-1 environment (BER_CTX_TAG, BER_CE_NAMES) so stage-1 features load identically.
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from ber import matcher as M
from ber import submit
from ber.io import OUTPUT, WORK
from ber.log import Progress, get, timed
from ber.metric import macro_f05

log = get("stage2")
STAGE1 = os.environ.get("BER_STAGE1", "v3full")
TAG2 = os.environ.get("BER_TAG", "v4") or "v4"
CONF = 0.9
G_FEATURES = ["p1", "s_rank", "s_top1", "s_gap", "s_nconf", "s_mass", "s_nmid",
              "s_claims", "s_prank", "s_best_other", "s_margin"]
C_FEATURES = ["c_n", "c_name_max", "c_addr_max", "c_name_eq", "c_compact_eq", "c_num_any",
              "c_same_name_n", "c_same_name_conf"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=os.cpu_count())
REC = ["idx", "name_core", "name_compact", "addr_norm", "addr_nums"]


def features() -> list[str]:
    return M.features("train") + G_FEATURES + C_FEATURES


def p1_table(split: str) -> pl.DataFrame:
    """Stage-1 probability for every candidate pair of the split."""
    if split == "test":
        return pl.read_parquet(WORK / f"test_probs_{STAGE1}.parquet")
    a = pl.read_parquet(WORK / f"train_oof_{STAGE1}.parquet")
    # queries outside the scored sample: v1 OOF, as stage 1 did for its competition context
    rest = pl.read_parquet(WORK / "train_oof.parquet").filter(~pl.col("q_idx").is_in(a["q_idx"].unique().implode()))
    return pl.concat([a, rest.select(a.columns)])


def global_feats(p1: pl.DataFrame) -> pl.DataFrame:
    """Features that need all queries at once (competition across S1s)."""
    g = p1.rename({"prob": "p1"}).with_columns(
        s_rank=pl.col("p1").rank("ordinal", descending=True).over("q_idx").cast(pl.Float32),
        s_top1=pl.col("p1").max().over("q_idx"),
        s_nconf=(pl.col("p1") >= CONF).sum().over("q_idx").cast(pl.Float32),
        s_mass=pl.col("p1").sum().over("q_idx"),
        s_nmid=pl.col("p1").is_between(0.1, CONF, closed="left").sum().over("q_idx").cast(pl.Float32),
        s_claims=(pl.col("p1") >= 0.1).sum().over("p_idx").cast(pl.Float32),
        s_prank=pl.col("p1").rank("ordinal", descending=True).over("p_idx").cast(pl.Float32),
        _m1=pl.col("p1").max().over("p_idx"),
        _m2=pl.col("p1").sort(descending=True).get(1, null_on_oob=True).over("p_idx").fill_null(0.0),
    ).with_columns(
        s_gap=pl.col("s_top1") - pl.col("p1"),
        s_best_other=pl.when(pl.col("s_prank") == 1).then(pl.col("_m2")).otherwise(pl.col("_m1")),
    ).with_columns(s_margin=pl.col("p1") - pl.col("s_best_other")).drop("_m1", "_m2")
    return g


def _sim(a: list, b: list, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def cluster_feats(split: str, g: pl.DataFrame) -> pl.DataFrame:
    """g: rows (q_idx, p_idx, p1) for whole queries. Returns (q_idx, p_idx, C_FEATURES)."""
    recs = (pl.scan_parquet(WORK / f"{split}_pool.parquet").select(REC)
            .filter(pl.col("idx").is_in(g["p_idx"].unique().implode())).collect())
    x = g.select("q_idx", "p_idx", "p1").with_row_index("r").join(
        recs.rename({"idx": "p_idx"}), on="p_idx", how="left")
    # same-name ambiguity within the query
    x = x.with_columns(
        c_same_name_n=(pl.len().over("q_idx", "name_core") - 1).cast(pl.Float32),
        c_same_name_conf=((pl.col("p1") >= CONF).sum().over("q_idx", "name_core")
                          - (pl.col("p1") >= CONF).cast(pl.UInt32)).cast(pl.Float32),
    )
    anc = x.filter(pl.col("p1") >= CONF).select(
        "q_idx", a_idx="p_idx", a_name="name_core", a_compact="name_compact", a_addr="addr_norm", a_nums="addr_nums")
    pr = x.select("r", "q_idx", "p_idx", "name_core", "name_compact", "addr_norm", "addr_nums").join(
        anc, on="q_idx").filter(pl.col("a_idx") != pl.col("p_idx"))
    if pr.height:
        pr = pr.with_columns(
            sn=pl.Series(_sim(pr["name_core"].fill_null("").to_list(), pr["a_name"].fill_null("").to_list(), fuzz.token_set_ratio)),
            sa=pl.Series(_sim(pr["addr_norm"].fill_null("").to_list(), pr["a_addr"].fill_null("").to_list(), fuzz.token_set_ratio)),
        ).with_columns(
            sa=pl.when((pl.col("addr_norm").fill_null("") == "") | (pl.col("a_addr").fill_null("") == ""))
            .then(None).otherwise(pl.col("sa")),
            eq=(pl.col("name_core") == pl.col("a_name")).cast(pl.Int8),
            ceq=(pl.col("name_compact") == pl.col("a_compact")).cast(pl.Int8),
            num=(pl.col("addr_nums").list.set_intersection(pl.col("a_nums")).list.len() > 0).cast(pl.Int8),
        )
        agg = pr.group_by("r").agg(
            c_n=pl.len().cast(pl.Float32), c_name_max=pl.col("sn").max(), c_addr_max=pl.col("sa").max(),
            c_name_eq=pl.col("eq").max().cast(pl.Float32), c_compact_eq=pl.col("ceq").max().cast(pl.Float32),
            c_num_any=pl.col("num").max().cast(pl.Float32))
    else:
        agg = pl.DataFrame(schema={"r": pl.UInt32, **{k: pl.Float32 for k in
                           ["c_n", "c_name_max", "c_addr_max", "c_name_eq", "c_compact_eq", "c_num_any"]}})
    out = x.select("r", "q_idx", "p_idx", "c_same_name_n", "c_same_name_conf").join(agg, on="r", how="left")
    return out.with_columns(pl.col("c_n").fill_null(0.0)).drop("r")


def rows(split: str, q_ids: pl.Series, g: pl.DataFrame) -> pl.DataFrame:
    base = M._rows(split, q_ids)
    gg = g.filter(pl.col("q_idx").is_in(q_ids.implode()))
    c = cluster_feats(split, gg.select("q_idx", "p_idx", "p1"))
    return base.join(gg, on=["q_idx", "p_idx"], how="left").join(c, on=["q_idx", "p_idx"], how="left")


def build(split: str, q_ids: pl.Series, g: pl.DataFrame, chunk: int = 100_000, labels: bool = False) -> pl.DataFrame:
    gt = pl.read_parquet(WORK / "train_gt_pairs.parquet").with_columns(label=pl.lit(1, pl.Int8)) if labels else None
    prog = Progress(log, f"{split} stage-2 rows (queries)", len(q_ids))
    out = []
    for s in range(0, len(q_ids), chunk):
        ids = q_ids.slice(s, chunk)
        r = rows(split, ids, g)
        if labels:
            r = r.join(gt, on=["q_idx", "p_idx"], how="left").with_columns(pl.col("label").fill_null(0))
        out.append(r.select(["q_idx", "p_idx", *features(), *(["label"] if labels else [])]))
        prog.step(len(ids))
    return pl.concat(out)


def X(df: pl.DataFrame) -> np.ndarray:
    return df.select(pl.col(features()).cast(pl.Float32)).to_numpy()


def train_models(df: pl.DataFrame, seeds: list[int]) -> list[lgb.Booster]:
    """Early stopping on a 10% query slice; one model per seed (different bagging)."""
    val = (df["q_idx"].hash(seed=13) % 10) == 0
    tr, va = df.filter(~val), df.filter(val)
    dtr = lgb.Dataset(X(tr), tr["label"].to_numpy(), free_raw_data=False)
    dva = lgb.Dataset(X(va), va["label"].to_numpy(), reference=dtr, free_raw_data=False)
    models = []
    for sd in seeds:
        with timed(log, f"LightGBM seed {sd}: {tr.height:,} train / {va.height:,} val pairs, {len(features())} features"):
            m = lgb.train({**PARAMS, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd}, dtr,
                          num_boost_round=3000, valid_sets=[dva], valid_names=["val"],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
        log.info(f"seed {sd}: best iteration {m.best_iteration}, val logloss {m.best_score['val']['binary_logloss']:.5f}")
        models.append(m)
    return models


def predict(models: list[lgb.Booster], df: pl.DataFrame) -> np.ndarray:
    x = X(df)
    return np.mean([m.predict(x, num_iteration=m.best_iteration) for m in models], axis=0).astype(np.float32)


def oof(seeds: list[int]) -> None:
    all_q, good = M.eligible_train_queries()
    with timed(log, "stage-1 probabilities + global features (train)"):
        g = global_feats(p1_table("train"))
    df = build("train", good, g, labels=True)
    fold = (df["q_idx"].hash(seed=7) % 2).cast(pl.Int8)  # same folds as stage 1
    preds = []
    for k in (0, 1):
        models = train_models(df.filter(fold != k), seeds)
        te = df.filter(fold == k)
        preds.append(te.select("q_idx", "p_idx").with_columns(prob=pl.Series(predict(models, te))))
        if k == 0:
            imp = sorted(zip(models[0].feature_importance("gain"), features()), reverse=True)[:25]
            print("top features:", [f for _, f in imp], flush=True)
    pr = pl.concat(preds)
    pr.write_parquet(WORK / f"train_oof_{TAG2}.parquet")
    # competition context for queries outside the sample: stage-1 probabilities, as stage 1 did
    full = pl.concat([pr, g.filter(~pl.col("q_idx").is_in(pr["q_idx"].unique().implode()))
                      .select("q_idx", "p_idx", prob=pl.col("p1"))])
    best = M.choose_rule(full, good)
    (WORK / f"decision_{TAG2}.json").write_text(json.dumps(best))
    base = pl.concat([pl.read_parquet(WORK / f"train_oof_{STAGE1}.parquet"),
                      g.filter(~pl.col("q_idx").is_in(pr["q_idx"].unique().implode())).select("q_idx", "p_idx", prob=pl.col("p1"))])
    f1 = M.eval_cfg(base, good, json.loads((WORK / f"decision_{STAGE1}.json").read_text()))
    log.info(f"SAME EVAL SET ({len(good):,} S1): {STAGE1} F0.5 {f1:.4f}  vs  {TAG2} F0.5 {best['oof_f05']:.4f}  "
             f"(delta {best['oof_f05'] - f1:+.4f})")


def fit(seeds: list[int], chunk: int = 150_000) -> None:
    _, good = M.eligible_train_queries()
    with timed(log, "stage-1 probabilities + global features (train)"):
        g_tr = global_feats(p1_table("train"))
    models = train_models(build("train", good, g_tr, labels=True), seeds)
    for i, m in enumerate(models):
        m.save_model(str(WORK / f"stage2_{TAG2}_seed{i}.txt"), num_iteration=m.best_iteration)
    del g_tr
    with timed(log, "stage-1 probabilities + global features (test)"):
        g = global_feats(p1_table("test"))
    q_ids = g["q_idx"].unique().sort()
    prog = Progress(log, "test stage-2 predict (queries)", len(q_ids))
    outs = []
    for s in range(0, len(q_ids), chunk):
        ids = q_ids.slice(s, chunk)
        r = rows("test", ids, g).select(["q_idx", "p_idx", *features()])
        outs.append(r.select("q_idx", "p_idx").with_columns(prob=pl.Series(predict(models, r))))
        prog.step(len(ids))
    pr = pl.concat(outs)
    pr.write_parquet(WORK / f"test_probs_{TAG2}.parquet")
    log.info(f"test probabilities: {pr.height:,} pairs, mean {pr['prob'].mean():.4f}")


def decide() -> None:
    cfg = json.loads((WORK / f"decision_{TAG2}.json").read_text())
    pr = pl.read_parquet(WORK / f"test_probs_{TAG2}.parquet")
    matches = M.apply_rule(pr, cfg)
    n_q = pl.scan_parquet(WORK / "test_s1.parquet").select(pl.len()).collect().item()
    log.info(f"rule {cfg} -> {matches.height:,} matches, {matches['q_idx'].n_unique() / n_q:.3f} of S1 non-empty")
    out = OUTPUT / TAG2
    submit.write(matches, pr.select("q_idx", "p_idx"), out=out)
    assert submit.validate(out), "submission failed validation"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["oof", "fit", "decide"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    t = time.time()
    {"oof": lambda: oof(a.seeds), "fit": lambda: fit(a.seeds), "decide": decide}[a.step]()
    log.info(f"done in {time.time() - t:.0f}s")
