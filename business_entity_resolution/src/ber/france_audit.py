"""Audit French-specific text patterns in test France records and where v2 is unsure.

Counts how often each candidate normalization gap occurs (raw text), and prints examples
of uncertain / empty France predictions, to decide which France-only fixes to implement.
"""
import argparse

import polars as pl

from ber.io import WORK
from ber.log import get

log = get("fr_audit")

PATTERNS = {
    # name side
    "name: elision l'/d'/qu' (both apostrophes)": ("business_name", r"(?i)\b(?:l|d|qu|j|s|n|c)['’]\w"),
    "name: Sté / Ste": ("business_name", r"(?i)\bst[ée]\b\.?"),
    "name: spelled-out legal form": ("business_name", r"(?i)soci[ée]t[ée] (?:à|a) responsabilit|soci[ée]t[ée] par actions|entreprise individuelle|soci[ée]t[ée] anonyme|soci[ée]t[ée] civile"),
    "name: Ets / Établissements": ("business_name", r"(?i)\bets\b|[ée]tablissements"),
    "name: legal acronym (SARL/SAS/SA/EURL/SASU/SCI/SNC/SELARL)": ("business_name", r"(?i)\b(?:sarl|sas|sasu|eurl|sci|snc|selarl|sa)\b"),
    "name: & / et": ("business_name", r"(?i)&|\bet\b"),
    # address side
    "addr: 5-digit postcode": ("business_address", r"\b\d{5}\b"),
    "addr: CEDEX": ("business_address", r"(?i)\bcedex\b"),
    "addr: BP / CS (mail box)": ("business_address", r"(?i)\b(?:bp|cs)\s*\d+"),
    "addr: bis/ter/quater": ("business_address", r"(?i)\b\d+\s*(?:bis|ter|quater)\b"),
    "addr: number+letter (12B)": ("business_address", r"(?i)\b\d+[a-d]\b"),
    "addr: arrondissement (8e/8ème/1er)": ("business_address", r"(?i)\b\d{1,2}\s*(?:e|er|ème|eme)\b"),
    "addr: Dr/Gal/Mal/Pdt/Pte/Fg abbreviations": ("business_address", r"(?i)\b(?:dr|gal|g[ée]n|mal|pdt|pte|fg|st|ste)\b\.?"),
    "addr: street-type abbreviation (bd/av/r./pl/rte/ch/imp/all)": ("business_address", r"(?i)\b(?:bd|bld|av|ave|r|pl|rte|ch|chem|imp|all|crs|qu|sq)\b\.?"),
    "addr: elision d'/l'": ("business_address", r"(?i)\b(?:l|d)['’]\w"),
    "addr: region name": ("business_address", r"(?i)[îi]le-de-france|nouvelle-aquitaine|auvergne|occitanie|bretagne|normandie|grand est|hauts-de-france|provence|bourgogne|pays de la loire|centre-val|corse"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="test_probs_v2.parquet")
    ap.add_argument("--examples", type=int, default=25)
    a = ap.parse_args()
    cols = ["idx", "entity_id", "business_name", "business_address", "country", "name_core", "addr_norm", "addr_nums"]
    s1 = pl.scan_parquet(WORK / "test_s1.parquet").select(cols).filter(pl.col("country") == "France").collect()
    pool = pl.scan_parquet(WORK / "test_pool.parquet").select(cols).filter(pl.col("country") == "France").collect()
    log.info(f"France: {s1.height:,} S1, {pool.height:,} pool records")

    rows = []
    for label, (col, rx) in PATTERNS.items():
        rows.append({"pattern": label, "S1 share": s1[col].fill_null("").str.contains(rx).mean(),
                     "pool share": pool[col].fill_null("").str.contains(rx).mean()})
    with pl.Config(tbl_rows=40, fmt_str_lengths=70, tbl_width_chars=200, float_precision=4):
        print(pl.DataFrame(rows).sort("pool share", descending=True), flush=True)

    # how postcodes leak into address numbers used by the twin features
    pc = pool.filter(pl.col("business_address").fill_null("").str.contains(r"\b\d{5}\b"))
    print(f"\npool records with a 5-digit postcode: {pc.height:,}; examples of addr_nums:", flush=True)
    for r in pc.head(8).iter_rows(named=True):
        print(f"  {r['business_address'][:70]:70} -> {r['addr_nums']}")

    # where v2 is unsure / predicts nothing for France
    pr = pl.read_parquet(WORK / a.probs).join(s1.select(q_idx="idx"), on="q_idx")
    q = pr.group_by("q_idx").agg(top1=pl.col("prob").max(), n_unsure=pl.col("prob").is_between(0.1, 0.9).sum())
    log.info(f"France S1 with top prob < 0.5 (likely empty prediction): {(q['top1'] < 0.5).mean():.3%}; "
             f"with >=1 unsure candidate (0.1-0.9): {(q['n_unsure'] > 0).mean():.3%}")
    un = pr.filter(pl.col("prob").is_between(0.1, 0.9)).sample(min(a.examples, pr.height), seed=0)
    s1r = s1.select(q_idx="idx", s1_name="business_name", s1_addr="business_address")
    pr_ = pool.select(p_idx="idx", c_name="business_name", c_addr="business_address")
    print("\nUNSURE FRANCE PAIRS (v2 prob 0.1-0.9):", flush=True)
    for r in un.join(s1r, on="q_idx").join(pr_, on="p_idx").iter_rows(named=True):
        print(f"  p={r['prob']:.2f} | {r['s1_name'][:34]:34} | {str(r['c_name'])[:34]:34} | "
              f"{str(r['s1_addr'])[:45]:45} | {str(r['c_addr'])[:45]}")
    emp = q.filter(pl.col("top1") < 0.3).sample(min(a.examples // 2, q.height), seed=1)
    best = pr.sort("prob", descending=True).group_by("q_idx", maintain_order=True).first()
    print("\nFRANCE S1 WITH NO CONFIDENT CANDIDATE (best candidate shown):", flush=True)
    for r in emp.join(best, on="q_idx").join(s1r, on="q_idx").join(pr_, on="p_idx").iter_rows(named=True):
        print(f"  p={r['prob']:.2f} | {r['s1_name'][:34]:34} | {str(r['c_name'])[:34]:34} | "
              f"{str(r['s1_addr'])[:45]:45} | {str(r['c_addr'])[:45]}")


if __name__ == "__main__":
    main()
