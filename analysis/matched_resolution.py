#!/usr/bin/env python
"""Readout or resolution? Verification quantised to the generative arm's own value histogram.

An alternative explanation: verification's P(yes) is continuous, the generative arm emits a
handful of integer levels, and the quantile calibration rewards resolution. So part of the gap
may be resolution rather than a better ordering of images. The control, implemented here:

  per model and reliable category, take exactly the multiset of values the generative arm
  emitted on the images it parsed, and hand it out to the same images in P(yes) order (ties in
  P(yes) broken at random, fixed seed). The result has the generative arm's histogram -- same
  levels, same counts per level, same ties -- and differs from it only in *which* image gets
  which value. It then goes through the identical calibration and scorer.

  matched - generative   = what ordering alone is worth (same resolution, same images)
  continuous - matched   = what resolution adds on top

Reported per model and fleet: kappa_w (calibrated, reliable five) and mAP (binarised gold), and
a paired image-bootstrap CI (B=500, calibration refit in every replicate) of matched -
generative and of matched - anchor.

Guards: printed headline figures (controls_common.reproduce), then the audit figures below.

Usage:
    python analysis/matched_resolution.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controls_common import add_common_args, ci, f3, fci, pmap, tex_name  # noqa: E402
from e0_report import macro_ap  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--match-seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("results/matched_resolution.json"))
    ap.add_argument("--tex", type=Path, default=Path("analysis/tables/matched_resolution.tex"))
    args = ap.parse_args()
    return compute(args)


def match_tieaware(vcol, gcol):
    """Tie-aware matched-resolution column: images in P(yes) order receive the generative arm's
    sorted values, but a block of tied P(yes) receives ONE value, the one at the block midpoint
    (upper median), so the construction is deterministic and order-free (no random tie-break).
    Images without a generative parse stay NaN, so all three columns share one image set."""
    out = np.full(vcol.shape, np.nan)
    ok = ~(np.isnan(vcol) | np.isnan(gcol))
    if ok.sum() < 1:
        return out
    gv = np.sort(gcol[ok])
    vals, inv, cnt = np.unique(vcol[ok], return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt
    pos = np.minimum(np.floor(start + cnt / 2.0).astype(int), len(gv) - 1)
    out[ok] = gv[pos][inv]
    return out


def compute(args) -> int:
    """Matched resolution.

    Three columns on the SAME images (those the text condition parsed): text, matched (P(yes)
    order, text histogram, tie-aware: `match_tieaware`), P(yes). The match is rebuilt inside
    every bootstrap replicate from that replicate's images (it depends on the other images' text
    values), so the interval no longer conditions on one realised match. Ordering share =
    (matched - text) / (P(yes) - text), fleet means, with a percentile CI of the ratio. Held-out
    primary, oracle beside it. matched - anchor uses the anchor on the same images per model.
    Output: results/matched_resolution.json, tables/matched_resolution.tex.
    """
    import scoring as s2

    D, gen, ver, head = s2.headline(args)
    s2.assert_headline(head)
    tags = sorted(ver)
    rel, n = D["rel"], D["n"]
    relmask = np.isin(np.arange(D["gold"].shape[1]), rel)
    parsed = {t: ~np.isnan(gen[t][1][:, rel[0]]) for t in tags}
    verc = {}
    for t in tags:
        v = ver[t].copy()
        v[~parsed[t]] = np.nan
        verc[t] = v

    def build(rows):
        """Matched matrices for this sample (only the reliable columns are filled)."""
        out = {}
        for t in tags:
            m = np.full(ver[t].shape, np.nan)
            for j in rel:
                col = np.full(n, np.nan)
                # rows may hold duplicates; the match is a function of the multiset, and tied
                # copies share a value, so assign through the sample and write back by image
                vals = match_tieaware(ver[t][rows, j], gen[t][1][rows, j])
                col[rows] = vals
                m[:, j] = col
            out[t] = m
        return out

    def score(Sm):
        mt = build(Sm.rows)
        o = {}
        for est in s2.ESTIMATORS:
            o[est] = np.array([[Sm.model(x, rel, est) for x in (gen[t][1], mt[t], verc[t], ver[t])]
                               for t in tags])            # (models, [text, matched, ver_same, ver_all])
        o["anchor_same"] = np.array([np.mean([s2.anchor_cat(Sm.H[parsed[t][Sm.rows]][:, j, :])
                                              for j in rel]) for t in tags])
        return o

    full = s2.Sample(D, np.arange(n), S=args.S)
    point = score(full)
    mt_full = build(np.arange(n))
    # histogram fidelity of the tie-aware match: share of cells whose value differs from the
    # exact (random-tie) rank assignment's multiset, i.e. total variation between histograms
    tv = []
    for t in tags:
        for j in rel:
            ok = parsed[t]
            a = np.unique(gen[t][1][ok, j], return_counts=True)
            b = dict(zip(*np.unique(mt_full[t][ok, j], return_counts=True)))
            tv.append(0.5 * sum(abs(c - b.get(v, 0)) for v, c in zip(*a)) / ok.sum())
    dr = s2.draws(n, args.replicates, args.seed)
    print(f"paired image bootstrap B={args.replicates}, S={args.S}, {args.workers} workers",
          flush=True)
    reps = pmap(lambda b: score(s2.Sample(D, dr[b], rep=b, S=args.S)),
                list(range(args.replicates)), args.workers)
    A = np.stack([r["anchor_same"] for r in reps])

    res = {"models": tags, "text_arm": {t: gen[t][0] for t in tags},
           "histogram_tv_mean": float(np.mean(tv)), "histogram_tv_max": float(np.max(tv)),
           "estimators": {}}
    for est in s2.ESTIMATORS:
        P = point[est]
        R = np.stack([r[est] for r in reps])                     # (B, m, 4)
        rows = []
        for i, t in enumerate(tags):
            rows.append({
                "model": t, "kappa_text": float(P[i, 0]), "kappa_matched": float(P[i, 1]),
                "kappa_pyes_same_images": float(P[i, 2]), "kappa_pyes_all_images": float(P[i, 3]),
                "matched_minus_text": float(P[i, 1] - P[i, 0]),
                "matched_minus_text_ci": ci(R[:, i, 1] - R[:, i, 0]),
                "anchor_same_images": float(point["anchor_same"][i]),
                "matched_minus_anchor_ci": ci(R[:, i, 1] - A[:, i]),
                "levels_per_category": float(np.mean(
                    [len(np.unique(gen[t][1][parsed[t], j])) for j in rel])),
            })
        num = (P[:, 1] - P[:, 0]).mean()
        den = (P[:, 2] - P[:, 0]).mean()
        Rn = (R[:, :, 1] - R[:, :, 0]).mean(1)
        Rd = (R[:, :, 2] - R[:, :, 0]).mean(1)
        Rd_all = (R[:, :, 3] - R[:, :, 0]).mean(1)
        res["estimators"][est] = {"rows": rows, "fleet": {
            "kappa_text": float(P[:, 0].mean()), "kappa_matched": float(P[:, 1].mean()),
            "kappa_pyes_same_images": float(P[:, 2].mean()),
            "kappa_pyes_all_images": float(P[:, 3].mean()),
            "matched_minus_text": float(num), "matched_minus_text_ci": ci(Rn),
            "pyes_minus_matched_ci": ci(Rd - Rn),
            "ordering_share": float(num / den), "ordering_share_ci": ci(Rn / Rd),
            "ordering_share_vs_reported_delta": float(num / (P[:, 3] - P[:, 0]).mean()),
            "ordering_share_vs_reported_delta_ci": ci(Rn / Rd_all),
            "matched_sig_positive": int(sum(r["matched_minus_text_ci"][0] > 0 for r in rows)),
            "matched_sig_negative": int(sum(r["matched_minus_text_ci"][1] < 0 for r in rows)),
            "matched_point_above_anchor": int(sum(r["kappa_matched"] > r["anchor_same_images"]
                                                  for r in rows)),
            "matched_ci_above_anchor": int(sum(r["matched_minus_anchor_ci"][0] > 0 for r in rows)),
            "matched_ci_below_anchor": int(sum(r["matched_minus_anchor_ci"][1] < 0 for r in rows)),
        }}
    res["map"] = {t: {"text": macro_ap(gen[t][1], D["gold"], relmask),
                      "matched": macro_ap(mt_full[t], D["gold"], relmask),
                      "pyes": macro_ap(ver[t], D["gold"], relmask)} for t in tags}
    res["map_fleet"] = {k: float(np.mean([v[k] for v in res["map"].values()]))
                        for k in ("text", "matched", "pyes")}

    for est in s2.ESTIMATORS:
        f = res["estimators"][est]["fleet"]
        print(f"{est:8} text {f['kappa_text']:.4f} matched {f['kappa_matched']:.4f} P(yes) same "
              f"images {f['kappa_pyes_same_images']:.4f} (all {f['kappa_pyes_all_images']:.4f}); "
              f"matched-text {f['matched_minus_text']:+.4f} [{f['matched_minus_text_ci'][0]:+.4f},"
              f"{f['matched_minus_text_ci'][1]:+.4f}]; share {f['ordering_share']:.3f} "
              f"[{f['ordering_share_ci'][0]:.3f},{f['ordering_share_ci'][1]:.3f}]; sig +/- "
              f"{f['matched_sig_positive']}/{f['matched_sig_negative']}; above anchor "
              f"{f['matched_point_above_anchor']} ({f['matched_ci_above_anchor']})")
    print(f"tie-aware match histogram TV distance: mean {res['histogram_tv_mean']:.4f}, max "
          f"{res['histogram_tv_max']:.4f}; mAP text/matched/P(yes) "
          + "/".join(f"{v:.3f}" for v in res["map_fleet"].values()))

    e = res["estimators"]["heldout"]
    lines = ["% generated by analysis/matched_resolution.py -- do not edit by hand",
             "% held-out mapping; all three columns on the images the text condition parsed",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & levels & text & matched & $P(\mathrm{yes})$ & matched$-$text [95\% CI] \\",
             r"\midrule"]
    for r in e["rows"]:
        lines.append(f"{tex_name(r['model'])} & ${r['levels_per_category']:.1f}$ & "
                     f"{f3(r['kappa_text'])} & {f3(r['kappa_matched'])} & "
                     f"{f3(r['kappa_pyes_same_images'])} & {f3(r['matched_minus_text'], True)} "
                     f"{fci(*r['matched_minus_text_ci'])} \\\\")
    f = e["fleet"]
    lines += [r"\midrule",
              f"Mean & & {f3(f['kappa_text'])} & {f3(f['kappa_matched'])} & "
              f"{f3(f['kappa_pyes_same_images'])} & {f3(f['matched_minus_text'], True)} "
              f"{fci(*f['matched_minus_text_ci'])} \\\\",
              f"\\multicolumn{{6}}{{l}}{{ordering share ${f['ordering_share']:.2f}$ "
              f"$[{f['ordering_share_ci'][0]:.2f}, {f['ordering_share_ci'][1]:.2f}]$}} \\\\",
              r"\bottomrule", r"\end{tabular}", ""]
    res["provenance"] = s2.provenance("matched_resolution.py", args,
                                      sources=[args.gen_dir, args.ver_dir, args.hq_csv])
    s2.write_json("matched_resolution.json", res)
    s2.write_tex(s2.TABLES_DIR / "matched_resolution.tex", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
