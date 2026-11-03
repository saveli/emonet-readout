#!/usr/bin/env python
"""Leave-one-rater-out anchor: is a model above *consensus* agreement, or only above one expert?

The paper's anchor (0.468 on the reliable five) is one expert against another, raw. The models
are rank-matched to the marginal of the four-rater mean -- a mean that includes the rater they
are then scored against -- so a model is effectively predicting consensus and is compared with a
single noisy expert. The like-for-like anchor is a three-expert panel; this script computes it and puts the models through the same leave-one-out construction.

Anchors, per category, pooled over the four held-out folds k (a fold is a rating *slot*, not a
named person: hq.csv carries exactly four ratings per cell, in row order):

  hh            expert vs expert, raw                      the paper's 0.468 / 0.204
  single_cal    one expert pushed through the models' calibration (rank-matched to the
                four-rater-mean marginal, ties broken at random) vs each other expert
  loro_gold     mean of the other three, rank-matched to the four-rater-mean marginal (the
                models' exact path, so the held-out rater leaks into the calibration target
                just as it does for the models), vs held-out rater k
  loro_own      mean of the other three, rank-matched to its own three-rater marginal (no
                leakage), vs held-out rater k

Models, per category:

  std           the published path: rank-matched to the four-rater-mean marginal, scored
                against all four raters (`kappa_cal`)
  loro          rank-matched to the three-rater marginal of fold k, scored against held-out
                rater k, pooled over folds -- the model analogue of loro_own, with no
                leakage on either side

Primary comparison: model loro - anchor loro_own, paired image bootstrap (one resample shared
by anchor and every model, calibration refit inside every replicate). Secondary, for continuity
with an earlier version of this analysis: model std - anchor loro_gold.

Guard: the printed headline figures (controls_common.reproduce) must reproduce first.

Usage:
    python analysis/anchor_three_rater.py                 # B=500, ~1 min on 30 cores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controls_common import Drift, add_common_args, ci, pmap  # noqa: E402
from reliability_vs_performance import (  # noqa: E402
    _kappa_from_pairs, human_human_kappa, model_single_rater_kappa, quantile_bin,
)
from verify_bootstrap_ci import quantile_bin_shuffled  # noqa: E402

SETS = ("reliable_5", "all_40")


def pooled(pairs):
    x = np.concatenate([a for a, _ in pairs])
    y = np.concatenate([b for _, b in pairs])
    ok = ~(np.isnan(x) | np.isnan(y))
    return _kappa_from_pairs(x[ok], y[ok])


def category(H, g, cols, rng):
    """All anchors and all model scores for one category on one image sample.

    H: (n, 4) ratings; g: (n,) four-rater mean; cols: {key: (n,) model predictions}.
    """
    K = H.shape[1]
    cons = [(H.sum(axis=1) - H[:, k]) / (K - 1) for k in range(K)]
    out = {"hh": human_human_kappa(H)}
    sc = []
    for o in range(K):
        b = quantile_bin_shuffled(H[:, o], g, rng)
        sc += [(b, H[:, k]) for k in range(K) if k != o]
    out["single_cal"] = pooled(sc)
    out["loro_gold"] = pooled([(quantile_bin(cons[k], g), H[:, k]) for k in range(K)])
    out["loro_own"] = pooled([(quantile_bin(cons[k], cons[k]), H[:, k]) for k in range(K)])
    for key, p in cols.items():
        out[(key, "std")] = model_single_rater_kappa(quantile_bin(p, g), H)
        out[(key, "loro")] = pooled([(quantile_bin(p, cons[k]), H[:, k]) for k in range(K)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out", type=Path, default=Path("results/anchor_three_rater.json"))
    ap.add_argument("--tex", type=Path, default=Path("analysis/tables/anchor_three_rater.tex"))
    args = ap.parse_args()
    return compute(args)


def compute(args) -> int:
    """References (analysis/scoring.py), both estimators.

    single expert         mean pairwise kappa_w (the anchor)
    single expert, mapped one slot's ratings through the models' map, vs the other three
    three-expert panel    mean of three slots on its own 3-rater marginal vs the held-out slot
    Models: `std` = the reported score (map onto the 4-rater-mean marginal, all four slots);
    `loro` = mapped onto the same 3-rater marginal as the panel, scored vs the held-out slot.
    Tests (paired, same draws and splits): std - anchor, std - single_mapped, loro - panel.
    Output: results/anchor_three_rater.json, tables/anchor_three_rater.tex.
    """
    import json

    import scoring as s2

    D, gen, ver, head = s2.headline(args)
    s2.assert_headline(head)
    tags = sorted(ver)
    mats = {"text": {t: gen[t][1] for t in tags}, "pyes": ver}
    sets = {"reliable_5": D["rel"], "all_40": D["all40"]}
    n = D["n"]

    def score(Sm):
        out = {}
        for sn, cs in sets.items():
            o = {"anchor": Sm.anchor(cs, "mean"), "anchor_pooled": Sm.anchor(cs, "pooled")}
            for est in s2.ESTIMATORS:
                o[("single_mapped", est)] = Sm.single_mapped(cs, est)
                o[("panel", est)] = Sm.panel(cs, est)
                for c in ("text", "pyes"):
                    for t in tags:
                        o[(c, t, "std", est)] = Sm.model(mats[c][t], cs, est)
                        o[(c, t, "loro", est)] = Sm.model_panel(mats[c][t], cs, est)
            out[sn] = o
        return out

    point = score(s2.Sample(D, np.arange(n), S=args.S))
    dr = s2.draws(n, args.replicates, args.seed)
    print(f"paired image bootstrap B={args.replicates}, S={args.S}, {args.workers} workers",
          flush=True)
    reps = pmap(lambda b: score(s2.Sample(D, dr[b], rep=b, S=args.S)),
                list(range(args.replicates)), args.workers)

    res = {"models": tags, "text_arm": {t: gen[t][0] for t in tags}, "sets": {}}
    for sn in sets:
        P = point[sn]

        def R(k):
            return np.array([r[sn][k] for r in reps])

        refs = {"anchor": {"point": P["anchor"], "ci": ci(R("anchor"))},
                "anchor_pooled": {"point": P["anchor_pooled"], "ci": ci(R("anchor_pooled"))}}
        for est in s2.ESTIMATORS:
            for rk in ("single_mapped", "panel"):
                refs[f"{rk}/{est}"] = {"point": P[(rk, est)], "ci": ci(R((rk, est)))}
        tests = {}
        comps = {"std_vs_anchor": ("std", lambda est: "anchor"),
                 "std_vs_single_mapped": ("std", lambda est: ("single_mapped", est)),
                 "loro_vs_panel": ("loro", lambda est: ("panel", est))}
        rows = []
        for est in s2.ESTIMATORS:
            for c in ("text", "pyes"):
                cnt = {}
                for name, (mk, rk) in comps.items():
                    above = below = pt = 0
                    for t in tags:
                        d = R((c, t, mk, est)) - R(rk(est))
                        lo, hi = ci(d)
                        diff = P[(c, t, mk, est)] - P[rk(est)]
                        rows.append({"model": t, "condition": c, "estimator": est,
                                     "comparison": name, "kappa": P[(c, t, mk, est)],
                                     "reference": P[rk(est)], "diff": diff, "ci": [lo, hi],
                                     "verdict": s2.verdict(lo, hi)})
                        pt += diff > 0
                        above += lo > 0
                        below += hi < 0
                    cnt[name] = {"point_above": int(pt), "ci_above": int(above),
                                 "ci_below": int(below)}
                tests[f"{c}/{est}"] = cnt
        fleet = {}
        for est in s2.ESTIMATORS:
            for mk in ("std", "loro"):
                g = np.array([P[("text", t, mk, est)] for t in tags])
                v = np.array([P[("pyes", t, mk, est)] for t in tags])
                dd = np.mean([R(("pyes", t, mk, est)) - R(("text", t, mk, est)) for t in tags], 0)
                fleet[f"{mk}/{est}"] = {"text": float(g.mean()), "pyes": float(v.mean()),
                                        "delta": float((v - g).mean()), "delta_ci": ci(dd)}
        res["sets"][sn] = {"references": refs, "tests": tests, "fleet": fleet, "rows": rows}

        print(f"\n[{sn}] references: " + "  ".join(
            f"{k} {v['point']:.4f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]" for k, v in refs.items()))
        for key, cnt in tests.items():
            print(f"  {key:15}" + "  ".join(f"{k} {v['point_above']}/{v['ci_above']}/"
                                           f"{v['ci_below']}" for k, v in cnt.items()))
        for key, f in fleet.items():
            print(f"  fleet {key:14} text {f['text']:.4f} P(yes) {f['pyes']:.4f} Delta "
                  f"{f['delta']:+.4f} [{f['delta_ci'][0]:+.4f},{f['delta_ci'][1]:+.4f}]")
    print("  (counts: point above / CI above / CI below)")

    # Cross-check against fleet.py: same draws and splits, so identical model scores and
    # identical model - anchor intervals, to rounding.
    fj = s2.RESULTS_DIR / "fleet.json"
    if fj.exists():
        F = json.load(open(fj))
        if F["provenance"]["B"] == args.replicates and F["provenance"]["seed"] == args.seed:
            bad = []
            for est in s2.ESTIMATORS:
                for c in ("text", "pyes"):
                    for fr in F["sets"]["reliable_5"]["estimators"][est]["conditions"][c]["rows"]:
                        mine = [r for r in res["sets"]["reliable_5"]["rows"]
                                if r["model"] == fr["model"] and r["condition"] == c
                                and r["estimator"] == est and r["comparison"] == "std_vs_anchor"][0]
                        if (abs(mine["kappa"] - fr["kappa"]) > 1e-12
                                or not np.allclose(mine["ci"], fr["minus_anchor_ci"], atol=1e-12)):
                            bad.append((fr["model"], c, est))
            if bad:
                raise Drift(f"STOP: model - anchor differs from fleet.json: {bad}")
            print("model scores and model - anchor CIs identical to results/fleet.json")

    lab = {"anchor": "one expert (mean pairwise $\\kappa_w$)",
           "single_mapped/heldout": "one expert through the mapping",
           "panel/heldout": "three-expert panel vs held-out expert"}
    cmp_for = {"anchor": "std_vs_anchor", "single_mapped/heldout": "std_vs_single_mapped",
               "panel/heldout": "loro_vs_panel"}
    lines = ["% generated by analysis/anchor_three_rater.py -- do not edit by hand",
             "% held-out mapping; counts: P(yes) models above the reference, point (CI excludes 0)",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Reference & \multicolumn{2}{c}{five categories} & \multicolumn{2}{c}{all 40} \\",
             r" & $\kappa_w$ [95\% CI] & $P(\mathrm{yes})$ above & $\kappa_w$ [95\% CI] & "
             r"$P(\mathrm{yes})$ above \\", r"\midrule"]
    for k in lab:
        cells = []
        for sn in sets:
            x = res["sets"][sn]["references"][k]
            c = res["sets"][sn]["tests"]["pyes/heldout"][cmp_for[k]]
            cells += [f"${x['point']:.3f}$ $[{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]$",
                      f"${c['point_above']}\\,({c['ci_above']})/11$"]
        lines.append(f"{lab[k]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    res["provenance"] = s2.provenance("anchor_three_rater.py", args,
                                      sources=[args.gen_dir, args.ver_dir, args.hq_csv],
                                      secondary_sets=["all_40"])
    s2.write_json("anchor_three_rater.json", res)
    s2.write_tex(s2.TABLES_DIR / "anchor_three_rater.tex", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
