#!/usr/bin/env python
"""Per-model kappa_w for both conditions, fleet contrast, and the single-expert test.

Through the scoring path (analysis/scoring.py):

  * per model, text parsing (better of paper_once / ours, chosen under held-out, J6) and
    P(yes) scoring (results_verify_prefill), held-out (primary) and oracle mapping, 95%
    percentile CI from a B=500 image bootstrap (sorted draws, splits redrawn per replicate);
  * expert reference = mean pairwise kappa_w (pooled-pairs version alongside);
  * paired tests model - reference per model: "above by point / CI excludes 0 above / below";
  * fleet means and Delta = P(yes) - text with paired CI, both estimators;
  * between-model SD and median CI width per condition (forest figure, panel b);
  * per-category kappa for every model and the anchor, all 40 categories (fig_alpha,
    tab:percat), with the reliable five and all 40 as category-set means.

The single-expert-through-the-mapping and three-expert-panel references are in
anchor_three_rater.py, which runs on the same draws and splits and must reproduce this
script's model - anchor tests exactly (it asserts so when both JSONs exist).

Output: results/fleet.json, tables/permodel_core.tex.

Usage:
    python analysis/fleet.py                 # B=500, ~2 min on 30 cores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoring as s2  # noqa: E402

SETS = ("reliable_5", "all_40", "unreliable_35")
CONDS = ("text", "pyes")


def score_sample(Sm, mats, tags, n_cat, ests=s2.ESTIMATORS):
    """{est: {cond: (models, 40)}}, anchor per category (mean, pooled)."""
    cats = list(range(n_cat))
    out = {est: {c: np.array([Sm.model_cats(mats[c][t], cats, est) for t in tags])
                 for c in CONDS} for est in ests}
    out["anchor_mean"] = Sm.anchor_cats(cats, "mean")
    out["anchor_pooled"] = Sm.anchor_cats(cats, "pooled")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    s2.add_args(ap)
    args = ap.parse_args()

    D, gen, ver, head = s2.headline(args)
    s2.assert_headline(head)
    tags = sorted(ver)
    assert len(tags) == 11 and set(gen) == set(ver), (sorted(gen), sorted(ver))
    mats = {"text": {t: gen[t][1] for t in tags}, "pyes": ver}
    n, n_cat = D["n"], len(D["short"])
    sets = {"reliable_5": D["rel"], "all_40": D["all40"],
            "unreliable_35": [j for j in D["all40"] if j not in D["rel"]]}

    full = s2.Sample(D, np.arange(n), S=args.S)
    point = score_sample(full, mats, tags, n_cat)
    # second model-vs-expert convention (J5): one kappa pooled over the four slots, point only
    mean_slots = {est: {c: np.array([full.model_cats(mats[c][t], list(range(n_cat)), est, "pooled")
                                     for t in tags]) for c in CONDS} for est in s2.ESTIMATORS}
    dr = s2.draws(n, args.replicates, args.seed)
    print(f"image bootstrap B={args.replicates}, S={args.S} splits per replicate, "
          f"{args.workers} workers", flush=True)
    reps = s2.pmap(lambda b: score_sample(s2.Sample(D, dr[b], rep=b, S=args.S), mats, tags,
                                          n_cat), list(range(args.replicates)), args.workers)

    def setmean(arr, cs):          # arr (..., 40)
        return np.nanmean(arr[..., cs], axis=-1)

    _, _, cands = s2.select_gen(args, D)
    res = {"models": tags, "text_arm": {t: gen[t][0] for t in tags},
           "text_arms_both": {t: cands[t] for t in tags},
           "text_arm_note": "reported text = better of paper_once / ours on the evaluation images "
                            "under the primary estimator (J6: text-favourable upper envelope); "
                            "text_arms_both = both arms' held-out kappa, reliable five",
           "sets": {}}
    for sname, cs in sets.items():
        anc = {h: float(setmean(point[f"anchor_{h}"], cs)) for h in ("mean", "pooled")}
        anc_r = {h: np.array([setmean(r[f"anchor_{h}"], cs) for r in reps])
                 for h in ("mean", "pooled")}
        block = {"anchor": {h: {"point": anc[h], "ci": s2.ci(anc_r[h])} for h in anc},
                 "estimators": {}}
        for est in s2.ESTIMATORS:
            P = {c: setmean(point[est][c], cs) for c in CONDS}                        # (m,)
            R = {c: np.stack([setmean(r[est][c], cs) for r in reps]) for c in CONDS}  # (B, m)
            eb = {"conditions": {}, "fleet": {}}
            for c in CONDS:
                rows = []
                for i, t in enumerate(tags):
                    lo, hi = s2.ci(R[c][:, i])
                    dlo, dhi = s2.ci(R[c][:, i] - anc_r["mean"])
                    rows.append({"model": t, "kappa": float(P[c][i]),
                                 "kappa_pooled_over_slots": float(setmean(mean_slots[est][c][i], cs)),
                                 "ci_lo": lo, "ci_hi": hi,
                                 "se": float(R[c][:, i].std(ddof=1)), "width": hi - lo,
                                 "minus_anchor": float(P[c][i] - anc["mean"]),
                                 "minus_anchor_ci": [dlo, dhi], "verdict": s2.verdict(dlo, dhi)})
                k = P[c]
                eb["conditions"][c] = {
                    "rows": rows, "mean": float(k.mean()), "mean_ci": s2.ci(R[c].mean(1)),
                    "pooled_over_slots_convention": float(np.mean([r["kappa_pooled_over_slots"]
                                                                 for r in rows])),
                    "between_model_sd": float(k.std(ddof=1)),
                    "median_ci_width": float(np.median([r["width"] for r in rows])),
                    "above_point": int((k > anc["mean"]).sum()),
                    "above_ci": int(sum(r["verdict"] == "above" for r in rows)),
                    "below_ci": int(sum(r["verdict"] == "below" for r in rows)),
                }
            d = P["pyes"] - P["text"]
            Rd = R["pyes"] - R["text"]
            eb["fleet"] = {
                "delta": float(d.mean()), "delta_ci": s2.ci(Rd.mean(1)),
                "per_model_delta": {t: float(d[i]) for i, t in enumerate(tags)},
                "per_model_delta_ci": {t: s2.ci(Rd[:, i]) for i, t in enumerate(tags)},
                "pyes_wins": int((d > 0).sum()),
                "pyes_wins_ci": int(sum(s2.ci(Rd[:, i])[0] > 0 for i in range(len(tags)))),
            }
            block["estimators"][est] = eb
        res["sets"][sname] = block

    # per category (all 40), primary estimator, plus the oracle column
    res["per_category"] = []
    for j in range(n_cat):
        row = {"emotion": D["short"][j], "alpha": float(D["alpha"][j]),
               "reliable": j in D["rel"],
               "anchor_kappa": float(point["anchor_mean"][j]),
               "anchor_kappa_pooled": float(point["anchor_pooled"][j]),
               "anchor_ci": s2.ci([r["anchor_mean"][j] for r in reps])}
        for est in s2.ESTIMATORS:
            for c in CONDS:
                v = point[est][c][:, j]
                row[f"{c}_{est}_mean"] = float(np.nanmean(v))
                row[f"{c}_{est}_per_model"] = {t: float(v[i]) for i, t in enumerate(tags)}
        res["per_category"].append(row)
    res["per_category"].sort(key=lambda r: -r["alpha"])

    # report ------------------------------------------------------------------------------------
    for sname in ("reliable_5", "all_40"):
        b = res["sets"][sname]
        a = b["anchor"]
        print(f"\n[{sname}] anchor mean-pairwise {a['mean']['point']:.4f} "
              f"[{a['mean']['ci'][0]:.3f},{a['mean']['ci'][1]:.3f}]  pooled "
              f"{a['pooled']['point']:.4f}")
        for est in s2.ESTIMATORS:
            e = b["estimators"][est]
            f = e["fleet"]
            print(f"  {est:8} text {e['conditions']['text']['mean']:.4f}  P(yes) "
                  f"{e['conditions']['pyes']['mean']:.4f}  Delta {f['delta']:+.4f} "
                  f"[{f['delta_ci'][0]:+.4f},{f['delta_ci'][1]:+.4f}]  (pooled-over-slots "
                  f"convention: text {e['conditions']['text']['pooled_over_slots_convention']:.4f}, "
                  f"P(yes) {e['conditions']['pyes']['pooled_over_slots_convention']:.4f})  P(yes) wins "
                  f"{f['pyes_wins']}/11 ({f['pyes_wins_ci']} CI)")
            for c in CONDS:
                x = e["conditions"][c]
                print(f"           {c:5} above anchor: point {x['above_point']}/11, CI "
                      f"{x['above_ci']}/11, below CI {x['below_ci']}/11; sd "
                      f"{x['between_model_sd']:.4f}, median CI width {x['median_ci_width']:.4f}")
    b = res["sets"]["reliable_5"]["estimators"]
    print(f"\n{'model':32}{'arm':>11}{'text ho':>9}{'P(yes) ho':>10}{'text or':>9}{'P(yes) or':>10}")
    for i, t in enumerate(tags):
        print(f"{t[:32]:32}{gen[t][0]:>11}"
              + "".join(f"{b[e]['conditions'][c]['rows'][i]['kappa']:{w}.3f}"
                        for e in s2.ESTIMATORS for c, w in (("text", 9), ("pyes", 10))))

    res["provenance"] = s2.provenance(
        "fleet.py", args, category_set="reliable_5",
        sources=[args.gen_dir, args.ver_dir, args.hq_csv], secondary_sets=["all_40", "unreliable_35"])
    s2.write_json("fleet.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
