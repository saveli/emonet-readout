#!/usr/bin/env python
"""How much of either arm's score is the calibration? Raw, oracle and held-out, both arms.

An alternative explanation reads the per-category rank-match onto 0--7 as a source of the
verification gain. It is applied identically to both arms, and this script shows what each arm
scores under three mappings, with the paired contrast under each:

  raw        no label information at all. Generative: the emitted integer, clipped to 0..7.
             Verification: round(7 * P(yes)), the only label-free linear map from [0, 1] onto
             the scale (P(yes) rounded directly would collapse to 0/1, which is the
             thresholded control, reported elsewhere).
  oracle     the published path: rank-matched to the gold marginal of the scored images
             (`quantile_bin`), refit inside every bootstrap replicate.
  held-out   images split in half with seed 3407, a mapping fit on one half's predictions
             and gold marginal, applied to the other half, both directions, mean of the two
             kappas. Nothing from the scored half's labels reaches its mapping. Two fits:

    heldout_cut     the protocol of `e1_heldout_calibration.py`, which produced
                    `verify_prefill_heldout_calibration_all11.json`: score cutpoints at the
                    fit half's empirical quantiles, `searchsorted(side="right")`. Correct for
                    continuous P(yes); NOT for the generative arm, whose integer outputs tie
                    massively: when a cut lands on a tied value (it does whenever more images
                    share a value than the level it should fill, e.g. the 0 block), the whole
                    tied block is pushed above the cut, so every emitted 0 becomes level >= 1.
                    The generative collapse under this fit (0.410 -> 0.283) is that artifact,
                    not calibration leakage; it is kept and labelled so nobody re-derives it.
    heldout         tie-aware, used for both arms and in the controls table: rank-match the
                    fit half exactly as `quantile_bin` does, give each distinct fit value the
                    median level its images received, and map a test value to the level of
                    the nearest fit value at or below it (lowest level below the fit range).
                    For untied P(yes) this reduces to the cutpoint rule up to where inside a
                    gap the cut sits; the verification column moves by at most a few 0.001.

Paired image bootstrap (B=500, shared draws; the held-out split is a fixed property of each
image, so a replicate fits on its resampled A-half and scores its resampled B-half).

Guards: printed headline figures (controls_common.reproduce); the verification held-out
column must equal `verify_prefill_heldout_calibration_all11.json` per model; the generative raw
mean must land on the audit's 0.393.

Usage:
    python analysis/calibration_both_arms.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controls_common import add_common_args, ci, f3, fci, pmap, tex_name  # noqa: E402
from reliability_vs_performance import N_LEVELS  # noqa: E402


def raw_map(arm, p):
    if arm == "generative":
        return np.clip(np.round(p), 0, N_LEVELS - 1)
    return np.clip(np.round(p * (N_LEVELS - 1)), 0, N_LEVELS - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out", type=Path, default=Path("results/calibration_both_arms.json"))
    ap.add_argument("--tex", type=Path, default=Path("analysis/tables/calibration_both_arms.tex"))
    args = ap.parse_args()
    return compute(args)


def compute(args) -> int:
    """Raw scale vs in-sample (oracle) vs held-out map, both conditions.

    raw       no label information: text clip(round(x), 0, 7); P(yes) clip(round(7 p), 0, 7)
              (one label-free choice among several, not the unique one)
    oracle    the tie-aware map fit on the scored images
    heldout   the same map family, cross-fit over S=20 two-fold splits, scored on the same
              images with the same kappa: oracle - heldout is label access alone (J3), so it is
              reported as `leakage` with a paired CI.
    Output: results/calibration_both_arms.json, tables/calibration_both_arms.tex.
    """
    import scoring as s2

    D, gen, ver, head = s2.headline(args)
    s2.assert_headline(head)
    tags = sorted(ver)
    rel, n = D["rel"], D["n"]
    mats = {"text": {t: gen[t][1] for t in tags}, "pyes": ver}
    MAPS = ("raw", "oracle", "heldout")
    arm_of = {"text": "generative", "pyes": "verification"}

    def score(Sm):
        o = {"anchor": Sm.anchor(rel, "mean")}
        for c, ms in mats.items():
            for t in tags:
                o[(c, t, "raw")] = float(np.nanmean([s2.kappa_slots(
                    raw_map(arm_of[c], ms[t][Sm.rows, j]), Sm.H[:, j, :]) for j in rel]))
                for est in s2.ESTIMATORS:
                    o[(c, t, est)] = Sm.model(ms[t], rel, est)
        return o

    point = score(s2.Sample(D, np.arange(n), S=args.S))
    dr = s2.draws(n, args.replicates, args.seed)
    print(f"paired image bootstrap B={args.replicates}, S={args.S}, {args.workers} workers",
          flush=True)
    reps = pmap(lambda b: score(s2.Sample(D, dr[b], rep=b, S=args.S)),
                list(range(args.replicates)), args.workers)

    def S(k):
        return np.array([r[k] for r in reps])

    anchor = point["anchor"]
    rows, fleet = [], {}
    for t in tags:
        row = {"model": t, "text_arm": gen[t][0]}
        for mp_ in MAPS:
            g, v = point[("text", t, mp_)], point[("pyes", t, mp_)]
            row[mp_] = {"text": g, "pyes": v, "delta": v - g,
                        "delta_ci": ci(S(("pyes", t, mp_)) - S(("text", t, mp_))),
                        "pyes_minus_anchor_ci": ci(S(("pyes", t, mp_)) - S("anchor")),
                        "text_minus_anchor_ci": ci(S(("text", t, mp_)) - S("anchor"))}
        for c in mats:
            row[f"leakage_{c}"] = point[(c, t, "oracle")] - point[(c, t, "heldout")]
            row[f"leakage_{c}_ci"] = ci(S((c, t, "oracle")) - S((c, t, "heldout")))
        rows.append(row)
    for mp_ in MAPS:
        g = np.mean([r[mp_]["text"] for r in rows])
        v = np.mean([r[mp_]["pyes"] for r in rows])
        dd = np.mean([S(("pyes", t, mp_)) - S(("text", t, mp_)) for t in tags], 0)
        fleet[mp_] = {
            "text": float(g), "pyes": float(v), "delta": float(v - g), "delta_ci": ci(dd),
            "pyes_wins": int(sum(r[mp_]["delta"] > 0 for r in rows)),
            "pyes_wins_ci": int(sum(r[mp_]["delta_ci"][0] > 0 for r in rows)),
            "pyes_point_above_anchor": int(sum(r[mp_]["pyes"] > anchor for r in rows)),
            "pyes_ci_above_anchor": int(sum(r[mp_]["pyes_minus_anchor_ci"][0] > 0 for r in rows)),
            "text_point_above_anchor": int(sum(r[mp_]["text"] > anchor for r in rows)),
            "text_ci_above_anchor": int(sum(r[mp_]["text_minus_anchor_ci"][0] > 0 for r in rows)),
        }
    for c in mats:
        fleet[f"leakage_{c}"] = float(np.mean([r[f"leakage_{c}"] for r in rows]))
        fleet[f"leakage_{c}_ci"] = ci(np.mean([S((c, t, "oracle")) - S((c, t, "heldout"))
                                               for t in tags], 0))
    for mp_ in MAPS:
        f = fleet[mp_]
        print(f"{mp_:8} text {f['text']:.4f} P(yes) {f['pyes']:.4f} Delta {f['delta']:+.4f} "
              f"[{f['delta_ci'][0]:+.4f},{f['delta_ci'][1]:+.4f}]  P(yes) wins {f['pyes_wins']} "
              f"({f['pyes_wins_ci']}); above anchor P(yes) {f['pyes_point_above_anchor']} "
              f"({f['pyes_ci_above_anchor']}), text {f['text_point_above_anchor']} "
              f"({f['text_ci_above_anchor']})")
    for c in mats:
        print(f"leakage (oracle - heldout) {c}: {fleet[f'leakage_{c}']:+.4f} "
              f"[{fleet[f'leakage_{c}_ci'][0]:+.4f},{fleet[f'leakage_{c}_ci'][1]:+.4f}]")

    lines = ["% generated by analysis/calibration_both_arms.py -- do not edit by hand",
             r"\begin{tabular}{lccccccccc}", r"\toprule",
             r"Model & \multicolumn{3}{c}{raw scale} & \multicolumn{3}{c}{in-sample map} & "
             r"\multicolumn{3}{c}{held-out map} \\",
             r" & text & $P(\mathrm{yes})$ & $\Delta$ & text & $P(\mathrm{yes})$ & $\Delta$ & "
             r"text & $P(\mathrm{yes})$ & $\Delta$ \\", r"\midrule"]
    for r in rows:
        lines.append(tex_name(r["model"]) + " & " + " & ".join(
            f"{f3(r[m]['text'])} & {f3(r[m]['pyes'])} & {f3(r[m]['delta'], True)}"
            for m in MAPS) + r" \\")
    lines += [r"\midrule", "Mean & " + " & ".join(
        f"{f3(fleet[m]['text'])} & {f3(fleet[m]['pyes'])} & {f3(fleet[m]['delta'], True)}"
        for m in MAPS) + r" \\",
        r"$\Delta$ 95\% CI & " + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{{fci(*fleet[m]['delta_ci'])}}}" for m in MAPS) + r" \\",
        r"\bottomrule", r"\end{tabular}", ""]
    s2.write_json("calibration_both_arms.json", {
        "anchor": anchor, "rows": rows, "fleet": fleet,
        "raw_definition": {"text": "clip(round(x), 0, 7)", "pyes": "clip(round(7 * P(yes)), 0, 7)"},
        "provenance": s2.provenance("calibration_both_arms.py", args,
                                    sources=[args.gen_dir, args.ver_dir, args.hq_csv])})
    s2.write_tex(s2.TABLES_DIR / "calibration_both_arms.tex", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
