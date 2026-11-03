#!/usr/bin/env python
"""Supplementary analyses under the primary estimator (fractional ties onto the pooled rater
marginal, held-out cross-fit, mean-over-slots kappa; analysis/scoring.py).

Sections:

  phrasings   three P(yes) query phrasings (results_verify_prefill = v0, results_verify_v1,
              results_verify_v2; all with the answer prefix). Per cell: kappa_w, CI, paired
              test against the expert reference. -> tables/phrasings.tex
  decoding    the reported text arm under greedy and under model-card sampling for every model
              (results_e0 holds 7 greedy + 4 card; results_e0_decode the mirror). Contrast
              under each uniform policy.
  resolution  text parsing at 512x512 (results_e0) vs 1024x1024 (results_e0_px1024), arm fixed
              to the reported one, paired; contrast at each cap held matched (P(yes) at 512:
              results_verify_px512, 10 models; at 1024: results_verify_prefill).
  retry       the benchmark's retry loop, round 1 vs the accepted round, same model, same image
              (paper_retry files, both rounds parsed).
  training    E4 LoRA runs at the last checkpoint (step 608) vs the untrained base and vs the
              permuted-label control, on the fixed 400-image subset with the anchor recomputed
              on that subset. The cross-fit halves are 200 images; the spread of the point
              estimate over the 20 splits is reported per arm.

All: reliable five, paired image bootstrap B=500 (sorted draws, splits redrawn per replicate),
provenance as in scoring. Output: results/supp_<section>.json.

Usage:
    python analysis/supplementary_tables.py [--only phrasings,decoding,...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoring as s2  # noqa: E402
from controls_common import ci, pmap, tex_name  # noqa: E402
from e0_enforcement_paired import load_paired  # noqa: E402
from e0_report import load_arm_results  # noqa: E402
from e1_report import load_verify_results  # noqa: E402
from e4_report import load_runs  # noqa: E402

SECTIONS = ("phrasings", "decoding", "resolution", "retry", "training")
DECODE_DIR = Path("results_e0_decode")
# assembled locally from $SHARED/results_e4 (6 models) and the HPC cluster
# $SCRATCH/results_e4 (InternVL3_5-8B-HF, Qwen3.5-9B)
E4_DIR = Path("results_e4")


def boot(D, rows, fn, args, B=None):
    """Point estimate and replicates of fn(Sample) on image set `rows`."""
    B = B or args.replicates
    point = fn(s2.Sample(D, rows, S=args.S))
    dr = [np.sort(rows[d]) for d in s2.draws(len(rows), B, args.seed)]
    reps = pmap(lambda b: fn(s2.Sample(D, dr[b], rep=b, S=args.S)), list(range(B)), args.workers)
    return point, reps


def stack(reps, key):
    return np.array([r[key] for r in reps])


def verify_mats(d, D):
    return {t: m.astype(np.float64) for t, (m, _) in load_verify_results(d, D["short"], D["n"]).items()}


# ---------------------------------------------------------------------------------------------
def phrasings(D, gen, ver, args):
    rel, n = D["rel"], D["n"]
    dirs = {"v0": args.ver_dir, "v1": Path("results_verify_v1"), "v2": Path("results_verify_v2")}
    mats = {k: verify_mats(d, D) for k, d in dirs.items()}
    tags = sorted(ver)
    assert all(set(m) == set(tags) for m in mats.values()), {k: sorted(m) for k, m in mats.items()}

    def fn(Sm):
        o = {"anchor": Sm.anchor(rel)}
        for k in dirs:
            o[k] = np.array([Sm.model(mats[k][t], rel, "heldout") for t in tags])
        return o

    P, R = boot(D, np.arange(n), fn, args)
    # v0 must be the P(yes) column of fleet.json
    F = json.load(open(s2.RESULTS_DIR / "fleet.json"))
    fv = {r["model"]: r["kappa"] for r in
          F["sets"]["reliable_5"]["estimators"]["heldout"]["conditions"]["pyes"]["rows"]}
    same_setup = F["provenance"]["S"] == args.S
    bad = [t for i, t in enumerate(tags) if same_setup and abs(P["v0"][i] - fv[t]) > 1e-12]
    if bad:
        raise SystemExit(f"STOP: v0 phrasing differs from fleet.json P(yes): {bad}")
    A = stack(R, "anchor")
    rows, widths = [], []
    for i, t in enumerate(tags):
        row = {"model": t}
        for k in dirs:
            Rk = stack(R, k)[:, i]
            lo, hi = ci(Rk)
            dlo, dhi = ci(Rk - A)
            widths.append(hi - lo)
            row[k] = {"kappa": float(P[k][i]), "ci": [lo, hi],
                      "minus_anchor": float(P[k][i] - P["anchor"]), "minus_anchor_ci": [dlo, dhi],
                      "verdict": s2.verdict(dlo, dhi)}
        vals = [row[k]["kappa"] for k in dirs]
        row["range"] = float(max(vals) - min(vals))
        rows.append(row)
    cells = [(r["model"], k, r[k]) for r in rows for k in dirs]
    med_w = float(np.median(widths))
    out = {"anchor": P["anchor"], "anchor_ci": ci(A), "rows": rows,
           "n_cells": len(cells),
           "cells_above_point": int(sum(c[2]["kappa"] > P["anchor"] for c in cells)),
           "cells_above_ci": int(sum(c[2]["verdict"] == "above" for c in cells)),
           "lowest_cell": min(({"model": m, "phrasing": k, **v} for m, k, v in cells),
                              key=lambda c: c["kappa"]),
           "lowest_minus_anchor_ci_lo": min(c[2]["minus_anchor_ci"][0] for c in cells),
           "mean_range": float(np.mean([r["range"] for r in rows])),
           "median_ci_width": med_w,
           "range_exceeds_median_ci_width": [r["model"] for r in rows if r["range"] > med_w]}
    print(f"phrasings: {out['cells_above_point']}/{len(cells)} cells above the reference "
          f"{P['anchor']:.3f} by point, {out['cells_above_ci']} by CI; lowest "
          f"{out['lowest_cell']['model']} {out['lowest_cell']['phrasing']} "
          f"{out['lowest_cell']['kappa']:.3f}; mean range {out['mean_range']:.3f} vs median CI "
          f"width {med_w:.3f}; range > width: {out['range_exceeds_median_ci_width']}")
    for r in rows:
        print(f"  {r['model'][:30]:30}" + "".join(f"  {k} {r[k]['kappa']:.3f} ({r[k]['verdict'][:5]})"
                                               for k in dirs) + f"  range {r['range']:.3f}")
    lines = ["% generated by analysis/supplementary_tables.py (phrasings) -- do not edit by hand",
             "% five categories, primary (held-out); * = paired CI vs expert reference "
             "excludes 0 from above",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Model & \texttt{v0} & \texttt{v1} & \texttt{v2} & range \\", r"\midrule"]
    for r in sorted(rows, key=lambda r: tex_name(r["model"]).casefold()):
        cells_ = [f"${r[k]['kappa']:.3f}{'^{*}' if r[k]['verdict'] == 'above' else ''}$"
                  for k in dirs]
        lines.append(f"{tex_name(r['model'])} & " + " & ".join(cells_) + f" & ${r['range']:.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    s2.write_tex(Path("tables/phrasings.tex"), "\n".join(lines))
    s2.write_tex(s2.TABLES_DIR / "phrasings.tex", "\n".join(lines))
    return out, [args.ver_dir, "results_verify_v1", "results_verify_v2", args.gen_dir]


# ---------------------------------------------------------------------------------------------
def policy(info):
    smp = info.get("sampling") or {}
    pol = smp.get("policy")
    if pol in ("greedy", "card"):
        return pol
    t = smp.get("temperature")
    return "greedy" if (t is None or t == 0) else "card"


def decoding(D, gen, ver, args):
    rel, n = D["rel"], D["n"]
    tags = sorted(ver)
    base = load_arm_results(args.gen_dir, D["short"], n)
    flip = load_arm_results(DECODE_DIR, D["short"], n)
    pol_mats, pol_of = {"greedy": {}, "card": {}}, {}
    for t in tags:
        arm = gen[t][0]
        a, b = base[(t, arm)], flip[(t, arm)]
        pa, pb = policy(a[1]), policy(b[1])
        if {pa, pb} != {"greedy", "card"}:
            raise SystemExit(f"STOP: {t} {arm}: policies {pa}/{pb}, expected one greedy one card")
        pol_mats[pa][t] = a[0].astype(np.float64)
        pol_mats[pb][t] = b[0].astype(np.float64)
        pol_of[t] = pa

    def fn(Sm):
        o = {"anchor": Sm.anchor(rel), "pyes": np.array([Sm.model(ver[t], rel, "heldout") for t in tags])}
        for p in pol_mats:
            o[p] = np.array([Sm.model(pol_mats[p][t], rel, "heldout") for t in tags])
        return o

    P, R = boot(D, np.arange(n), fn, args)
    A = stack(R, "anchor")
    out = {"reported_policy": pol_of, "policies": {}}
    for p in pol_mats:
        Rp, Rv = stack(R, p), stack(R, "pyes")
        out["policies"][p] = {
            "text": float(P[p].mean()), "delta": float((P["pyes"] - P[p]).mean()),
            "delta_ci": ci((Rv - Rp).mean(1)),
            "pyes_wins_ci": int(sum(ci(Rv[:, i] - Rp[:, i])[0] > 0 for i in range(len(tags)))),
            "text_above_point": int((P[p] > P["anchor"]).sum()),
            "text_above_ci": int(sum(ci(Rp[:, i] - A)[0] > 0 for i in range(len(tags)))),
            "per_model": {t: float(P[p][i]) for i, t in enumerate(tags)}}
    d = P["card"] - P["greedy"]
    dR = stack(R, "card") - stack(R, "greedy")
    out["card_minus_greedy"] = {t: {"d": float(d[i]), "ci": ci(dR[:, i])} for i, t in enumerate(tags)}
    out["max_abs_card_minus_greedy"] = float(np.abs(d).max())
    out["mean_card_minus_greedy"] = float(d.mean())
    out["mean_card_minus_greedy_ci"] = ci(dR.mean(1))
    rep = np.array([P["greedy"][i] if pol_of[t] == "greedy" else P["card"][i]
                    for i, t in enumerate(tags)])
    out["as_reported_delta"] = float((P["pyes"] - rep).mean())
    for p, x in out["policies"].items():
        print(f"decoding {p:6}: text {x['text']:.4f}  Delta {x['delta']:+.4f} [{x['delta_ci'][0]:+.4f},"
              f"{x['delta_ci'][1]:+.4f}]  P(yes) ahead (CI) {x['pyes_wins_ci']}/11  text > ref "
              f"{x['text_above_point']} ({x['text_above_ci']})")
    print(f"  card - greedy per model: mean {out['mean_card_minus_greedy']:+.4f}, max |d| "
          f"{out['max_abs_card_minus_greedy']:.4f}; as reported Delta {out['as_reported_delta']:+.4f}")
    return out, [args.gen_dir, DECODE_DIR, args.ver_dir]


# ---------------------------------------------------------------------------------------------
def resolution(D, gen, ver, args):
    rel, n = D["rel"], D["n"]
    tags = sorted(ver)
    hi_text = load_arm_results(Path("results_e0_px1024"), D["short"], n)
    t1024 = {t: hi_text[(t, gen[t][0])][0].astype(np.float64) for t in tags}
    v512 = verify_mats(Path("results_verify_px512"), D)
    tags512 = sorted(v512)

    def fn(Sm):
        return {"t512": np.array([Sm.model(gen[t][1], rel, "heldout") for t in tags]),
                "t1024": np.array([Sm.model(t1024[t], rel, "heldout") for t in tags]),
                "v1024": np.array([Sm.model(ver[t], rel, "heldout") for t in tags]),
                "v512": np.array([Sm.model(v512[t], rel, "heldout") for t in tags512])}

    P, R = boot(D, np.arange(n), fn, args)
    i512 = [tags.index(t) for t in tags512]
    T5, T1, V1, V5 = (stack(R, k) for k in ("t512", "t1024", "v1024", "v512"))
    dt = P["t512"] - P["t1024"]
    out = {
        "text_512_minus_1024": float(dt.mean()), "text_512_minus_1024_ci": ci((T5 - T1).mean(1)),
        "text_512_minus_1024_per_model": {t: {"d": float(dt[i]), "ci": ci(T5[:, i] - T1[:, i])}
                                          for i, t in enumerate(tags)},
        "text_cap_sig": int(sum(ci(T5[:, i] - T1[:, i])[0] > 0 or ci(T5[:, i] - T1[:, i])[1] < 0
                                for i in range(len(tags)))),
        "contrast_at_512": float((P["v512"] - P["t512"][i512]).mean()),
        "contrast_at_512_ci": ci((V5 - T5[:, i512]).mean(1)), "n_models_512": len(tags512),
        "models_512": tags512,
        "contrast_at_1024": float((P["v1024"] - P["t1024"]).mean()),
        "contrast_at_1024_ci": ci((V1 - T1).mean(1)),
        "contrast_as_reported": float((P["v1024"] - P["t512"]).mean()),
        "contrast_as_reported_ci": ci((V1 - T5).mean(1)),
        "pyes_512_minus_1024": float((P["v512"] - P["v1024"][i512]).mean()),
        "pyes_512_minus_1024_ci": ci((V5 - V1[:, i512]).mean(1)),
        "text_arm_fixed": {t: gen[t][0] for t in tags}}
    print(f"resolution: text 512-1024 {out['text_512_minus_1024']:+.4f} "
          f"[{out['text_512_minus_1024_ci'][0]:+.4f},{out['text_512_minus_1024_ci'][1]:+.4f}] "
          f"({out['text_cap_sig']} models with CI excl. 0); contrast at 512 "
          f"{out['contrast_at_512']:+.4f} [{out['contrast_at_512_ci'][0]:+.4f},"
          f"{out['contrast_at_512_ci'][1]:+.4f}] ({len(tags512)} models), at 1024 "
          f"{out['contrast_at_1024']:+.4f} [{out['contrast_at_1024_ci'][0]:+.4f},"
          f"{out['contrast_at_1024_ci'][1]:+.4f}], as reported {out['contrast_as_reported']:+.4f}; "
          f"P(yes) 512-1024 {out['pyes_512_minus_1024']:+.4f}")
    return out, [args.gen_dir, "results_e0_px1024", "results_verify_px512", args.ver_dir]


# ---------------------------------------------------------------------------------------------
def retry(D, gen, ver, args):
    rel, n = D["rel"], D["n"]
    paths = sorted(args.gen_dir.glob("*__paper_retry_eval.json"))
    have = {p.name.split("__")[0] for p in paths}
    paths += [p for p in sorted(args.gen_dir.glob("*__paper_retry_eval.json.partial"))
              if p.name.split("__")[0] not in have]
    runs = {}
    for p in paths:
        first, final, compliant, paired, rounds, info = load_paired(p, D["short"], n)
        if info.get("total_images", 0) < 1000 or paired == 0:
            continue
        runs[p.name.split("__")[0]] = (first.astype(np.float64), final.astype(np.float64),
                                       paired, rounds, p.name.endswith(".partial"))
    tags = sorted(runs)

    def fn(Sm):
        return {"first": np.array([Sm.model(runs[t][0], rel, "heldout") for t in tags]),
                "final": np.array([Sm.model(runs[t][1], rel, "heldout") for t in tags])}

    P, R = boot(D, np.arange(n), fn, args)
    Fi, Fl = stack(R, "first"), stack(R, "final")
    rows = []
    for i, t in enumerate(tags):
        lo, hi = ci(Fl[:, i] - Fi[:, i])
        rows.append({"model": t, "paired_images": runs[t][2], "rounds": runs[t][3],
                     "partial": runs[t][4], "kappa_first": float(P["first"][i]),
                     "kappa_final": float(P["final"][i]),
                     "final_minus_first": float(P["final"][i] - P["first"][i]),
                     "ci": [lo, hi], "verdict": s2.verdict(lo, hi)})
    out = {"rows": rows, "mean_final_minus_first": float((P["final"] - P["first"]).mean()),
           "mean_ci": ci((Fl - Fi).mean(1))}
    print(f"retry (final - first round), {len(tags)} models: mean {out['mean_final_minus_first']:+.4f} "
          f"[{out['mean_ci'][0]:+.4f},{out['mean_ci'][1]:+.4f}]")
    for r in sorted(rows, key=lambda r: r["final_minus_first"]):
        print(f"  {r['model'][:30]:30} first {r['kappa_first']:.3f} final {r['kappa_final']:.3f} "
              f"d {r['final_minus_first']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
              f"{r['verdict']}{' (partial run)' if r['partial'] else ''}")
    return out, [args.gen_dir]


# ---------------------------------------------------------------------------------------------
def training(D, gen, ver, args, step=608):
    rel, n, short = D["rel"], D["n"], D["short"]
    runs = load_runs(E4_DIR)
    models = sorted({m for m, a, s in runs if a == "a3_sft_shuf"})

    def mat(scores):
        m = np.full((n, len(short)), np.nan)
        for i, em in scores.items():
            for j, e in enumerate(short):
                v = em.get(e, 0)
                m[i, j] = v if isinstance(v, (int, float)) else 0.0
        return m

    mats, subsets = {}, set()
    for mdl in models:
        base = [k for k in runs if k[0] == mdl and k[1] == "base"]
        assert len(base) >= 1, (mdl, "no base run")
        mats[(mdl, "base")] = mat(runs[sorted(base)[0]][0])
        for arm in ("a1_sft", "a3_sft_shuf"):
            mats[(mdl, arm)] = mat(runs[(mdl, arm, step)][0])
        for k in [(mdl, "base"), (mdl, "a1_sft"), (mdl, "a3_sft_shuf")]:
            subsets.add(tuple(np.flatnonzero(~np.isnan(mats[k][:, rel[0]]))))
    union = np.array(sorted(set().union(*subsets)))
    if len(union) != 400:
        raise SystemExit(f"STOP: E4 runs cover {len(union)} images, expected the fixed 400")
    arms = ("base", "a1_sft", "a3_sft_shuf")

    def fn(Sm):
        o = {"anchor": Sm.anchor(rel)}
        for a in arms:
            o[a] = np.array([Sm.model(mats[(m, a)], rel, "heldout") for m in models])
        return o

    P, R = boot(D, union, fn, args)
    # split spread of the point estimate (200-image halves)
    Sm = s2.Sample(D, union, S=args.S)
    spread = {}
    for m in models:
        for a in arms:
            per = [np.mean([s2.kappa_slots(s2.map_heldout(mats[(m, a)][union, j], Sm.tgt(j), mk),
                                           Sm.H[:, j, :]) for j in rel]) for mk in Sm.masks]
            spread[(m, a)] = (float(np.std(per, ddof=1)), float(np.min(per)), float(np.max(per)))
    A = stack(R, "anchor")
    rows = []
    for i, m in enumerate(models):
        row = {"model": m, "emit": {a: float(np.nanmean((mats[(m, a)][union][:, :] > 0.5).sum(1)))
                                    for a in arms}}
        for a in arms:
            Ra = stack(R, a)[:, i]
            row[a] = {"kappa": float(P[a][i]), "ci": ci(Ra),
                      "split_sd": spread[(m, a)][0], "split_min": spread[(m, a)][1],
                      "split_max": spread[(m, a)][2]}
        for x, y in (("a1_sft", "base"), ("a3_sft_shuf", "base"), ("a1_sft", "a3_sft_shuf")):
            d = stack(R, x)[:, i] - stack(R, y)[:, i]
            lo, hi = ci(d)
            row[f"{x}-{y}"] = {"d": float(P[x][i] - P[y][i]), "ci": [lo, hi],
                               "verdict": s2.verdict(lo, hi)}
        rows.append(row)
    out = {"step": step, "n_images": int(len(union)), "anchor_subset": P["anchor"],
           "anchor_subset_ci": ci(A), "rows": rows,
           "split_sd_max": max(v[0] for v in spread.values()),
           "split_sd_median": float(np.median([v[0] for v in spread.values()])),
           "note": "held-out halves are 200 images; split_sd is the SD of the point estimate "
                   "over the 20 two-fold splits (Monte Carlo spread of the estimator)"}
    print(f"training (step {step}, {len(union)} images, subset anchor {P['anchor']:.3f}); split SD "
          f"median {out['split_sd_median']:.4f}, max {out['split_sd_max']:.4f}")
    for r in rows:
        print(f"  {r['model'][:28]:28} base {r['base']['kappa']:.3f} a1 {r['a1_sft']['kappa']:.3f} "
              f"a3 {r['a3_sft_shuf']['kappa']:.3f} | a1-base {r['a1_sft-base']['d']:+.3f} "
              f"[{r['a1_sft-base']['ci'][0]:+.3f},{r['a1_sft-base']['ci'][1]:+.3f}] "
              f"a3-base {r['a3_sft_shuf-base']['d']:+.3f} [{r['a3_sft_shuf-base']['ci'][0]:+.3f},"
              f"{r['a3_sft_shuf-base']['ci'][1]:+.3f}] a1-a3 {r['a1_sft-a3_sft_shuf']['d']:+.3f} "
              f"[{r['a1_sft-a3_sft_shuf']['ci'][0]:+.3f},{r['a1_sft-a3_sft_shuf']['ci'][1]:+.3f}]")
    return out, [E4_DIR]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    s2.add_args(ap)
    ap.add_argument("--only", default=",".join(SECTIONS))
    args = ap.parse_args()
    D, gen, ver, head = s2.headline(args)
    s2.assert_headline(head)
    for sec in args.only.split(","):
        print(f"\n===== {sec}", flush=True)
        out, sources = globals()[sec](D, gen, ver, args)
        out["provenance"] = s2.provenance(f"supplementary_tables.py {sec}", args,
                                          sources=list(dict.fromkeys(map(str, sources))))
        s2.write_json(f"supp_{sec}.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
