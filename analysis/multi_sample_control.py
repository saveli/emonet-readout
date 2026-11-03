#!/usr/bin/env python
"""Multi-sample control: does averaging K sampled text answers close the gap to P(yes)?

Alternative explanation: the generative arm is one sample; sample it K times per image, average the
parsed scores, and the text readout should catch up with the logit readout. The
`ours_multi` arm runs on the HPC cluster (K=16 sampled completions per image, 500 fixed images, seed
3407, the same images for every model). This script collects and scores it:

  1. rsync the finished multi files (never `.partial`), aggregate them with
     evaluation/score_multi_sample.py (--check-ours results_e0 must pass), the K-curve with
     --take 1/2/4/8, and run analysis/e0_report.py on mean_k16/ and sample0/.
  2. Per model, on the 500-image subset, kappa_w (reliable five; all 40 secondary) through the
     same quantile mapping as every other figure, of
       sample0  one sample of the same decoder: the paired single-sample reference (matters
                for Qwen2.5-VL and Ministral, whose near-greedy cards are overridden to T=1.0)
       mean16   mean of the K=16 parsed samples
       ours     the paper's `ours` text-parsing arm (results_e0), restricted to these images
       ver      verification P(yes) (results_verify_prefill), restricted to these images
     Each arm is scored over the images it parsed, as everywhere else.
  3. Paired image bootstrap over the 500 images (B=500, seed-0 draws, calibration refit in
     every replicate) of mean16 - sample0, ver - mean16 and mean16 - ours, per model and fleet;
     how many models clear the expert reference (published 0.468 by point estimate; paired CI
     against the reference recomputed on the same resample); the K-curve K=1,2,4,8,16.

Guards (before anything is written):
  - the paper's printed headline figures (controls_common.reproduce);
  - K=1 (--take 1) reproduces sample0 exactly, per model;
  - sample0 / mean16 kappa equal what e0_report.py writes for those directories;
  - the `ours` 500-image kappa equals e0_report.py run on the restricted `ours` files.
  Reported, not asserted: verification on the subset vs its published full-set figure.

Outputs: results/multi_sample_control.json, analysis/tables/multi_sample_control.tex. The
multi_sample row of analysis/controls_table.py reads the JSON.

Usage:
    python analysis/multi_sample_control.py               # fetch, score, analyse
    python analysis/multi_sample_control.py --no-fetch    # results_e0_multi/ as it is
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controls_common import Drift, add_common_args, ci, f3, fci, pmap, tex_name  # noqa: E402
from e0_report import load_arm_results  # noqa: E402

REMOTE = "the HPC cluster:$SCRATCH/results_e0_multi/"
K_FULL = 16
N_SUBSET = 500   # the fixed random subset (seed 3407); the full-2500 runs carry no _n suffix
K_CURVE = (1, 2, 4, 8, 16)
N_MODELS_TOTAL = 11
EXACT = 1e-9


def run(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def collect(args):
    py = sys.executable
    if not args.no_fetch:
        run(["rsync", "-a", "--exclude=*.partial", REMOTE, f"{args.multi_dir}/"])
    if args.no_score:
        return
    out = args.scored_dir
    run([py, "evaluation/score_multi_sample.py", "--in-dir", args.multi_dir, "--out-dir", out,
         "--only-n", str(N_SUBSET),
         "--check-ours", args.gen_dir])
    for k in K_CURVE[:-1]:
        run([py, "evaluation/score_multi_sample.py", "--in-dir", args.multi_dir,
             "--out-dir", out, "--only-n", str(N_SUBSET), "--take", k])
    for sub in (f"mean_k{K_FULL}", "sample0"):
        run([py, "analysis/e0_report.py", "--results-dir", out / sub, "--hq-csv", args.hq_csv,
             "--out", out / f"e0_report_{sub}.json"])


def same(a, b):
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--multi-dir", type=Path, default=Path("results_e0_multi"))
    ap.add_argument("--scored-dir", type=Path, default=None,
                    help="default results_e0_multi_scored (500) / results_e0_multi_scored_full "
                         "(2500): separate, because load_arm_results keeps the larger run of a "
                         "(model, arm) and would mix the two sizes in one directory")
    ap.add_argument("--n", type=int, choices=(500, 2500), default=500,
                    help="500 = the fixed seed-3407 subset (reference on those images); "
                         "2500 = the full set (reference = the full-set anchor of fleet.json). "
                         "Outputs for 2500 carry a _full suffix.")
    ap.add_argument("--no-fetch", action="store_true", help="skip the rsync from the HPC cluster")
    ap.add_argument("--no-score", action="store_true",
                    help="skip score_multi_sample.py / e0_report.py (use what is on disk)")
    ap.add_argument("--out", type=Path, default=Path("results/multi_sample_control.json"))
    ap.add_argument("--tex", type=Path, default=Path("analysis/tables/multi_sample_control.tex"))
    args = ap.parse_args()
    if args.scored_dir is None:
        args.scored_dir = Path("results_e0_multi_scored" + ("_full" if args.n == 2500 else ""))
    return compute(args)


def compute(args) -> int:
    """Multi-sample control, on the 500 fixed images (--only-n 500).

    Arms per model, all restricted to the subset and each scored over the images it parsed:
      sample0   first of the K samples, same prompt (`ours`) and decoder as mean16: the paired
                single-sample reference for "does averaging help"
      mean16    mean of the parsed samples, K=16
      reported  the text condition the paper reports for this model (paper_once or ours,
                chosen under the held-out estimator, J6), from results_e0, greedy/card decode
      ours      results_e0 `ours` (= reported where reported == ours)
      pyes      verification P(yes)
    Tests (paired image bootstrap over the subset, splits redrawn per replicate):
      mean16 - sample0, mean16 - reported, pyes - mean16, pyes - reported.
    Counts above the expert reference use the SUBSET anchor (mean pairwise kappa on the same
    500 images) for both the point and the CI count. Temperature overrides are recorded per model.
    Output: results/multi_sample_control.json, tables/multi_sample_control.tex.
    """
    import scoring as s2

    global N_SUBSET
    N_SUBSET = args.n
    full_set = args.n == 2500
    sfx = "_full" if full_set else ""
    if not args.no_score:
        collect(args)
    D, gen, ver, head = s2.headline(args)
    s2.assert_headline(head)
    rel, all40, n, short = D["rel"], D["all40"], D["n"], D["short"]

    multi = {}
    pattern = (f"*__ours_multi_k{K_FULL}_eval.json" if full_set
               else f"*__ours_multi_k{K_FULL}_n{N_SUBSET}_eval.json")
    for path in sorted(args.multi_dir.glob(pattern)):
        info = json.load(open(path))["batch_info"]
        multi[info["model_id"].split("/")[-1]] = info
    assert multi, f"no finished ours_multi n{N_SUBSET} runs in {args.multi_dir}"
    subsets = {tuple(sorted(i["image_indices"])) for i in multi.values()}
    if len(subsets) != 1:
        raise Drift("the multi runs do not share one image list")
    sub = np.array(next(iter(subsets)))
    if len(sub) != N_SUBSET or len(np.unique(sub)) != N_SUBSET:
        raise Drift(f"subset has {len(sub)} entries, {len(np.unique(sub))} distinct; want {N_SUBSET}")
    if full_set and not np.array_equal(sub, np.arange(n)):
        raise Drift("full-set multi runs do not cover images 0..2499")
    seeds = {i.get("limit_seed") for i in multi.values()}
    if not full_set and seeds != {3407}:
        raise Drift(f"subset seeds {seeds}, want {{3407}}")
    tags = sorted(multi)
    if set(tags) - set(ver):
        raise Drift(f"multi runs outside the eleven: {set(tags) - set(ver)}")
    print(f"multi-sample runs: {len(tags)} of {N_MODELS_TOTAL} on {len(sub)} images: {', '.join(tags)}")

    def load_scored(subdir):
        got = {t: m.astype(np.float64) for (t, arm), (m, _) in
               load_arm_results(args.scored_dir / subdir, short, n).items()}
        missing = set(tags) - set(got)
        if missing:
            raise Drift(f"{args.scored_dir / subdir} lacks {missing}; rerun without --no-score")
        return got

    curve = {k: load_scored(f"mean_k{k}") for k in K_CURVE}
    s0 = load_scored("sample0")
    for t in tags:
        if not same(curve[1][t], s0[t]):
            raise Drift(f"{t}: --take 1 mean does not reproduce sample0")
    arms_e0 = load_arm_results(args.gen_dir, short, n)
    insub = np.zeros(n, bool)
    insub[sub] = True

    def on_sub(m):
        m = m.astype(np.float64).copy()
        m[~insub] = np.nan
        return m

    ARMS2 = ("sample0", "mean16", "reported", "ours", "pyes")
    mats = {t: {"sample0": on_sub(s0[t]), "mean16": on_sub(curve[K_FULL][t]),
                "reported": on_sub(gen[t][1]), "ours": on_sub(arms_e0[(t, "ours")][0]),
                "pyes": on_sub(ver[t])} for t in tags}
    COMPS = (("mean16", "sample0"), ("mean16", "reported"), ("pyes", "mean16"),
             ("pyes", "reported"), ("mean16", "ours"))
    cats = {"reliable_5": rel, "all_40": all40}

    def score(Sm):
        out = {}
        for cn, cs in cats.items():
            out[cn] = {"anchor": Sm.anchor(cs, "mean")}
            for est in s2.ESTIMATORS:
                out[cn][est] = np.array([[Sm.model(mats[t][a], cs, est) for a in ARMS2]
                                         for t in tags])
                if cn == "reliable_5":
                    out[cn][f"curve_{est}"] = np.array(
                        [[Sm.model(on_sub(curve[k][t]), cs, est) for k in K_CURVE] for t in tags])
        return out

    point = score(s2.Sample(D, sub, S=args.S))
    dr = [np.sort(sub[d]) for d in s2.draws(len(sub), args.replicates, args.seed)]
    print(f"paired image bootstrap over the {len(sub)} images, B={args.replicates}, S={args.S}, "
          f"{args.workers} workers", flush=True)
    reps = pmap(lambda b: score(s2.Sample(D, dr[b], rep=b, S=args.S)),
                list(range(args.replicates)), args.workers)

    ia = {a: i for i, a in enumerate(ARMS2)}
    result = {}
    for cn in cats:
        anc = point[cn]["anchor"]
        anc_r = np.array([r[cn]["anchor"] for r in reps])
        result[cn] = {"anchor_subset": anc, "anchor_subset_ci": ci(anc_r), "estimators": {}}
        for est in s2.ESTIMATORS:
            P = point[cn][est]
            R = np.stack([r[cn][est] for r in reps])
            rows = []
            for i, t in enumerate(tags):
                info = multi[t]
                smp = info["sampling"]
                r = {"model": t, "reported_arm": gen[t][0],
                     "temperature": smp.get("temperature"),
                     "card_temperature": smp.get("card_temperature", smp.get("temperature")),
                     "temperature_overridden": bool(smp.get("temperature_override")),
                     "samples_parsed": info.get("samples_parsed_total"),
                     "samples_total": info.get("samples_total")}
                for a in ARMS2:
                    r[f"kappa_{a}"] = float(P[i, ia[a]])
                    r[f"n_parsed_{a}"] = int((~np.isnan(mats[t][a][sub, rel[0]])).sum())
                    lo, hi = ci(R[:, i, ia[a]] - anc_r)
                    r[f"{a}_above_anchor_point"] = bool(P[i, ia[a]] > anc)
                    r[f"{a}_vs_anchor"] = s2.verdict(lo, hi)
                for a, b in COMPS:
                    r[f"{a}_minus_{b}"] = float(P[i, ia[a]] - P[i, ia[b]])
                    r[f"{a}_minus_{b}_ci"] = ci(R[:, i, ia[a]] - R[:, i, ia[b]])
                if cn == "reliable_5":
                    r["k_curve"] = {str(k): float(point[cn][f"curve_{est}"][i, j])
                                    for j, k in enumerate(K_CURVE)}
                rows.append(r)
            fleet = {"n_models": len(tags)}
            for a in ARMS2:
                fleet[f"kappa_{a}"] = float(P[:, ia[a]].mean())
                fleet[f"{a}_above_anchor_point"] = int((P[:, ia[a]] > anc).sum())
                fleet[f"{a}_above_anchor_ci"] = int(sum(r[f"{a}_vs_anchor"] == "above" for r in rows))
                fleet[f"{a}_below_anchor_ci"] = int(sum(r[f"{a}_vs_anchor"] == "below" for r in rows))
            for a, b in COMPS:
                d = R[:, :, ia[a]] - R[:, :, ia[b]]
                fleet[f"{a}_minus_{b}"] = float((P[:, ia[a]] - P[:, ia[b]]).mean())
                fleet[f"{a}_minus_{b}_ci"] = ci(d.mean(1))
                fleet[f"{a}_minus_{b}_sig_pos"] = int(sum(r[f"{a}_minus_{b}_ci"][0] > 0 for r in rows))
                fleet[f"{a}_minus_{b}_sig_neg"] = int(sum(r[f"{a}_minus_{b}_ci"][1] < 0 for r in rows))
            if cn == "reliable_5":
                C = np.stack([r[cn][f"curve_{est}"] for r in reps])
                pc = point[cn][f"curve_{est}"]
                fleet["k_curve"] = [{"K": k, "kappa": float(pc[:, j].mean()),
                                     "minus_sample0": float((pc[:, j] - pc[:, 0]).mean()),
                                     "minus_sample0_ci": ci((C[:, :, j] - C[:, :, 0]).mean(1))}
                                    for j, k in enumerate(K_CURVE)]
            result[cn]["estimators"][est] = {"rows": rows, "fleet": fleet}

    for cn in cats:
        print(f"\n[{cn}] subset anchor {result[cn]['anchor_subset']:.4f}")
        for est in s2.ESTIMATORS:
            f = result[cn]["estimators"][est]["fleet"]
            print(f"  {est:8}" + " ".join(f"{a} {f[f'kappa_{a}']:.3f}" for a in ARMS2))
            for a, b in COMPS:
                c = f[f"{a}_minus_{b}_ci"]
                print(f"           {a}-{b}: {f[f'{a}_minus_{b}']:+.4f} [{c[0]:+.4f},{c[1]:+.4f}] "
                      f"sig +{f[f'{a}_minus_{b}_sig_pos']}/-{f[f'{a}_minus_{b}_sig_neg']}")
            print("           above subset anchor point (CI above / CI below): " + "  ".join(
                f"{a} {f[f'{a}_above_anchor_point']} ({f[f'{a}_above_anchor_ci']}/"
                f"{f[f'{a}_below_anchor_ci']})" for a in ARMS2))
    hr = result["reliable_5"]["estimators"]["heldout"]
    print("\nper model, held-out, reliable five:")
    for r in hr["rows"]:
        print(f"  {r['model'][:30]:30} T={r['temperature']} card={r['card_temperature']} "
              f"{'OVERRIDE ' if r['temperature_overridden'] else ''}rep={r['reported_arm']:10} "
              f"s0 {r['kappa_sample0']:.3f} m16 {r['kappa_mean16']:.3f} rep {r['kappa_reported']:.3f} "
              f"pyes {r['kappa_pyes']:.3f}  m16-s0 {r['mean16_minus_sample0']:+.3f} "
              f"m16-rep {r['mean16_minus_reported']:+.3f} pyes-m16 {r['pyes_minus_mean16']:+.3f}")
    print("K-curve: " + "  ".join(f"K={c['K']} {c['kappa']:.3f}" for c in hr["fleet"]["k_curve"]))

    f = hr["fleet"]
    m = len(tags)
    lines = ["% generated by analysis/multi_sample_control.py -- do not edit by hand",
             f"% Reliable five, {len(sub)} fixed images, {m} of {N_MODELS_TOTAL} models, held-out "
             f"mapping, paired image bootstrap B={args.replicates}. $^\\ddagger$ sampled at T=1.0 "
             "instead of a near-greedy card temperature.",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Model & reported & 1 sample & mean of 16 & $\Delta$ vs 1 sample [95\% CI] & "
             r"$P(\mathrm{yes})$ & $P(\mathrm{yes})-$mean [95\% CI] \\", r"\midrule"]
    for r in hr["rows"]:
        dag = "$^\\ddagger$" if r["temperature_overridden"] else ""
        lines.append(
            f"{tex_name(r['model'])}{dag} & {f3(r['kappa_reported'])} & {f3(r['kappa_sample0'])} & "
            f"{f3(r['kappa_mean16'])} & {f3(r['mean16_minus_sample0'], True)} "
            f"{fci(*r['mean16_minus_sample0_ci'])} & {f3(r['kappa_pyes'])} & "
            f"{f3(r['pyes_minus_mean16'], True)} {fci(*r['pyes_minus_mean16_ci'])} \\\\")
    lines += [r"\midrule",
              f"Mean ({m} of {N_MODELS_TOTAL}) & {f3(f['kappa_reported'])} & {f3(f['kappa_sample0'])} & "
              f"{f3(f['kappa_mean16'])} & {f3(f['mean16_minus_sample0'], True)} "
              f"{fci(*f['mean16_minus_sample0_ci'])} & {f3(f['kappa_pyes'])} & "
              f"{f3(f['pyes_minus_mean16'], True)} {fci(*f['pyes_minus_mean16_ci'])} \\\\",
              r"\bottomrule", r"\end{tabular}", ""]
    if full_set:
        # on all 2500 images the per-sample reference IS the full-set anchor of fleet.json
        F = json.load(open(s2.RESULTS_DIR / "fleet.json"))
        fa = F["sets"]["reliable_5"]["anchor"]["mean"]["point"]
        if abs(result["reliable_5"]["anchor_subset"] - fa) > 1e-12:
            raise Drift(f"full-set reference {result['reliable_5']['anchor_subset']} != "
                        f"fleet.json anchor {fa}")
        for cn in result:
            result[cn]["anchor_is_full_set"] = True
    s2.write_json(f"multi_sample_control{sfx}.json", {
        "k": K_FULL, "k_curve": list(K_CURVE), "n_images": int(len(sub)),
        "image_indices": sub.tolist(), "n_models": m, "n_models_total": N_MODELS_TOTAL,
        "models": tags, "results": result,
        "arms": {"sample0": "first of the K samples (ours prompt, multi-run decoder)",
                 "mean16": f"mean of the parsed samples, K={K_FULL}",
                 "reported": "the paper's reported text condition (results_e0) on these images",
                 "ours": "results_e0 `ours` on these images",
                 "pyes": "verification P(yes) on these images"},
        "provenance": s2.provenance("multi_sample_control.py", args,
                                    sources=[args.gen_dir, args.ver_dir, args.hq_csv,
                                             args.multi_dir], subset_n=N_SUBSET,
                                    subset_seed=3407)})
    s2.write_tex(s2.TABLES_DIR / f"multi_sample_control{sfx}.tex", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
