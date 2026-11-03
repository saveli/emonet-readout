#!/usr/bin/env python
"""Graded-digit control: one question per category, read as text or as a probability.

P(yes) scoring differs from the benchmark's text parsing in two ways at once: it asks one
question per category, and it reads a probability instead of a printed answer. This control
separates them. Every (image, category) gets the same one-token query design as P(yes)
(evaluation/verify_eval.py --question-set rate, same answer prefix and prefix cache):

    Rate how strongly <category> is shown, from 0 (absent) to 7 (maximal). Answer with one digit.

and one forward pass yields two readouts:
  argmax    the most probable digit token among "0".."7": what a greedy decode prints (text)
  expected  the expected digit under the renormalised distribution over the eight digit
            tokens (a probability readout, continuous)

Scored on the 500-image subset of the multi-sample runs (seed 3407; the image list is asserted
identical to results_e0_multi's), with the primary (held-out map, fractional ties,
pooled-rater target, mean over slots; in-sample beside it), against the expert reference on the
same 500 images. Also on those images: P(yes) (results_verify_prefill) and the reported parsed-
JSON text arm (results_e0). Paired image bootstrap B=500 of
    expected - argmax    what the probability readout adds within the same question
    P(yes) - expected    yes/no probability vs graded-digit probability
    argmax - text        one question per category vs one 40-key JSON reply, both printed text
Per model: digit mass (share of next-token probability on the eight digit tokens, the
answer-mass analogue) and the pixel cap the run used (MiniCPM-V-4.6 runs at 150000 px on the HPC cluster,
where the native minicpmv4_6 fails on multi-tile images; its P(yes) ran at 1048576, so its
P(yes) - expected carries a resolution confound and is flagged).

Output: results/rate_control.json. controls_table.py adds two rows from it
("Graded digit, argmax (500)", "Graded digit, expected (500)") when it exists, none otherwise.

Usage (after collecting from the HPC cluster):
    rsync -av --exclude='*.partial' the HPC cluster:$SCRATCH/results_rate/ results_rate/
    python analysis/rate_control.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoring as s2  # noqa: E402
from controls_common import ci, pmap  # noqa: E402

N_SUBSET = 500
ARMS = ("argmax", "expected", "pyes", "text")
COMPS = (("expected", "argmax"), ("pyes", "expected"), ("argmax", "text"), ("pyes", "argmax"))
MINICPM = "MiniCPM-V-4.6"


def load_rate(d: Path, short, n):
    """{tag: (expected matrix, argmax matrix, mean digit mass, batch_info)}."""
    out = {}
    for path in sorted(d.glob(f"*__rate_n{N_SUBSET}_eval.json")):
        p = json.load(open(path))
        info = p["batch_info"]
        if info.get("arm") != "rate":
            continue
        tag = info["model"].split(" ")[0]
        ex = np.full((n, len(short)), np.nan)
        am = np.full((n, len(short)), np.nan)
        masses = []
        for r in p["results"]:
            i = r["image_index"]
            for j, e in enumerate(short):
                ex[i, j] = r["predicted_emotions"][e]
                am[i, j] = r["argmax_emotions"][e]
                masses.append(r["digit_mass"][e])
        out[tag] = (ex, am, float(np.mean(masses)), info)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    s2.add_args(ap)
    ap.add_argument("--rate-dir", type=Path, default=Path("results_rate"))
    ap.add_argument("--multi-dir", type=Path, default=Path("results_e0_multi"))
    args = ap.parse_args()

    D, gen, ver, head = s2.headline(args)
    s2.assert_headline(head)
    rel, n, short = D["rel"], D["n"], D["short"]

    runs = load_rate(args.rate_dir, short, n)
    if not runs:
        raise SystemExit(f"no *__rate_n{N_SUBSET}_eval.json in {args.rate_dir}")
    # the image list: identical across rate runs and to the multi-sample subset
    multi = sorted(args.multi_dir.glob(f"*__ours_multi_k16_n{N_SUBSET}_eval.json"))
    if not multi:
        raise SystemExit(f"no multi-sample n{N_SUBSET} file in {args.multi_dir} to pin the subset")
    sub = np.array(sorted(json.load(open(multi[0]))["batch_info"]["image_indices"]))
    for t, (_, _, _, info) in runs.items():
        got = np.array(sorted(info["image_indices"]))
        if len(got) != N_SUBSET or not np.array_equal(got, sub):
            raise SystemExit(f"STOP: {t} rate run covers a different image list than the "
                             f"multi-sample subset")
        if (info.get("limit_mode"), info.get("subset_seed")) != ("random", 3407):
            raise SystemExit(f"STOP: {t} rate run subset {info.get('limit_mode')}/"
                             f"{info.get('subset_seed')}, expected random/3407")
    unknown = set(runs) - set(ver)
    if unknown:
        raise SystemExit(f"rate runs for models outside the eleven: {unknown}")
    tags = sorted(runs)
    insub = np.zeros(n, bool)
    insub[sub] = True

    def on_sub(m):
        m = m.astype(np.float64).copy()
        m[~insub] = np.nan
        return m

    mats = {t: {"argmax": on_sub(runs[t][1]), "expected": on_sub(runs[t][0]),
                "pyes": on_sub(ver[t]), "text": on_sub(gen[t][1])} for t in tags}
    print(f"rate runs: {len(tags)} of 11 on {len(sub)} images: {', '.join(tags)}")

    def score(Sm):
        o = {"anchor": Sm.anchor(rel)}
        for est in s2.ESTIMATORS:
            o[est] = np.array([[Sm.model(mats[t][a], rel, est) for a in ARMS] for t in tags])
        return o

    P = score(s2.Sample(D, sub, S=args.S))
    dr = [np.sort(sub[d]) for d in s2.draws(len(sub), args.replicates, args.seed)]
    R = pmap(lambda b: score(s2.Sample(D, dr[b], rep=b, S=args.S)), list(range(args.replicates)),
             args.workers)
    A = np.array([r["anchor"] for r in R])
    ia = {a: i for i, a in enumerate(ARMS)}
    res = {"n_images": int(len(sub)), "image_indices": sub.tolist(), "models": tags,
           "n_models": len(tags), "anchor_subset": P["anchor"], "anchor_subset_ci": ci(A),
           "prompt": runs[tags[0]][3].get("prompt_text"), "estimators": {}}
    for est in s2.ESTIMATORS:
        Pe = P[est]
        Re = np.stack([r[est] for r in R])                       # (B, models, arms)
        rows = []
        for i, t in enumerate(tags):
            info = runs[t][3]
            row = {"model": t, "digit_mass": runs[t][2],
                   "rate_pixels": info.get("max_image_pixels"),
                   "text_arm": gen[t][0],
                   "resolution_confound": t == MINICPM}
            for a in ARMS:
                row[f"kappa_{a}"] = float(Pe[i, ia[a]])
                lo, hi = ci(Re[:, i, ia[a]] - A)
                row[f"{a}_vs_anchor"] = s2.verdict(lo, hi)
            for a, b in COMPS:
                row[f"{a}_minus_{b}"] = float(Pe[i, ia[a]] - Pe[i, ia[b]])
                row[f"{a}_minus_{b}_ci"] = ci(Re[:, i, ia[a]] - Re[:, i, ia[b]])
            rows.append(row)
        fleet = {}
        for a in ARMS:
            fleet[f"kappa_{a}"] = float(Pe[:, ia[a]].mean())
            fleet[f"{a}_above_point"] = int((Pe[:, ia[a]] > P["anchor"]).sum())
            fleet[f"{a}_above_ci"] = int(sum(r[f"{a}_vs_anchor"] == "above" for r in rows))
            fleet[f"{a}_below_ci"] = int(sum(r[f"{a}_vs_anchor"] == "below" for r in rows))
        for a, b in COMPS:
            fleet[f"{a}_minus_{b}"] = float((Pe[:, ia[a]] - Pe[:, ia[b]]).mean())
            fleet[f"{a}_minus_{b}_ci"] = ci((Re[:, :, ia[a]] - Re[:, :, ia[b]]).mean(1))
            fleet[f"{a}_minus_{b}_sig_pos"] = int(sum(r[f"{a}_minus_{b}_ci"][0] > 0 for r in rows))
            fleet[f"{a}_minus_{b}_sig_neg"] = int(sum(r[f"{a}_minus_{b}_ci"][1] < 0 for r in rows))
        fleet["digit_mass_mean"] = float(np.mean([r["digit_mass"] for r in rows]))
        res["estimators"][est] = {"rows": rows, "fleet": fleet}

    f = res["estimators"]["heldout"]["fleet"]
    print(f"\nsubset anchor {P['anchor']:.4f}; held-out fleet: " + "  ".join(
        f"{a} {f[f'kappa_{a}']:.3f} ({f[f'{a}_above_point']}/{f[f'{a}_above_ci']})" for a in ARMS))
    for a, b in COMPS:
        c = f[f"{a}_minus_{b}_ci"]
        print(f"  {a} - {b}: {f[f'{a}_minus_{b}']:+.4f} [{c[0]:+.4f},{c[1]:+.4f}]  "
              f"sig +{f[f'{a}_minus_{b}_sig_pos']}/-{f[f'{a}_minus_{b}_sig_neg']}")
    for r in res["estimators"]["heldout"]["rows"]:
        print(f"  {r['model'][:30]:30} mass {r['digit_mass']:.3f} px {r['rate_pixels']}  argmax "
              f"{r['kappa_argmax']:.3f} expected {r['kappa_expected']:.3f} P(yes) "
              f"{r['kappa_pyes']:.3f} text {r['kappa_text']:.3f}"
              + ("  [resolution confound]" if r["resolution_confound"] else ""))

    # two controls-table rows: text column = the graded readout, P(yes) column = P(yes)
    def rows_for(est):
        fe = res["estimators"][est]["fleet"]
        out = {}
        for a, lab in (("argmax", "Graded digit, argmax (500)"),
                       ("expected", "Graded digit, expected (500)")):
            out[a] = {"label": lab, "n_models": len(tags), "anchor": P["anchor"],
                      "text": fe[f"kappa_{a}"], "pyes": fe["kappa_pyes"],
                      "delta": fe[f"pyes_minus_{a}"], "delta_ci": fe[f"pyes_minus_{a}_ci"],
                      "pyes_above": fe["pyes_above_point"], "pyes_above_ci": fe["pyes_above_ci"],
                      "text_above": fe[f"{a}_above_point"], "text_above_ci": fe[f"{a}_above_ci"]}
        return out
    res["controls_rows"] = {est: rows_for(est) for est in s2.ESTIMATORS}
    res["provenance"] = s2.provenance("rate_control.py", args,
                                      sources=[args.gen_dir, args.ver_dir, args.rate_dir,
                                               args.multi_dir], subset_n=N_SUBSET,
                                      subset_seed=3407)
    s2.write_json("rate_control.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
