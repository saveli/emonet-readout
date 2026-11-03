#!/usr/bin/env python
"""FACES under the reporting rules: all eleven models, no exclusion.

FACES is six-way forced choice scored by accuracy (argmax), so the mapping never enters; what
matters is the reporting. faces_e3_report.py --gate on drops MiMo-VL-7B-RL from the aggregate
because its free-text answers name several emotions (single-answer gap 0.174). This script
reports all eleven, keeps the gap as a diagnostic column, and gives the fleet mean over the
eleven and over the ten that pass the check, each with a paired person-bootstrap CI.

Draws: the person resamples of faces_e3_report.paired_person_bootstrap (seed 0, B=2000), which
re-seeds per model and so already shares one draw sequence across models; the per-model CIs are
asserted to reproduce faces_e3_report.json exactly, and the fleet means are taken inside the same
draws (paired across models).

Output: results/faces_fleet.json.

Usage:
    python analysis/faces_fleet.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoring as s2  # noqa: E402
from faces_e3_report import load  # noqa: E402

HERE = Path(__file__).resolve().parent


def per_person(records, people):
    idx = {p: i for i, p in enumerate(people)}
    s = np.zeros(len(people))
    c = np.zeros(len(people))
    for r in records:
        i = idx.get(r["person"])
        if i is not None:
            s[i] += r["correct"]
            c[i] += 1
    return s, c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/faces"))
    ap.add_argument("-B", "--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = load(args.results_dir)
    rep = {r["model"]: r for r in json.load(open(HERE / "faces_e3_report.json"))["rows"]}
    gv = json.load(open(HERE / "faces_generative_validity.json"))
    f1 = {r["model"]: r for r in json.load(open(HERE / "faces_macro_f1.json"))["rows"]}
    tags = sorted({t for t, _ in runs})
    assert len(tags) == 11, tags

    data, people_of = {}, {}
    for t in tags:
        ver, gen = runs[(t, "verify")][0], runs[(t, "generative")][0]
        people = sorted({r["person"] for r in ver} & {r["person"] for r in gen})
        people_of[t] = people
        data[t] = (per_person(ver, people), per_person(gen, people))
    assert len({tuple(p) for p in people_of.values()}) == 1, "models cover different persons"
    people = people_of[tags[0]]

    # the same draws as faces_e3_report.paired_person_bootstrap (re-seeded per model)
    rng = np.random.default_rng(args.seed)
    pos = {p: i for i, p in enumerate(people)}
    W = np.zeros((args.replicates, len(people)))
    for b in range(args.replicates):
        draw = rng.choice(people, size=len(people), replace=True)
        W[b] = np.bincount([pos[p] for p in draw], minlength=len(people))

    D = {}
    for t in tags:
        (sv, cv), (sg, cg) = data[t]
        D[t] = (W @ sv) / (W @ cv) - (W @ sg) / (W @ cg)
    rows, bad = [], []
    for t in tags:
        (sv, cv), (sg, cg) = data[t]
        delta = sv.sum() / cv.sum() - sg.sum() / cg.sum()
        lo, hi = np.percentile(D[t], [2.5, 97.5])
        r0 = rep[t]
        if (abs(delta - r0["delta"]) > 1e-12 or abs(lo - r0["paired_ci"]["ci_lo"]) > 1e-12
                or abs(hi - r0["paired_ci"]["ci_hi"]) > 1e-12):
            bad.append(t)
        rows.append({"model": t, "acc_pyes": sv.sum() / cv.sum(), "acc_text": sg.sum() / cg.sum(),
                     "delta": float(delta), "ci": [float(lo), float(hi)],
                     "verdict": s2.verdict(lo, hi),
                     "single_answer_gap": float(gv[t]["gap"]),
                     "single_answer_check": gv[t]["verdict"],
                     "delta_macro_f1": float(f1[t]["delta_f1"]),
                     "delta_macro_f1_ci": f1[t]["delta_f1_ci"]})
    if bad:
        raise SystemExit(f"STOP: per-model FACES deltas/CIs differ from faces_e3_report.json: {bad}")
    print("per-model deltas and CIs reproduce faces_e3_report.json exactly (11/11)")

    passing = [t for t in tags if gv[t]["verdict"] == "CLEAN"]
    fleet = {}
    for name, ms in (("all_11", tags), (f"check_passing_{len(passing)}", passing)):
        d = np.mean([D[t] for t in ms], axis=0)
        point = float(np.mean([r["delta"] for r in rows if r["model"] in ms]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        fleet[name] = {"models": ms, "mean_delta": point, "ci": [float(lo), float(hi)],
                       "gains_ci": int(sum(r["verdict"] == "above" for r in rows if r["model"] in ms)),
                       "null_ci": int(sum(r["verdict"] == "indistinguishable"
                                          for r in rows if r["model"] in ms)),
                       "reversals_ci": int(sum(r["verdict"] == "below"
                                               for r in rows if r["model"] in ms)),
                       "mean_delta_macro_f1": float(np.mean([r["delta_macro_f1"] for r in rows
                                                             if r["model"] in ms]))}
    print(f"\n{'model':32}{'text':>7}{'P(yes)':>8}{'Delta':>8}   95% CI (persons)   gap   F1 Delta")
    for r in rows:
        print(f"{r['model'][:32]:32}{r['acc_text']:7.3f}{r['acc_pyes']:8.3f}{r['delta']:+8.3f}   "
              f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]   {r['single_answer_gap']:.3f}   "
              f"{r['delta_macro_f1']:+.3f}")
    for k, f in fleet.items():
        print(f"mean over {k:18} {f['mean_delta']:+.4f} [{f['ci'][0]:+.4f}, {f['ci'][1]:+.4f}]  "
              f"gains/null/reversals {f['gains_ci']}/{f['null_ci']}/{f['reversals_ci']}  "
              f"macro-F1 Delta {f['mean_delta_macro_f1']:+.4f}")

    out = s2.RESULTS_DIR / "faces_fleet.json"
    s2.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "rows": rows, "fleet": fleet, "n_persons": len(people),
        "provenance": {"B": args.replicates, "seed": args.seed,
                       "resampling": "persons, shared across models",
                       "metric": "six-way accuracy (argmax); no mapping",
                       "exclusion": "none; single-answer gap reported as a diagnostic",
                       "script": "faces_fleet.py",
                       "sources": {str(args.results_dir): s2.dir_sha(args.results_dir)}},
    }, indent=2))
    print(f"wrote {out.relative_to(HERE.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
