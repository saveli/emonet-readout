#!/usr/bin/env python
"""Is the FACES age gap a bias signal, or class difficulty correlated with age?

`empathic_insight_faces.py` reports old faces 6 points below middle/young. That number is
not interpretable on its own: two of the six classes collapse entirely under the taxonomy
map (angry -> disgust 279/342, neutral -> sad 207/342), so overall accuracy is really
accuracy on four classes plus noise. If the age groups are not identically composed, or if
the classes that survive are differentially hard per age group, the gap is arithmetic
rather than a property of the model.

Three cuts, all on the per-image records the scorer now dumps:

  1. class x group accuracy, so a per-class effect is visible directly
  2. class composition per group -- FACES is balanced by design, and this checks it
  3. macro-averaged group accuracy (every class weighted equally), which is invariant to
     composition. If the gap survives here it is not a mix effect.

Bootstrap CIs are over persons, not images: each person contributes 12 images (6 emotions
x 2 sets), so resampling images treats correlated rows as independent and gives an
interval that is too narrow.

Usage:
    python analysis/faces_class_by_group.py --in results/empathic_faces_core.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
AXES = {"age": ["young", "middle", "old"], "gender": ["female", "male"]}
N_BOOT = 2000
SEED = 3407


def zscore_predictions(recs):
    """Re-argmax after standardising each label's aggregated score across the dataset.

    The released heads are not on a common scale -- over FACES the anger group averages
    -6.34 and the disgust group +16.81, and happy's sd is 36.0 against disgust's 16.4. A
    raw argmax therefore reports head calibration as much as it reports the image, which is
    why `angry` never wins despite ranking 2nd or 3rd on 91% of angry faces (mean rank
    2.85, never below 5). Standardising recovers angry from 0.000 to 0.82.

    This is a diagnostic, not a fix to ship: it uses the whole evaluation set's marginal, so
    it is an oracle in the same sense as C4's quantile calibration, and it assumes a uniform
    gold marginal -- true for FACES (342 per class) and not in general. Its purpose is to
    check whether a demographic result is a property of the model or of the head scaling.
    """
    labels = sorted({l for r in recs for l in r.get("label_scores", {})})
    if not labels:
        return False
    mu = {l: np.mean([r["label_scores"][l] for r in recs]) for l in labels}
    sd = {l: np.std([r["label_scores"][l] for r in recs]) or 1.0 for l in labels}
    for r in recs:
        z = {l: (r["label_scores"][l] - mu[l]) / sd[l] for l in labels}
        r["pred"] = max(labels, key=lambda l: z[l])
    return True


def cell_table(recs, axis):
    """{(gold, group): [correct, total]}"""
    tab = defaultdict(lambda: [0, 0])
    for r in recs:
        tab[(r["gold"], r[axis])][0] += int(r["pred"] == r["gold"])
        tab[(r["gold"], r[axis])][1] += 1
    return tab


def macro_by_group(recs, axis, groups, classes):
    """Group accuracy with every class weighted equally -> composition-invariant."""
    tab = cell_table(recs, axis)
    out = {}
    for g in groups:
        per = [tab[(c, g)][0] / tab[(c, g)][1] for c in classes if tab[(c, g)][1]]
        out[g] = float(np.mean(per)) if per else float("nan")
    return out


def bootstrap_spread(recs, axis, groups, classes, n_boot=N_BOOT, seed=SEED):
    """CI on max-min of the macro group accuracies, resampling whole persons."""
    by_person = defaultdict(list)
    for r in recs:
        by_person[r["person"]].append(r)
    persons = list(by_person)
    rng = np.random.default_rng(seed)
    spreads = []
    for _ in range(n_boot):
        pick = rng.choice(len(persons), size=len(persons), replace=True)
        sample = [r for i in pick for r in by_person[persons[i]]]
        m = macro_by_group(sample, axis, groups, classes)
        vals = [v for v in m.values() if v == v]
        spreads.append(max(vals) - min(vals) if len(vals) > 1 else float("nan"))
    s = np.array(spreads)
    s = s[~np.isnan(s)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("results/empathic_faces_core.json"))
    ap.add_argument("--out", type=Path, default=Path("results/faces_class_by_group.json"))
    ap.add_argument("--calibrate-heads", action="store_true",
                    help="re-argmax on per-label z-scores; see zscore_predictions")
    args = ap.parse_args()

    payload = json.load(open(args.inp))
    recs = payload.get("records")
    if not recs:
        print(f"{args.inp} has no per-image records; re-run empathic_insight_faces.py")
        return 1
    if args.calibrate_heads:
        if not zscore_predictions(recs):
            print(f"{args.inp} has no per-label score vectors; re-run "
                  "empathic_insight_faces.py to get them")
            return 1
        print("head scores z-scored per label before the argmax (diagnostic, oracle)\n")

    # Classes the mapping does not collapse. Reported separately because a class the model
    # can never predict contributes a constant 0 to every group and only dilutes the spread.
    tot = cell_table(recs, "age")
    alive = [c for c in CLASSES
             if sum(tot[(c, g)][0] for g in AXES["age"]) > 0.05 * sum(tot[(c, g)][1] for g in AXES["age"])]
    dead = [c for c in CLASSES if c not in alive]

    out = {"n": len(recs), "collapsed_classes": dead, "surviving_classes": alive, "axes": {}}
    print(f"n = {len(recs)} | collapsed under the mapping: {', '.join(dead) or 'none'}\n")

    for axis, groups in AXES.items():
        tab = cell_table(recs, axis)
        print(f"=== {axis}: accuracy per class x group")
        print(f"{'class':10}" + "".join(f"{g:>18}" for g in groups) + f"{'spread':>9}")
        per_class = {}
        for c in CLASSES:
            accs, cells = [], ""
            for g in groups:
                k, n = tab[(c, g)]
                a = k / n if n else float("nan")
                accs.append(a)
                cells += f"{a:.3f} (n={n})".rjust(18)
            per_class[c] = {g: a for g, a in zip(groups, accs)}
            print(f"{c:10}{cells}{max(accs) - min(accs):9.3f}")

        print(f"\n{'class counts per group (composition check)':}")
        print(f"{'group':10}" + "".join(f"{c:>10}" for c in CLASSES))
        for g in groups:
            print(f"{g:10}" + "".join(f"{tab[(c, g)][1]:10d}" for c in CLASSES))

        micro = {g: sum(tab[(c, g)][0] for c in CLASSES) / sum(tab[(c, g)][1] for c in CLASSES)
                 for g in groups}
        macro_all = macro_by_group(recs, axis, groups, CLASSES)
        macro_alive = macro_by_group(recs, axis, groups, alive)
        lo, hi = bootstrap_spread(recs, axis, groups, CLASSES)
        lo_a, hi_a = bootstrap_spread(recs, axis, groups, alive)
        sp_all = max(macro_all.values()) - min(macro_all.values())
        sp_alive = max(macro_alive.values()) - min(macro_alive.values())

        print(f"\n{'group':10}{'micro':>9}{'macro(6)':>10}{'macro(' + str(len(alive)) + ')':>10}")
        for g in groups:
            print(f"{g:10}{micro[g]:9.3f}{macro_all[g]:10.3f}{macro_alive[g]:10.3f}")
        print(f"\n  macro(6) spread  {sp_all:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  macro({len(alive)}) spread  {sp_alive:.3f}   95% CI [{lo_a:.3f}, {hi_a:.3f}]")
        # A CI whose lower bound sits at ~0 means the observed spread is consistent with
        # no group effect at all -- the interval is on |max-min|, which is non-negative and
        # therefore biased away from zero, so a lower bound near zero is the strong signal.
        verdict = ("consistent with no group effect" if lo_a <= 0.01
                   else "group effect survives composition control")
        print(f"  verdict: {verdict}\n")

        out["axes"][axis] = {
            "per_class": per_class, "micro": micro, "macro_all": macro_all,
            "macro_surviving": macro_alive, "spread_macro_all": sp_all,
            "spread_macro_surviving": sp_alive,
            "ci_macro_all": [lo, hi], "ci_macro_surviving": [lo_a, hi_a],
            "verdict": verdict,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
