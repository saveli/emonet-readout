#!/usr/bin/env python
"""Is the FACES age gap in the SigLIP2 representation, or in the EmoNet-trained heads?

The matched control the external-validity design calls for. Empathic-Insight-Face is a
frozen SigLIP2-so400m backbone plus 40 MLP heads that LAION trained on EmoNet, so its age
gap has two possible sources and the released model cannot separate them:

  (a) the backbone -- SigLIP2 encodes old faces less well, and any head inherits that
  (b) the heads    -- EmoNet's synthetic training faces misrepresent old expressions

This trains a linear probe on the SAME frozen features, supervised by FACES itself. Only
the training data differs, so the comparison is causal about training data in a way that
"EmoNet model is biased" is not:

  probe shows the same gap  -> source is (a), and EmoNet's training data is exonerated
  probe is flat             -> source is (b), the synthetic-fidelity argument survives

Splits are **person-disjoint** (GroupKFold on the FACES person id). Each person contributes
12 images -- 6 emotions x 2 sets -- so a random split would put the same face in train and
test and the probe would score identity, not expression.

The probe is deliberately weak: multinomial logistic regression on 1152-d features, no
hidden layer. A strong probe could fit around a representational gap and hide exactly the
effect being tested. If a linear map on these features already closes the gap, the gap was
never in the representation.

`neutral` is included here even though the taxonomy map cannot express it -- the probe
learns it directly from FACES labels, which is one thing it can do that the mapped EIF
heads structurally cannot. Per-class numbers are therefore comparable to EIF only on the
five mapped classes; the 6-class column is the probe's own ceiling.

Usage:
    python evaluation/empathic_insight_faces.py --dump-embeddings data/faces_siglip2_emb.npz
    python analysis/faces_linear_probe.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
MAPPED = ["angry", "disgust", "fear", "happy", "sad"]   # what EIF can express, for comparison
AGES = ["young", "middle", "old"]
GENDERS = ["female", "male"]
SEED = 3407


def cross_val_predict(emb, y, groups, n_splits, seed, C, quiet=False):
    """Out-of-fold predictions from a person-disjoint GroupKFold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    pred = np.empty(len(y), dtype=object)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(emb, y, groups)):
        sc = StandardScaler().fit(emb[tr])
        clf = LogisticRegression(max_iter=3000, C=C, random_state=seed)
        clf.fit(sc.transform(emb[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(emb[te]))
        if not quiet:
            print(f"  fold {fold + 1}/{n_splits}: train {len(tr)} / test {len(te)} "
                  f"({len(set(groups[tr]) & set(groups[te]))} shared persons)", flush=True)
    return pred


def group_table(gold, pred, attr, groups, classes):
    """{class: {group: acc}} plus macro-averaged per-group accuracy."""
    cell = defaultdict(lambda: [0, 0])
    for g, p, a in zip(gold, pred, attr):
        cell[(g, a)][0] += int(g == p)
        cell[(g, a)][1] += 1
    per_class = {c: {a: (cell[(c, a)][0] / cell[(c, a)][1] if cell[(c, a)][1] else float("nan"))
                     for a in groups} for c in classes}
    macro = {a: float(np.mean([per_class[c][a] for c in classes])) for a in groups}
    return per_class, macro


def bootstrap_spread(gold, pred, attr, persons, groups, classes, n_boot, seed):
    """CI on max-min of the macro group accuracies, resampling persons."""
    idx_by_person = defaultdict(list)
    for i, p in enumerate(persons):
        idx_by_person[p].append(i)
    plist = list(idx_by_person)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(plist), size=len(plist), replace=True)
        idx = [i for k in pick for i in idx_by_person[plist[k]]]
        _, macro = group_table(gold[idx], pred[idx], attr[idx], groups, classes)
        vals = [v for v in macro.values() if v == v]
        if len(vals) > 1:
            out.append(max(vals) - min(vals))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (float("nan"),) * 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", type=Path, default=Path("data/faces_siglip2_emb.npz"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("analysis/faces_linear_probe.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    # Defaults are Empathic-Insight-Face-Large's head-calibrated numbers on the same five
    # mapped classes (analysis/faces_class_by_group_zscored.json): macro-5 accuracy 0.742,
    # age spread 0.090. Override when comparing against a different reference model.
    ap.add_argument("--match-accuracy", type=float, default=0.742)
    ap.add_argument("--match-spread", type=float, default=0.090)
    ap.add_argument("--match-grid", type=float, nargs="+",
                    default=[1e-3, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
    args = ap.parse_args()

    if not args.emb.exists():
        print(f"{args.emb} missing -- run:\n  python evaluation/empathic_insight_faces.py "
              f"--dump-embeddings {args.emb}")
        return 1
    d = np.load(args.emb, allow_pickle=True)
    emb, gold = d["emb"], d["gold"].astype(str)
    persons, age, gender = d["person"].astype(str), d["age"].astype(str), d["gender"].astype(str)
    print(f"{emb.shape[0]} images x {emb.shape[1]} features | {len(set(persons))} persons\n")

    print(f"person-disjoint {args.folds}-fold probe (logistic regression, C={args.C})")
    pred = cross_val_predict(emb, gold, persons, args.folds, args.seed, args.C).astype(str)
    acc = float((pred == gold).mean())
    print(f"\nprobe overall accuracy {acc:.3f}  (chance {1 / len(CLASSES):.3f})\n")

    out = {"n": int(len(gold)), "n_persons": int(len(set(persons))), "folds": args.folds,
           "C": args.C, "accuracy": acc, "axes": {}}

    for axis, attr, groups in (("age", age, AGES), ("gender", gender, GENDERS)):
        for classes, tag in ((CLASSES, "all6"), (MAPPED, "mapped5")):
            per_class, macro = group_table(gold, pred, attr, groups, classes)
            spread = max(macro.values()) - min(macro.values())
            lo, hi = bootstrap_spread(gold, pred, attr, persons, groups, classes,
                                      args.n_boot, args.seed)
            if tag == "all6":
                print(f"=== {axis}: probe accuracy per class x group")
                print(f"{'class':10}" + "".join(f"{g:>10}" for g in groups))
                for c in classes:
                    print(f"{c:10}" + "".join(f"{per_class[c][g]:10.3f}" for g in groups))
            print(f"  macro({tag[-1]}) " + "  ".join(f"{g} {macro[g]:.3f}" for g in groups)
                  + f"   spread {spread:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
            out["axes"][f"{axis}_{tag}"] = {"per_class": per_class, "macro": macro,
                                            "spread": spread, "ci": [lo, hi]}
        print()

    # Comparing spreads at different accuracy levels is invalid. A model near ceiling has
    # no room to differ across groups, so a strong probe's small spread is not evidence that
    # its training data is fairer -- it is arithmetic. Handicap the probe with L2 until its
    # accuracy matches the reference, then compare. Regularisation is the right handicap
    # because it degrades gracefully toward the classes that are genuinely hard; additive
    # feature noise instead pushes every class to chance uniformly and *shrinks* the spread,
    # which would manufacture the opposite conclusion.
    print(f"=== matched-accuracy control (target macro-5 accuracy {args.match_accuracy:.3f})")
    print(f"{'C':>10}{'macro5 acc':>12}{'spread':>9}   per-age")
    sweep = []
    for C in args.match_grid:
        p = cross_val_predict(emb, gold, persons, args.folds, args.seed, C, quiet=True).astype(str)
        _, macro = group_table(gold, p, age, AGES, MAPPED)
        a = float(np.mean(list(macro.values())))
        s = max(macro.values()) - min(macro.values())
        sweep.append({"C": C, "macro5_accuracy": a, "spread": s,
                      "macro": macro, "pred": p})
        print(f"{C:10.0e}{a:12.3f}{s:9.3f}   " + " ".join(f"{g} {macro[g]:.3f}" for g in AGES))

    best = min(sweep, key=lambda r: abs(r["macro5_accuracy"] - args.match_accuracy))
    lo, hi = bootstrap_spread(gold, best["pred"], age, persons, AGES, MAPPED,
                              args.n_boot, args.seed)
    out["matched"] = {"target_accuracy": args.match_accuracy, "reference_spread": args.match_spread,
                      "C": best["C"], "macro5_accuracy": best["macro5_accuracy"],
                      "spread": best["spread"], "ci": [lo, hi],
                      "sweep": [{k: v for k, v in r.items() if k != "pred"} for r in sweep]}

    print(f"\nREAD-OUT")
    print(f"  EIF (head-calibrated)  macro-5 accuracy {args.match_accuracy:.3f}  "
          f"spread {args.match_spread:.3f}")
    print(f"  probe at C={best['C']:.0e}       macro-5 accuracy {best['macro5_accuracy']:.3f}  "
          f"spread {best['spread']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    if lo <= args.match_spread <= hi:
        print("  -> At matched accuracy the FACES-supervised probe shows the SAME age spread.")
        print("     The gap is a property of the task and the frozen representation, not of")
        print("     EmoNet's training data. Do NOT report it as a synthetic-fidelity finding.")
    elif hi < args.match_spread:
        print("  -> EIF's spread exceeds a matched-accuracy probe. That excess is the part")
        print("     attributable to EmoNet training; report the excess, not the raw spread.")
    else:
        print("  -> The matched probe spreads MORE than EIF. EIF is fairer than a linear")
        print("     readout of its own features at the same accuracy.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
