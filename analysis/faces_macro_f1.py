#!/usr/bin/env python
"""Macro-F1 on FACES: is the verification gain a class-prior shift?

Verification argmaxes six independent presence probabilities, so it can move mass between
classes in a way six-way accuracy rewards without perception improving. Accuracy alone cannot
distinguish those. Macro-F1 weights every class equally regardless of how often it is
predicted, so a gain that is purely a prior reallocation does not survive it.

This exists as a script, rather than as a number computed once in a shell, because the paper
quotes it and a reviewer must be able to reproduce it. Gated models are excluded on the same
criterion the aggregate uses (see generative_validity_check.py).

Every macro-F1 number carries a paired bootstrap interval over PERSONS, on the same
convention as faces_e3_report.py: each person contributes twelve images (six emotions, two
sets), so images are not independent and an image bootstrap would understate the width. The
draws are shared across models -- one B x 171 matrix of person indices, reused for every
model -- so the interval on the fleet mean is a CI on the mean over the same resampled
people, not a combination of independently resampled per-model CIs.

Usage:
    python analysis/faces_macro_f1.py --results-dir results_faces -B 2000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generative_validity_check import analyse  # noqa: E402


def macro_f1(rows) -> float:
    tp, fp, fn = Counter(), Counter(), Counter()
    for r in rows:
        g, p = r["gold"], r["pred"]
        if p is None:
            fn[g] += 1
            continue
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    out = []
    for c in set(tp) | set(fp) | set(fn):
        pr = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rc = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return statistics.mean(out)


def macro_f1_counts(gold: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Same statistic as macro_f1, from integer code arrays. pred == -1 means unparsed.

    Kept separate from the reference implementation above rather than replacing it: the
    bootstrap calls this 2000 times per model and the loop version is too slow, but the
    number the paper quotes must still come from code a reader can check by eye. main()
    asserts the two agree on the full sample for every arm, so they cannot drift.
    """
    hit = pred >= 0
    tp = np.bincount(gold[hit & (pred == gold)], minlength=k)
    pred_n = np.bincount(pred[hit], minlength=k)
    gold_n = np.bincount(gold, minlength=k)
    fp = pred_n - tp
    fn = gold_n - tp
    present = (tp + fp + fn) > 0
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return float(f1[present].mean()) if present.any() else 0.0


def encode(rows, classes, order):
    """Integer gold/pred vectors for one arm, in a fixed image order.

    Keyed on the image basename because the two clusters mount FACES at different absolute
    paths, so the `path` field is not comparable across arms (section 0a).
    """
    by_img = {Path(r["path"]).name: r for r in rows}
    gold = np.array([classes[by_img[i]["gold"]] for i in order], dtype=np.int64)
    pred = np.array([classes.get(by_img[i]["pred"], -1) for i in order], dtype=np.int64)
    return gold, pred


def load_arm(path: Path):
    payload = json.load(open(path))
    return payload, {Path(r["path"]).name: r for r in payload["results"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_faces"))
    ap.add_argument("--out", type=Path, default=Path("analysis/faces_macro_f1.json"))
    ap.add_argument("-B", "--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tags = sorted({p.name.split("__")[0]
                   for p in args.results_dir.glob("*__faces_verify.json")
                   if not p.name.startswith("INVALID_")})

    # One image order and one person index, shared by every model, so a single draw of
    # persons resamples all models identically -- the precondition for a CI on the mean.
    arms = {}
    for t in tags:
        gp = args.results_dir / f"{t}__faces_generative.json"
        vp = args.results_dir / f"{t}__faces_verify.json"
        if gp.is_file() and vp.is_file():
            arms[t] = (load_arm(gp), load_arm(vp))
    if not arms:
        print(f"no paired FACES arms in {args.results_dir}")
        return 1

    order = sorted(set.intersection(*[set(m) for pair in arms.values() for _, m in pair]))
    labels = sorted({r["gold"] for (gpay, _), _ in arms.values()
                     for r in gpay["results"]})
    classes = {c: i for i, c in enumerate(labels)}
    k = len(labels)

    persons = defaultdict(list)
    any_arm = next(iter(arms.values()))[0][1]
    for i, img in enumerate(order):
        persons[any_arm[img]["person"]].append(i)
    people = sorted(persons)
    pidx = [np.array(persons[p], dtype=np.int64) for p in people]

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(people), size=(args.replicates, len(people)))
    # Materialise each replicate's row index once, not once per model.
    reps = [np.concatenate([pidx[j] for j in draw]) for draw in draws]

    print(f"FACES macro-F1, {len(order)} images / {len(people)} persons / {k} classes")
    print(f"paired person bootstrap, B={args.replicates}, seed={args.seed}, "
          f"draws shared across models\n")
    print(f"{'model':<26}{'F1 gen':>8}{'F1 ver':>8}{'d F1':>8}"
          f"{'95% CI (persons)':>21}{'d acc':>8}  gate")

    rows, per_model_deltas = [], {}
    for t in sorted(arms):
        (gpay, gmap), (vpay, vmap) = arms[t]
        gg, gp_ = encode(gpay["results"], classes, order)
        vg, vp_ = encode(vpay["results"], classes, order)

        f1g, f1v = macro_f1_counts(gg, gp_, k), macro_f1_counts(vg, vp_, k)
        # The vectorised statistic must equal the readable one on the full sample.
        assert abs(f1g - macro_f1(gpay["results"])) < 1e-9, t
        assert abs(f1v - macro_f1(vpay["results"])) < 1e-9, t

        deltas = np.array([macro_f1_counts(vg[i], vp_[i], k) -
                           macro_f1_counts(gg[i], gp_[i], k) for i in reps])
        per_model_deltas[t] = deltas
        lo, hi = np.percentile(deltas, [2.5, 97.5])

        acc_g = float((gp_ == gg).mean())
        acc_v = float((vp_ == vg).mean())
        r = {"model": t, "gate": analyse(gpay)["verdict"],
             "acc_generative": acc_g, "acc_verify": acc_v, "delta_acc": acc_v - acc_g,
             "f1_generative": f1g, "f1_verify": f1v, "delta_f1": f1v - f1g,
             "delta_f1_ci": [float(lo), float(hi)],
             "delta_f1_excludes_zero": bool(lo > 0 or hi < 0)}
        rows.append(r)
        star = " *" if r["delta_f1_excludes_zero"] else ""
        print(f"{t:<26}{f1g:>8.3f}{f1v:>8.3f}{r['delta_f1']:>+8.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>19}{star:2s}{r['delta_acc']:>+8.3f}  {r['gate']}")

    keep = [r for r in rows if r["gate"] != "INVALID"]
    da = [r["delta_acc"] for r in keep]
    df = [r["delta_f1"] for r in keep]
    # The fleet mean is recomputed inside each replicate over the same resampled people.
    mean_reps = np.mean([per_model_deltas[r["model"]] for r in keep], axis=0)
    mlo, mhi = np.percentile(mean_reps, [2.5, 97.5])
    sig = sum(1 for r in keep if r["delta_f1_excludes_zero"])

    print(f"\n* = paired CI excludes zero ({sig}/{len(keep)} gate-passed models)")
    print(f"gate-passed n={len(keep)}: mean d accuracy {statistics.mean(da):+.3f}, "
          f"mean d macro-F1 {statistics.mean(df):+.3f} "
          f"95% CI [{mlo:+.3f}, {mhi:+.3f}]")
    print(f"gain on accuracy {sum(1 for x in da if x > 0)}/{len(da)}, "
          f"on macro-F1 {sum(1 for x in df if x > 0)}/{len(df)}")

    args.out.write_text(json.dumps(
        {"replicates": args.replicates, "seed": args.seed,
         "n_persons": len(people), "n_images": len(order),
         "n_gate_passed": len(keep),
         "mean_delta_acc": statistics.mean(da), "mean_delta_f1": statistics.mean(df),
         "mean_delta_f1_ci": [float(mlo), float(mhi)],
         "mean_delta_f1_excludes_zero": bool(mlo > 0 or mhi < 0),
         "n_delta_f1_excludes_zero": sig,
         "rows": rows}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
