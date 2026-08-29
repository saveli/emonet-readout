#!/usr/bin/env python
"""Does the E1 verification headline survive without the oracle calibration?

Every kappa in this project goes through `quantile_bin`, which rank-matches a model's
scores to the **gold marginal of the evaluation set itself**. C4 already flags that as an
upper bound, but the E1 result leans on it far harder than the E0 arms do: `P(yes)` lives on
0..1 while the ground truth is 0..7, so the uncalibrated number is not merely pessimistic,
it is undefined -- `raw_bin` clips every score to 0 or 1. The reported 0.511 for the
untrained base is therefore an oracle number with no honest fallback printed beside it.

This splits the images in half, fits the quantile cutpoints on half A, applies them to
half B, and scores B. Nothing from B's labels touches the mapping, so the result is
deployable in a way the oracle version is not.

Two halves are used, then swapped, and the two scores averaged -- a single split would make
the answer depend on which half happened to be easier.

If the held-out number tracks the oracle one, the headline stands. If it collapses, the
0.511 was calibration leakage and the "protocol beats the model" claim dies with it.

Usage:
    python analysis/e1_heldout_calibration.py --results-dir results_e1 --hq-csv /tmp/hq.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c4_calibration_control import build_baselines  # noqa: E402
from e1_report import ARM_DESC, ARM_ORDER, load_verify_results  # noqa: E402
from reliability_vs_performance import (  # noqa: E402
    N_LEVELS,
    RELIABLE_ALPHA,
    human_human_kappa,
    krippendorff_alpha,
    model_single_rater_kappa,
    parse_into_arrays,
    quantile_bin,
    scan_index,
)


def fit_cutpoints(pred_fit, gold_fit):
    """Score thresholds that reproduce the gold marginal on the FIT half.

    Returned as (levels, cuts) where cuts are score values, so they can be applied to a
    half whose labels were never seen. `quantile_bin` instead re-derives the marginal from
    whatever it is handed, which is exactly the leakage being tested.
    """
    ok = ~(np.isnan(pred_fit) | np.isnan(gold_fit))
    if ok.sum() < 2:
        return None
    p, g = pred_fit[ok], gold_fit[ok]
    levels, counts = np.unique(np.clip(np.round(g), 0, N_LEVELS - 1), return_counts=True)
    frac = np.cumsum(counts) / counts.sum()
    # Cut at the empirical quantiles of the FIT half's predictions.
    cuts = np.quantile(p, frac[:-1]) if len(levels) > 1 else np.array([])
    return levels, cuts


def apply_cutpoints(pred, fitted):
    out = np.full(pred.shape, np.nan)
    if fitted is None:
        return out
    levels, cuts = fitted
    ok = ~np.isnan(pred)
    out[ok] = levels[np.searchsorted(cuts, pred[ok], side="right")]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_e1"))
    ap.add_argument("--hq-csv", type=Path, default=Path("/tmp/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--out", type=Path, default=Path("results/e1_heldout_calibration.json"))
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, _ = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]
    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j]) for j in range(len(emotions))])
    reliable = alpha >= RELIABLE_ALPHA
    hh = np.nanmean([human_human_kappa(human[:, j, :])
                     for j in range(len(emotions)) if reliable[j]])
    print(f"{int(reliable.sum())}/{len(emotions)} categories reliable (alpha >= {RELIABLE_ALPHA})")
    print(f"human-human anchor on those: kappa_w = {hh:.3f}\n")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(images))
    halves = [perm[: len(perm) // 2], perm[len(perm) // 2:]]

    runs = load_verify_results(args.results_dir, short, len(images))
    if not runs:
        print(f"no E1 verify results in {args.results_dir}", file=sys.stderr)
        return 1
    matrices = {t: m for t, (m, _) in runs.items()}
    matrices.update(build_baselines(gold, 0))

    rows = []
    for tag, mat in matrices.items():
        oracle, heldout = [], []
        for j in range(len(emotions)):
            if not reliable[j]:
                continue
            oracle.append(model_single_rater_kappa(quantile_bin(mat[:, j], gold[:, j]),
                                                   human[:, j, :]))
            # Fit on one half, score the other, both directions.
            per_split = []
            for fit_idx, test_idx in (halves, halves[::-1]):
                fitted = fit_cutpoints(mat[fit_idx, j], gold[fit_idx, j])
                binned = np.full(len(images), np.nan)
                binned[test_idx] = apply_cutpoints(mat[test_idx, j], fitted)
                per_split.append(model_single_rater_kappa(binned, human[:, j, :]))
            heldout.append(np.nanmean(per_split))
        o, h = float(np.nanmean(oracle)), float(np.nanmean(heldout))
        rows.append({"arm": tag, "desc": ARM_DESC.get(tag, ""), "kappa_oracle": o,
                     "kappa_heldout": h, "leakage": o - h})

    order = {a: i for i, a in enumerate(ARM_ORDER)}
    rows.sort(key=lambda r: (order.get(r["arm"], 99), r["arm"]))
    print(f"{'arm':12}{'what':26}{'oracle':>9}{'held-out':>10}{'leakage':>9}")
    for r in rows:
        if r["arm"].startswith("baseline:"):
            continue
        print(f"{r['arm']:12}{r['desc'][:26]:26}{r['kappa_oracle']:9.3f}"
              f"{r['kappa_heldout']:10.3f}{r['leakage']:+9.3f}")
    print()
    for r in rows:
        if r["arm"].startswith("baseline:"):
            print(f"{r['arm']:38}{r['kappa_oracle']:9.3f}{r['kappa_heldout']:10.3f}"
                  f"{r['leakage']:+9.3f}")

    base = next((r for r in rows if r["arm"] == "e1_0_base"), None)
    if base:
        print(f"\nREAD-OUT: untrained base held-out kappa_w = {base['kappa_heldout']:.3f} "
              f"vs human-human {hh:.3f}")
        print(f"          E0 generative for the same weights: 0.286 paper_once / 0.103 ours")
        if base["kappa_heldout"] >= 0.35:
            print("          -> the protocol effect survives without the oracle. Headline stands.")
        else:
            print("          -> the protocol effect was substantially calibration leakage. "
                  "Do NOT quote 0.511.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"human_human_kappa": float(hh), "n_reliable": int(reliable.sum()),
               "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
