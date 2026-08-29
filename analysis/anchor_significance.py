#!/usr/bin/env python
"""Is the gap between a model and the human anchor larger than sampling noise?

The paper currently says every verification interval lies above the human--human anchor,
"the lowest lower bound being 0.490" against an anchor of 0.468. That sentence compares an
interval to a **point estimate**. The anchor is itself a statistic computed on the same 2500
images, with its own sampling error, so "the model interval sits above the number 0.468" is
not the same claim as "the model is above human agreement". A reviewer asked for the second
claim; this script is what licenses it.

The fix is a paired bootstrap. Images are resampled once per replicate and **both** the model
kappa and the anchor are recomputed on that same resample, so the difference
`model - anchor` carries the covariance between the two: the two quantities move together
across replicates (a resample heavy in easy images lifts humans and models alike), and an
unpaired comparison of two marginal intervals would therefore overstate the uncertainty.
Overlapping marginal intervals do not imply an insignificant difference, and this script
reports the difference interval rather than inviting that inference from the two margins.

Everything is recomputed through the functions that produced the published numbers --
`human_human_kappa` for the anchor, `kappa_cal` (quantile calibration refit inside every
replicate, per `verify_bootstrap_ci.py`) for the models. Step 1 below re-derives the three
published anchors and refuses to continue if they disagree, because a second implementation
that quietly disagrees with the first is the failure mode this project has hit repeatedly.

Three category sets, matching the paper's taxonomy table: the 5 reliable categories (alpha >= 0.3,
paper 0.468), the 35 excluded ones (paper 0.166), and all 40 (paper 0.204).

What it would be wrong to conclude:

  * A significantly positive difference on the 35 "unreliable" categories is **not** evidence
    that models read those emotions well. The anchor there is 0.166 -- humans barely agree --
    so clearing it significantly is clearing a floor, not a ceiling. The paper already
    declines to make that claim and this script does not change that.
  * The anchor is a mean over pairs of *these* eight annotators on *this* image sample. A CI
    on it covers image sampling only. Rater sampling is not resampled (raters vary in number
    per image and are not exchangeable units here), so the interval is a lower bound on the
    true uncertainty about "human agreement" as a population quantity.
  * Significance is about precision, not about validity. The calibration step reads the gold
    marginal, so the model side remains an oracle-calibrated upper bound, exactly as
    elsewhere in the project. A significant positive difference under an oracle-calibrated
    readout is still an oracle-calibrated result.

Usage:
    python analysis/anchor_significance.py --hq-csv data/hq.csv -B 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e1_report import load_verify_results  # noqa: E402
from reliability_vs_performance import (  # noqa: E402
    RELIABLE_ALPHA,
    human_human_kappa,
    krippendorff_alpha,
    parse_into_arrays,
    scan_index,
)
from verify_bootstrap_ci import kappa_cal  # noqa: E402

# The three numbers as they appear in the paper (sec/4_results.tex, sec/A_taxonomy.tex,
# sec/2_related.tex). Recomputation must land on these or the run aborts.
PAPER_ANCHORS = {"reliable_5": 0.468, "unreliable_35": 0.166, "all_40": 0.204}
ANCHOR_TOL = 0.0005  # they are quoted to three decimals, so this is pure rounding slack

SET_LABEL = {"reliable_5": "5 reliable", "unreliable_35": "35 excluded", "all_40": "all 40"}


def per_category_anchor(human: np.ndarray, rows: np.ndarray, n_emo: int) -> np.ndarray:
    """Human--human kappa_w per emotion, on the image subset `rows`.

    Kept per-category rather than pre-averaged because the three category sets are three
    different means over the same 40 values; computing the vector once per replicate makes
    the three sets free instead of tripling the cost.
    """
    return np.array([human_human_kappa(human[rows, j, :]) for j in range(n_emo)])


def per_category_model(mat, human, gold, rows: np.ndarray, n_emo: int) -> np.ndarray:
    """Calibrated model kappa_w per emotion, on the image subset `rows`.

    `kappa_cal` over a single-category list returns that category's kappa, so this goes
    through the identical code path as the published per-model numbers -- including the
    quantile calibration being refit on the resampled images.
    """
    return np.array([kappa_cal(mat, human, gold, [j], rows) for j in range(n_emo)])


def set_means(vec: np.ndarray, sets: dict[str, list[int]]) -> dict[str, float]:
    return {name: float(np.nanmean(vec[idx])) for name, idx in sets.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_verify_prefill"))
    ap.add_argument("--hq-csv", type=Path, default=Path("data/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--out", type=Path, default=Path("results/anchor_significance.json"))
    ap.add_argument("-B", "--replicates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.replicates < 2:
        print("this script exists to produce intervals; -B must be at least 2", file=sys.stderr)
        return 2

    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, _ = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]
    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j])
                      for j in range(len(emotions))])

    n_emo, n = len(emotions), len(images)
    rel = [j for j in range(n_emo) if alpha[j] >= RELIABLE_ALPHA]
    unrel = [j for j in range(n_emo) if alpha[j] < RELIABLE_ALPHA]
    sets = {"reliable_5": rel, "unreliable_35": unrel, "all_40": list(range(n_emo))}
    print(f"{n} images, {n_emo} emotions, {len(rel)} reliable (alpha >= {RELIABLE_ALPHA})")

    full = np.arange(n)

    # --- Step 1: reproduce the published anchors, or stop. -------------------------------
    hh_point = per_category_anchor(human, full, n_emo)
    anchor_point = set_means(hh_point, sets)

    print(f"\n{'category set':16}{'recomputed':>12}{'paper':>9}{'delta':>9}   match")
    all_match = True
    for name in ("reliable_5", "unreliable_35", "all_40"):
        got, want = anchor_point[name], PAPER_ANCHORS[name]
        ok = abs(got - want) <= ANCHOR_TOL
        all_match &= ok
        print(f"{SET_LABEL[name]:16}{got:12.4f}{want:9.3f}{got - want:+9.4f}   "
              f"{'YES' if ok else '*** NO ***'}")
    if not all_match:
        print("\nSTOP: the recomputed anchor does not reproduce the published value. Nothing "
              "below would be interpretable, and inventing a new anchor here would silently "
              "contradict the paper. Investigate before rerunning.", file=sys.stderr)
        return 1
    print("all three published anchors reproduced to within rounding\n")

    # --- Models -------------------------------------------------------------------------
    runs = load_verify_results(args.results_dir, short, n)
    if not runs:
        print(f"no verify results in {args.results_dir}", file=sys.stderr)
        return 1
    matrices = {tag: mat for tag, (mat, _) in sorted(runs.items())}
    print(f"{len(matrices)} models in the verification arm, B={args.replicates}, "
          f"resampling images\n")

    model_point = {tag: set_means(per_category_model(mat, human, gold, full, n_emo), sets)
                   for tag, mat in matrices.items()}

    # --- Paired bootstrap ---------------------------------------------------------------
    # One resample per replicate, shared by the anchor and by every model. That is what makes
    # the per-model difference paired with the anchor *and* the models paired with each other.
    rng = np.random.default_rng(args.seed)
    anchor_reps = {name: [] for name in sets}
    diff_reps = {tag: {name: [] for name in sets} for tag in matrices}
    model_reps = {tag: {name: [] for name in sets} for tag in matrices}

    for b in range(args.replicates):
        idx = rng.integers(0, n, size=n)
        a_b = set_means(per_category_anchor(human, idx, n_emo), sets)
        for name in sets:
            anchor_reps[name].append(a_b[name])
        for tag, mat in matrices.items():
            m_b = set_means(per_category_model(mat, human, gold, idx, n_emo), sets)
            for name in sets:
                model_reps[tag][name].append(m_b[name])
                diff_reps[tag][name].append(m_b[name] - a_b[name])
        if (b + 1) % 25 == 0:
            print(f"  {b + 1}/{args.replicates}", flush=True)

    def ci(v):
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    anchor_out = {}
    print("\nANCHOR with its own image-bootstrap interval")
    print(f"{'category set':16}{'kappa_w':>9}{'95% CI':>20}{'se':>8}")
    for name in ("reliable_5", "unreliable_35", "all_40"):
        lo, hi = ci(anchor_reps[name])
        se = float(np.std(anchor_reps[name], ddof=1))
        anchor_out[name] = {"point": anchor_point[name], "paper": PAPER_ANCHORS[name],
                            "matches_paper": True, "ci_lo": lo, "ci_hi": hi, "se": se}
        print(f"{SET_LABEL[name]:16}{anchor_point[name]:9.3f}   [{lo:.3f}, {hi:.3f}]{se:11.4f}")

    rows, summary = [], {}
    for name in ("reliable_5", "unreliable_35", "all_40"):
        print(f"\nPAIRED DIFFERENCE model - anchor, {SET_LABEL[name]} categories "
              f"(anchor {anchor_point[name]:.3f})")
        print(f"{'model':34}{'kappa':>8}{'diff':>9}{'95% CI of diff':>22}   verdict")
        counts = {"positive": 0, "indistinguishable": 0, "negative": 0}
        for tag in matrices:
            d = np.asarray(diff_reps[tag][name])
            lo, hi = ci(d)
            verdict = "positive" if lo > 0 else "negative" if hi < 0 else "indistinguishable"
            counts[verdict] += 1
            mk_lo, mk_hi = ci(model_reps[tag][name])
            rows.append({
                "model": tag, "category_set": name,
                "model_kappa": model_point[tag][name],
                "model_kappa_ci": [mk_lo, mk_hi],
                "anchor": anchor_point[name],
                "diff": model_point[tag][name] - anchor_point[name],
                "diff_boot_mean": float(d.mean()),
                "diff_ci_lo": lo, "diff_ci_hi": hi,
                "diff_se": float(d.std(ddof=1)),
                "excludes_zero": bool(lo > 0 or hi < 0),
                "verdict": verdict,
            })
            print(f"{tag:34}{model_point[tag][name]:8.3f}"
                  f"{model_point[tag][name] - anchor_point[name]:+9.3f}"
                  f"   [{lo:+.3f}, {hi:+.3f}]   {verdict}")
        summary[name] = counts
        print(f"  -> {counts['positive']} significantly above the anchor, "
              f"{counts['indistinguishable']} indistinguishable, "
              f"{counts['negative']} significantly below "
              f"(of {len(matrices)} models)")

    print("\nSUMMARY -- what the paper is entitled to say")
    for name in ("reliable_5", "unreliable_35", "all_40"):
        c = summary[name]
        print(f"  {SET_LABEL[name]:14}: {c['positive']}/{len(matrices)} models are "
              f"significantly ABOVE the human anchor of {anchor_point[name]:.3f} "
              f"(paired image bootstrap, B={args.replicates}); "
              f"{c['indistinguishable']} indistinguishable, {c['negative']} below.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results_dir": str(args.results_dir), "replicates": args.replicates,
               "seed": args.seed, "n_images": n, "n_emotions": n_emo,
               "reliable_threshold": RELIABLE_ALPHA,
               "reliable_categories": [short[j] for j in rel],
               "anchor": anchor_out, "summary": summary, "rows": rows},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
