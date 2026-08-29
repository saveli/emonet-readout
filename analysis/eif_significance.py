#!/usr/bin/env python
"""Is a verification model above the benchmark's own trained model, or does it only look it?

The paper says five of eleven checkpoints score above Empathic-Insight-Face, the model
EmoNet-Face trained to reach expert level. As first written that compared a model's bootstrap
interval to EIF's bare point estimate -- exactly the mistake `anchor_significance.py` exists to
correct for the human anchor, and it is worse here, because both quantities are computed on
the same 2500 images and therefore move together. A resample rich in images the taxonomy
handles cleanly lifts EIF and every VLM at once; two marginal intervals compared by eye throw
that covariance away and overstate the uncertainty.

So: one image resample per replicate, shared by EIF and by every model, and the reported
quantity is the paired difference `model - EIF`. EIF's own predictions are deterministic and
fixed per image, which is not an argument against an interval: the interval is the sampling
error of the 2500-image test set, not of the annotator, and the same is true of our
verification arm (a forward pass, nothing sampled) which has carried intervals all along.

Both references are scored: Small (the higher of the two under our path, 0.551) and Large
(0.534). Small is the harder bar and the one the paper's claim should rest on.

Guards, in the style of `anchor_significance.py`:

  * EIF's point kappa on the reliable five must reproduce `published_baselines.json`, and each
    model's must reproduce `verify_prefill_bootstrap_ci.json`, or the run aborts. Two scoring
    paths that disagree would make every difference below uninterpretable.
  * The published-baseline scorer and this one must therefore be the same estimator:
    per-emotion quantile calibration refit inside every replicate, scored against the
    individual raters, averaged over categories. `kappa_cal` is that estimator.

Only the reliable-five column can be checked against a published number, because
`published_baselines.py` reports that set alone; the all-40 and 35-excluded columns are new
and are reported without a cross-check.

Usage:
    python analysis/eif_significance.py                    # B=500, results_verify_prefill
    python analysis/eif_significance.py -B 2000
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
    krippendorff_alpha,
    parse_into_arrays,
    scan_index,
)
from verify_bootstrap_ci import kappa_cal  # noqa: E402

# The two references, as they are named in hq.csv's annotator column.
REFERENCES = {
    "eif_small": "Empathic Insight Face Small",
    "eif_large": "Empathic Insight Face Large",
}
SET_LABEL = {"reliable_5": "5 reliable", "unreliable_35": "35 excluded", "all_40": "all 40"}
TOL = 0.0005  # published numbers are quoted to three decimals; this is rounding slack


def per_category(mat, human, gold, rows: np.ndarray, n_emo: int) -> np.ndarray:
    """Calibrated kappa_w per emotion on the image subset `rows`.

    `kappa_cal` over a single-category list returns that category's kappa, so this is the
    identical code path as every published per-model number, calibration included.
    """
    return np.array([kappa_cal(mat, human, gold, [j], rows) for j in range(n_emo)])


def set_means(vec: np.ndarray, sets: dict[str, list[int]]) -> dict[str, float]:
    return {name: float(np.nanmean(vec[idx])) for name, idx in sets.items()}


def check(label: str, got: float, want: float) -> bool:
    ok = abs(got - want) <= TOL
    print(f"  {label:34}{got:9.4f}{want:9.3f}{got - want:+9.4f}   {'YES' if ok else '*** NO ***'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_verify_prefill"))
    ap.add_argument("--hq-csv", type=Path, default=Path("data/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--published", type=Path, default=Path("results/published_baselines.json"))
    ap.add_argument("--verify-ci", type=Path,
                    default=Path("results/verify_prefill_bootstrap_ci.json"))
    ap.add_argument("--out", type=Path, default=Path("results/eif_significance.json"))
    ap.add_argument("-B", "--replicates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.replicates < 2:
        print("this script exists to produce intervals; -B must be at least 2", file=sys.stderr)
        return 2

    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, preds = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]
    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j])
                      for j in range(len(emotions))])

    n_emo, n = len(emotions), len(images)
    rel = [j for j in range(n_emo) if alpha[j] >= RELIABLE_ALPHA]
    unrel = [j for j in range(n_emo) if alpha[j] < RELIABLE_ALPHA]
    sets = {"reliable_5": rel, "unreliable_35": unrel, "all_40": list(range(n_emo))}
    full = np.arange(n)
    print(f"{n} images, {n_emo} emotions, {len(rel)} reliable (alpha >= {RELIABLE_ALPHA})")

    # --- the two references ---------------------------------------------------------------
    ref_mats = {}
    for key, name in REFERENCES.items():
        if name not in annotators:
            print(f"{name!r} is not an annotator in {args.hq_csv}", file=sys.stderr)
            return 1
        ref_mats[key] = preds[annotators[name]]

    # --- the models -----------------------------------------------------------------------
    runs = load_verify_results(args.results_dir, short, n)
    if not runs:
        print(f"no verify results in {args.results_dir}", file=sys.stderr)
        return 1
    matrices = {tag: mat for tag, (mat, _) in sorted(runs.items())}

    ref_point = {key: set_means(per_category(mat, human, gold, full, n_emo), sets)
                 for key, mat in ref_mats.items()}
    model_point = {tag: set_means(per_category(mat, human, gold, full, n_emo), sets)
                   for tag, mat in matrices.items()}

    # --- Step 1: reproduce the published numbers, or stop. ---------------------------------
    print(f"\nreproducing published kappa_w on the reliable five"
          f"\n  {'annotator':34}{'recomputed':>9}{'published':>9}{'delta':>9}   match")
    ok = True
    published = {r["annotator"]: r["kappa_w"] for r in json.load(open(args.published))["rows"]}
    for key, name in REFERENCES.items():
        ok &= check(name, ref_point[key]["reliable_5"], published[name])
    verify_ci = {r["arm"]: r for r in json.load(open(args.verify_ci))["rows"]}
    for tag in matrices:
        if tag not in verify_ci:
            print(f"  {tag:34}{'':9}{'':9}{'':9}   *** ABSENT from {args.verify_ci.name} ***")
            ok = False
            continue
        ok &= check(tag, model_point[tag]["reliable_5"], verify_ci[tag]["kappa"])
    if not ok:
        print("\nSTOP: this scorer does not reproduce a published number, so no difference "
              "below would be interpretable. Investigate before rerunning.", file=sys.stderr)
        return 1
    print("every point estimate reproduced to within rounding\n")

    # --- Paired bootstrap -------------------------------------------------------------------
    # One resample per replicate, shared by both references and every model. That is what makes
    # each difference paired: the shared image-difficulty term cancels instead of accumulating.
    print(f"{len(matrices)} models x {len(REFERENCES)} references, B={args.replicates}, "
          f"resampling images")
    rng = np.random.default_rng(args.seed)
    ref_reps = {key: {name: [] for name in sets} for key in REFERENCES}
    model_reps = {tag: {name: [] for name in sets} for tag in matrices}
    diff_reps = {key: {tag: {name: [] for name in sets} for tag in matrices}
                 for key in REFERENCES}

    for b in range(args.replicates):
        idx = rng.integers(0, n, size=n)
        r_b = {key: set_means(per_category(mat, human, gold, idx, n_emo), sets)
               for key, mat in ref_mats.items()}
        for key in REFERENCES:
            for name in sets:
                ref_reps[key][name].append(r_b[key][name])
        for tag, mat in matrices.items():
            m_b = set_means(per_category(mat, human, gold, idx, n_emo), sets)
            for name in sets:
                model_reps[tag][name].append(m_b[name])
                for key in REFERENCES:
                    diff_reps[key][tag][name].append(m_b[name] - r_b[key][name])
        if (b + 1) % 50 == 0:
            print(f"  {b + 1}/{args.replicates}", flush=True)

    def ci(v):
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    # --- Reference intervals ----------------------------------------------------------------
    ref_out = {}
    print("\nREFERENCES with their own image-bootstrap intervals")
    print(f"{'annotator':34}{'set':16}{'kappa_w':>9}{'95% CI':>20}{'se':>8}")
    for key, name in REFERENCES.items():
        ref_out[key] = {"annotator": name, "sets": {}}
        for sname in ("reliable_5", "unreliable_35", "all_40"):
            lo, hi = ci(ref_reps[key][sname])
            se = float(np.std(ref_reps[key][sname], ddof=1))
            ref_out[key]["sets"][sname] = {"point": ref_point[key][sname],
                                           "ci_lo": lo, "ci_hi": hi, "se": se}
            print(f"{name if sname == 'reliable_5' else '':34}{SET_LABEL[sname]:16}"
                  f"{ref_point[key][sname]:9.3f}   [{lo:.3f}, {hi:.3f}]{se:11.4f}")

    # --- Paired differences -----------------------------------------------------------------
    rows, summary = [], {}
    for key, name in REFERENCES.items():
        summary[key] = {}
        for sname in ("reliable_5", "unreliable_35", "all_40"):
            print(f"\nPAIRED DIFFERENCE model - {name}, {SET_LABEL[sname]} categories "
                  f"({ref_point[key][sname]:.3f})")
            print(f"{'model':34}{'kappa':>8}{'diff':>9}{'95% CI of diff':>22}"
                  f"{'unpaired se':>13}   verdict")
            counts = {"above": 0, "indistinguishable": 0, "below": 0}
            for tag in matrices:
                d = np.array(diff_reps[key][tag][sname])
                lo, hi = ci(d)
                se = float(np.std(d, ddof=1))
                # What the same comparison would claim if the two arms were treated as
                # independent -- printed so the value of pairing is visible, never used.
                unpaired = float(np.hypot(np.std(model_reps[tag][sname], ddof=1),
                                          np.std(ref_reps[key][sname], ddof=1)))
                verdict = ("above" if lo > 0 else "below" if hi < 0 else "indistinguishable")
                counts[verdict] += 1
                rows.append({
                    "model": tag, "reference": name, "reference_key": key,
                    "category_set": sname,
                    "model_kappa": model_point[tag][sname],
                    "reference_kappa": ref_point[key][sname],
                    "diff": model_point[tag][sname] - ref_point[key][sname],
                    "diff_boot_mean": float(np.mean(d)),
                    "diff_ci_lo": lo, "diff_ci_hi": hi, "diff_se": se,
                    "unpaired_se": unpaired,
                    "excludes_zero": bool(lo > 0 or hi < 0), "verdict": verdict,
                })
                print(f"{tag:34}{model_point[tag][sname]:8.3f}"
                      f"{rows[-1]['diff']:+9.3f}   [{lo:+.3f}, {hi:+.3f}]"
                      f"{unpaired:13.4f}   {verdict}")
            summary[key][sname] = counts
            print(f"{'':34}{'':8}{'':9}{'':22}{'':13}   "
                  f"{counts['above']} above / {counts['indistinguishable']} indistinguishable"
                  f" / {counts['below']} below")

    args.out.write_text(json.dumps({
        "results_dir": str(args.results_dir), "replicates": args.replicates,
        "seed": args.seed, "n_images": n, "n_emotions": n_emo,
        "reliable_threshold": RELIABLE_ALPHA,
        "reliable_categories": [short[j] for j in rel],
        "references": ref_out, "summary": summary, "rows": rows,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
