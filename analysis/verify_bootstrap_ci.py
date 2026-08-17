#!/usr/bin/env python
"""Bootstrap CIs on the verification-protocol kappa_w, resampling images.

Every number in the verification sweep so far is a point estimate. That was tolerable
while the between-model spread was 0.072 (generative), but under the protocol the spread
collapses to sd 0.023 and the held-out calibration control leaks up to 0.026 -- so the
ranking is only meaningful if the per-model CI is narrower than that. This script produces
the missing error bars.

The whole scoring pipeline is resampled, calibration included: `quantile_bin` reads the
gold marginal, so recalibrating inside each bootstrap replicate is what keeps the CI
honest. Calibrating once on the full sample and then resampling would understate the
width, which is the same oracle mistake the report's own docstring warns about.

Images are the resampling unit because they are the independent observations; emotions are
not (the 40 queries share one image) and raters are not (variable count per image).

`--protocol generative` scores the E0 arms through the identical path, so the spread
collapse can be stated as a comparison of two spreads against a common CI width rather
than as two point estimates side by side.

`--tie-sensitivity` measures a second source of width that the image bootstrap cannot see:
`quantile_bin` ranks with `argsort(kind="stable")`, so predictions that are exactly tied are
assigned to gold levels **in image-index order**. For a saturated predictor a large tied
block straddles a level boundary and which images land on each side is arbitrary. Randomising
the order within ties and re-scoring exposes that as a spread. This is not sampling noise --
it is the same data scored twice by an equally defensible rule.

Usage:
    python analysis/verify_bootstrap_ci.py --results-dir results_verify_prefill -B 200
    python analysis/verify_bootstrap_ci.py --protocol generative --results-dir results_e0
    python analysis/verify_bootstrap_ci.py --tie-sensitivity -B 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c4_calibration_control import build_baselines  # noqa: E402
from e0_report import load_arm_results  # noqa: E402
from e1_report import load_verify_results  # noqa: E402
from reliability_vs_performance import (  # noqa: E402
    N_LEVELS,
    RELIABLE_ALPHA,
    krippendorff_alpha,
    model_single_rater_kappa,
    parse_into_arrays,
    quantile_bin,
    scan_index,
)


def quantile_bin_shuffled(pred, gold, rng):
    """`quantile_bin` with ties broken at random instead of by image index.

    Identical to the original except for the sort key: `lexsort((noise, pred))` orders by
    prediction and resolves exact ties uniformly at random, where `argsort(kind="stable")`
    resolves them by position in the array.
    """
    out = np.full(pred.shape, np.nan)
    ok = ~(np.isnan(pred) | np.isnan(gold))
    if ok.sum() < 2:
        return out
    p, g = pred[ok], gold[ok]
    levels, counts = np.unique(np.clip(np.round(g), 0, N_LEVELS - 1), return_counts=True)
    order = np.lexsort((rng.random(len(p)), p))
    assigned = np.empty(len(p))
    start = 0
    for lvl, cnt in zip(levels, counts):
        assigned[order[start:start + cnt]] = lvl
        start += cnt
    if start < len(p):
        assigned[order[start:]] = levels[-1]
    out[ok] = assigned
    return out


def kappa_cal(mat, human, gold, reliable_idx, rows, rng=None):
    """Mean calibrated kappa_w over the reliable categories, on the image subset `rows`.

    With `rng`, ties in the calibration step are broken at random rather than by index.
    """
    binner = (lambda p, g: quantile_bin_shuffled(p, g, rng)) if rng is not None else quantile_bin
    ks = []
    for j in reliable_idx:
        ks.append(model_single_rater_kappa(
            binner(mat[rows, j], gold[rows, j]), human[rows, j, :]))
    return float(np.nanmean(ks))


def tie_fraction(mat, reliable_idx):
    """Fraction of predictions sharing their exact value with another image, per emotion."""
    fr = []
    for j in reliable_idx:
        p = mat[:, j][~np.isnan(mat[:, j])]
        if len(p) < 2:
            continue
        _, counts = np.unique(p, return_counts=True)
        fr.append((counts[counts > 1].sum()) / len(p))
    return float(np.mean(fr)) if fr else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_verify_prefill"))
    ap.add_argument("--hq-csv", type=Path, default=Path("/tmp/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--out", type=Path, default=Path("analysis/verify_bootstrap_ci.json"))
    ap.add_argument("-B", "--replicates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baselines", action="store_true",
                    help="also bootstrap the five trivial baselines")
    ap.add_argument("--protocol", choices=["verification", "generative"],
                    default="verification",
                    help="verification: verify_eval.py files. generative: E0 arms, scored "
                         "through the identical path so the two spreads are comparable.")
    ap.add_argument("--arms", default="paper_once,ours",
                    help="generative only. paper_retry is quarantined (greedy decoding) and "
                         "is excluded by default.")
    ap.add_argument("--tie-sensitivity", type=int, default=0, metavar="T",
                    help="re-score T times with random tie-breaking in quantile_bin and "
                         "report the resulting spread alongside the image bootstrap")
    args = ap.parse_args()

    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, _ = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]
    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j])
                      for j in range(len(emotions))])
    reliable_idx = [j for j in range(len(emotions)) if alpha[j] >= RELIABLE_ALPHA]

    if args.protocol == "verification":
        runs = load_verify_results(args.results_dir, short, len(images))
        matrices = {tag: mat for tag, (mat, _) in runs.items()}
    else:
        keep = set(args.arms.split(","))
        arms = load_arm_results(args.results_dir, short, len(images))
        # One row per model, taking its better arm -- the same "gen kappa" the sweep tables
        # quote. Picking the max is a favourable-to-generative choice and is stated as one.
        best = {}
        for (tag, arm), (mat, _) in arms.items():
            if arm not in keep:
                continue
            k = kappa_cal(mat, human, gold, reliable_idx, np.arange(len(images)))
            if tag not in best or k > best[tag][0]:
                best[tag] = (k, arm, mat)
        matrices = {f"{tag} [{arm}]": mat for tag, (_, arm, mat) in best.items()}
    if not matrices:
        print(f"no {args.protocol} results in {args.results_dir}", file=sys.stderr)
        return 1
    if args.baselines:
        matrices.update(build_baselines(gold, args.seed))

    n = len(images)
    rng = np.random.default_rng(args.seed)
    # One shared set of replicates across models, so the CIs are paired and a between-model
    # difference can be read off the same resamples rather than assuming independence.
    draws = [rng.integers(0, n, size=n) for _ in range(args.replicates)]
    full = np.arange(n)

    rows = []
    for tag, mat in sorted(matrices.items()):
        point = kappa_cal(mat, human, gold, reliable_idx, full)
        row = {"arm": tag, "kappa": point}
        line = f"{tag:34s} {point:.3f}"
        if args.replicates:
            reps = np.array([kappa_cal(mat, human, gold, reliable_idx, d) for d in draws])
            lo, hi = np.percentile(reps, [2.5, 97.5])
            row.update({"ci_lo": float(lo), "ci_hi": float(hi),
                        "se": float(reps.std(ddof=1)), "width": float(hi - lo)})
            line += (f"  95% CI [{lo:.3f}, {hi:.3f}]  width {hi - lo:.3f}"
                     f"  se {reps.std(ddof=1):.3f}")
        if args.tie_sensitivity:
            trng = np.random.default_rng(args.seed + 1)
            ties = np.array([kappa_cal(mat, human, gold, reliable_idx, full, rng=trng)
                             for _ in range(args.tie_sensitivity)])
            row.update({"tie_fraction": tie_fraction(mat, reliable_idx),
                        "tie_sd": float(ties.std(ddof=1)),
                        "tie_range": float(ties.max() - ties.min()),
                        "tie_mean": float(ties.mean())})
            line += (f"  | ties {row['tie_fraction']:.3f}  tie-sd {row['tie_sd']:.3f}"
                     f"  tie-range {row['tie_range']:.3f}")
        rows.append(row)
        print(line, flush=True)

    models = [r for r in rows if not r["arm"].startswith("baseline")]
    k = np.array([r["kappa"] for r in models])
    summary = {"between_model_sd": float(k.std(ddof=1))}
    print(f"\nbetween-model sd {k.std(ddof=1):.3f}")

    sep = total = None
    if args.replicates:
        w = float(np.median([r["width"] for r in models]))
        summary["median_ci_width"] = w
        print(f"median CI width {w:.3f}  ->  spread is "
              f"{'WIDER than' if k.std(ddof=1) > w else 'INSIDE'} the CI")
        # Pairwise separation: how much of the ranking survives its own error bars.
        order = sorted(models, key=lambda r: -r["kappa"])
        sep = sum(1 for i in range(len(order)) for j in range(i + 1, len(order))
                  if order[i]["ci_lo"] > order[j]["ci_hi"])
        total = len(order) * (len(order) - 1) // 2
        print(f"non-overlapping CI pairs: {sep}/{total}")
    if args.tie_sensitivity:
        worst = max(models, key=lambda r: r["tie_sd"])
        summary["max_tie_sd"] = worst["tie_sd"]
        print(f"largest tie-break sd: {worst['arm']} {worst['tie_sd']:.3f} "
              f"(range {worst['tie_range']:.3f})")

    json.dump({"results_dir": str(args.results_dir), "protocol": args.protocol,
               "replicates": args.replicates, "tie_sensitivity": args.tie_sensitivity,
               "rows": rows, "separated_pairs": sep, "total_pairs": total, **summary},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
