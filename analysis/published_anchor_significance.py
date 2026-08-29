#!/usr/bin/env python
"""Are the benchmark's own published baselines above the human anchor, or just near it?

`anchor_significance.py` answers this for our eleven verification arms. This script answers
it for entries in the benchmark's own `hq.csv`, which is what \\cref{sec:published} needs: the
body claims one of their baselines clears the anchor and another is indistinguishable from
it, and both halves of that sentence are interval claims, not comparisons of printed decimals
(Hume's 0.466 against an anchor of 0.468 is a 0.002 gap -- eyeballing it either way is
indefensible).

Same protocol as `anchor_significance.py`, and deliberately the same code path: images are
resampled once per replicate, BOTH the baseline and the anchor are recomputed on that
resample so the difference is paired, and the quantile calibration is refit inside every
replicate. The anchor is recomputed once per replicate and shared across targets rather than
recomputed per target, which is the only difference and is arithmetic, not statistics.

Step 1 reproduces the published five-category anchor and aborts if it disagrees, for the
reason given at length in `anchor_significance.py`: a second implementation that quietly
disagrees with the first is the failure mode this project keeps hitting.

Usage:
    python analysis/published_anchor_significance.py
    python analysis/published_anchor_significance.py --targets "Hume Face*" -B 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reliability_vs_performance import (  # noqa: E402
    RELIABLE_ALPHA,
    human_human_kappa,
    krippendorff_alpha,
    parse_into_arrays,
    scan_index,
)
from verify_bootstrap_ci import kappa_cal  # noqa: E402

# The entries the body names. Others can be passed on the command line.
DEFAULT_TARGETS = ["Gemini 2.0 Flash ZS", "Hume Face*", "Empathic Insight Face Small"]

PAPER_ANCHOR = 0.468
ANCHOR_TOL = 0.0005


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hq-csv", type=Path, default=Path("data/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--out", type=Path, default=Path("results/published_anchor_significance.json"))
    ap.add_argument("-B", "--replicates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, preds = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j])
                      for j in range(len(emotions))])
    rel = [j for j in range(len(emotions)) if alpha[j] >= RELIABLE_ALPHA]
    n = len(images)

    def anchor_on(rows):
        return float(np.nanmean([human_human_kappa(human[rows, j, :]) for j in rel]))

    def baseline_on(mat, rows):
        return float(np.nanmean([kappa_cal(mat, human, gold, [j], rows) for j in rel]))

    full = np.arange(n)
    anchor_pt = anchor_on(full)
    print(f"anchor, {len(rel)} reliable categories: {anchor_pt:.4f} against a published "
          f"{PAPER_ANCHOR}")
    if abs(anchor_pt - PAPER_ANCHOR) > ANCHOR_TOL:
        print("STOP: the recomputed anchor does not reproduce the published value.",
              file=sys.stderr)
        return 1

    missing = [t for t in args.targets if t not in annotators]
    if missing:
        print(f"not in {args.hq_csv}: {missing}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    draws = [rng.integers(0, n, n) for _ in range(args.replicates)]
    points = {t: baseline_on(preds[annotators[t]], full) for t in args.targets}
    deltas = {t: [] for t in args.targets}

    for b, rows in enumerate(draws, 1):
        a = anchor_on(rows)                      # once per replicate, shared across targets
        for t in args.targets:
            deltas[t].append(baseline_on(preds[annotators[t]], rows) - a)
        if b % 100 == 0:
            print(f"  {b}/{args.replicates} replicates", file=sys.stderr)

    rows_out = []
    for t in args.targets:
        d = np.array(deltas[t])
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        rows_out.append({
            "annotator": t,
            "kappa_w": points[t],
            "delta_vs_anchor": points[t] - anchor_pt,
            "ci_lo": lo,
            "ci_hi": hi,
            "separated": bool(lo > 0 or hi < 0),
        })

    payload = {
        "anchor": anchor_pt,
        "n_reliable": len(rel),
        "replicates": args.replicates,
        "seed": args.seed,
        "rows": rows_out,
    }
    args.out.write_text(json.dumps(payload, indent=1))

    print(f"\n{'annotator':30}{'kappa':>8}{'delta':>9}{'95% CI':>22}   verdict")
    for r in rows_out:
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        verdict = ("above the anchor" if r["ci_lo"] > 0 else
                   "below the anchor" if r["ci_hi"] < 0 else
                   "indistinguishable from the anchor")
        print(f"{r['annotator'][:30]:30}{r['kappa_w']:8.3f}{r['delta_vs_anchor']:+9.3f}"
              f"{ci:>22}   {verdict}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
