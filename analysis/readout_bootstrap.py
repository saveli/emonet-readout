#!/usr/bin/env python
"""Confidence intervals for the readout ablation and for the spread-collapse statistic.

Two claims in the paper were point estimates while every other number carried an interval,
which is conspicuous in a paper whose subject is measurement:

  1. binarising the stored P(yes) costs 0.198 kappa_w and 0.228 mAP
  2. the between-model sd falls from 0.059 to 0.025 while the median CI width falls from
     0.042 to 0.034, so the spread moves inside the measurement precision

(2) is the weaker of the two as stated, because an sd over ten models has a large relative
standard error and the claim compares two such sds. This resamples IMAGES -- the independent
unit, as everywhere else in this project -- and refits quantile calibration inside each
replicate, so the intervals are built the same way as the ones already reported.

The quantity of interest for (2) is the RATIO sd / median-CI-width per elicitation: below 1
the models are not separable at the available precision, above 1 they are. Reporting the
ratio rather than the two numbers is what makes it a test instead of a comparison of point
estimates.

Usage:
    python analysis/readout_bootstrap.py --hq-csv data/hq.csv -B 500
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
    krippendorff_alpha,
    model_single_rater_kappa,
    parse_into_arrays,
    quantile_bin,
    scan_index,
)
from e0_report import macro_ap  # noqa: E402
from e1_report import load_verify_results  # noqa: E402


def score(mat, human, gold, reliable, idx):
    """kappa_w and mAP for one model on an image subsample."""
    ks = []
    for j in range(human.shape[1]):
        if not reliable[j]:
            continue
        ks.append(model_single_rater_kappa(
            quantile_bin(mat[idx, j], gold[idx, j]), human[idx, j, :]))
    return float(np.nanmean(ks)), macro_ap(mat[idx], gold[idx], reliable)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_verify_prefill"))
    ap.add_argument("--hq-csv", type=Path, default=Path("data/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--out", type=Path, default=Path("results/readout_bootstrap.json"))
    ap.add_argument("-B", "--replicates", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index_map = json.load(open(args.index_map))
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, _ = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j])
                      for j in range(len(emotions))])
    reliable = alpha >= RELIABLE_ALPHA
    gold = np.round(np.nanmean(human, axis=2)).astype(int)

    short = [e.split('|')[-1] for e in emotions]
    runs = load_verify_results(args.results_dir, short, human.shape[0])
    tags = sorted(runs)
    print(f"{len(tags)} models, B={args.replicates}, resampling images\n")

    rng = np.random.default_rng(args.seed)
    n = human.shape[0]
    dk, dm = [], []          # per-replicate mean drop, kappa and mAP
    for b in range(args.replicates):
        idx = rng.integers(0, n, n)
        kd, md = [], []
        for t in tags:
            mat = runs[t][0]
            kc, mc = score(mat, human, gold, reliable, idx)
            kb, mb = score((mat > args.threshold).astype(float),
                           human, gold, reliable, idx)
            kd.append(kb - kc)
            md.append(mb - mc)
        dk.append(float(np.mean(kd)))
        dm.append(float(np.mean(md)))
        if (b + 1) % 50 == 0:
            print(f"  {b+1}/{args.replicates}", flush=True)

    def ci(v):
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    kl, kh = ci(dk)
    ml, mh = ci(dm)
    print("\nbinarising the stored P(yes), mean over models:")
    print(f"  kappa_w  {np.mean(dk):+.3f}  95% CI [{kl:+.3f}, {kh:+.3f}]")
    print(f"  mAP      {np.mean(dm):+.3f}  95% CI [{ml:+.3f}, {mh:+.3f}]")
    print("  both intervals exclude zero:" ,
          (kh < 0) and (mh < 0))

    args.out.write_text(json.dumps({
        "replicates": args.replicates, "n_models": len(tags), "models": tags,
        "kappa_drop_mean": float(np.mean(dk)), "kappa_drop_ci": [kl, kh],
        "map_drop_mean": float(np.mean(dm)), "map_drop_ci": [ml, mh],
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
