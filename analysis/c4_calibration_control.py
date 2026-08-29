#!/usr/bin/env python
"""C4/C6 -- how much does per-emotion calibration alone buy, and do trivial baselines beat it?

Before any training claim is credible, two cheap controls have to be cleared:

  C4  Per-emotion quantile calibration with no training at all. If simply rank-matching a
      model's raw scores to the ground-truth marginal recovers most of the gain a
      fine-tune produces, the fine-tune is not contributing much.
  C6  Trivial baselines (all-zero, ground-truth prior, random, all-one) carried through
      the identical metric path, so a degenerate solution cannot masquerade as a result.

Both run on cached predictions -- no GPU. Reuses the loaders and metrics from
reliability_vs_performance.py so the numbers are directly comparable.

Usage:
    python analysis/c4_calibration_control.py --hq-csv /tmp/hq.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reliability_vs_performance import (  # noqa: E402
    N_LEVELS,
    RELIABLE_ALPHA,
    krippendorff_alpha,
    load_local_models,
    model_single_rater_kappa,
    parse_into_arrays,
    quantile_bin,
    scan_index,
)


def raw_bin(pred: np.ndarray) -> np.ndarray:
    """Uncalibrated mapping: clip the model's own scale onto 0..7 and round.

    This is what you get by taking a model's emitted intensities at face value, which is
    what the original pipeline did.
    """
    out = np.full(pred.shape, np.nan)
    ok = ~np.isnan(pred)
    out[ok] = np.clip(np.round(pred[ok]), 0, N_LEVELS - 1)
    return out


def build_baselines(gold: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """C6 trivial predictors, on the same (image, emotion) grid as the models."""
    rng = np.random.default_rng(seed)
    n, k = gold.shape
    present = gold > 0
    mean_k = max(1, int(round(np.nanmean(present.sum(axis=1)))))

    zero = np.zeros_like(gold)
    prior = np.zeros_like(gold)
    prior[:, np.argsort(np.nanmean(present, axis=0))[::-1][:mean_k]] = 1.0
    rand = rng.random((n, k)).astype(np.float32)
    allone = np.ones_like(gold)
    # Per-emotion mean rating repeated for every image: the strongest "no vision" predictor.
    const = np.tile(np.nan_to_num(np.nanmean(gold, axis=0)), (n, 1)).astype(np.float32)
    return {
        "baseline:all-zero": zero,
        "baseline:gt-prior": prior,
        "baseline:random": rand,
        "baseline:all-one": allone,
        "baseline:per-emotion-mean": const,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hq-csv", type=Path, default=Path("/tmp/hq.csv"))
    ap.add_argument("--results-dir", type=Path, nargs="+",
                    default=[Path("results"), Path("results_dpo"), Path("results_e0")])
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--out", type=Path, default=Path("results/c4_calibration_control.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, preds = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]
    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j]) for j in range(len(emotions))])
    reliable = alpha >= RELIABLE_ALPHA
    print(f"{len(images)} images | {len(emotions)} emotions | "
          f"{int(reliable.sum())} reliable (alpha>={RELIABLE_ALPHA})")

    matrices = {f"[paper] {n}": preds[m] for n, m in annotators.items()}
    matrices.update({f"[ours] {n}": m
                     for n, m in load_local_models(args.results_dir, short, len(images)).items()})
    matrices.update(build_baselines(gold, args.seed))

    rows = []
    for name, mat in matrices.items():
        raw_k, cal_k = [], []
        for j in range(len(emotions)):
            if not reliable[j]:
                continue
            raw_k.append(model_single_rater_kappa(raw_bin(mat[:, j]), human[:, j, :]))
            cal_k.append(model_single_rater_kappa(quantile_bin(mat[:, j], gold[:, j]), human[:, j, :]))
        raw_m = float(np.nanmean(raw_k)) if raw_k else float("nan")
        cal_m = float(np.nanmean(cal_k)) if cal_k else float("nan")
        rows.append({"model": name, "kappa_raw": raw_m, "kappa_calibrated": cal_m,
                     "calibration_gain": cal_m - raw_m})

    rows.sort(key=lambda r: -(r["kappa_calibrated"] if r["kappa_calibrated"] == r["kappa_calibrated"] else -9))
    print(f"\n{'model':44}{'raw':>9}{'calibrated':>12}{'gain':>9}")
    for r in rows:
        print(f"{r['model'][:44]:44}{r['kappa_raw']:9.3f}{r['kappa_calibrated']:12.3f}"
              f"{r['calibration_gain']:9.3f}")

    real = [r for r in rows if not r["model"].startswith("baseline:")]
    gains = [r["calibration_gain"] for r in real if r["calibration_gain"] == r["calibration_gain"]]
    print(f"\nC4: mean calibration-only gain over {len(gains)} real models = {np.mean(gains):+.3f}")
    print("    Any training arm must beat this margin to be worth reporting.")
    best_base = max((r for r in rows if r["model"].startswith("baseline:")),
                    key=lambda r: r["kappa_calibrated"])
    print(f"C6: best trivial baseline = {best_base['model']} at {best_base['kappa_calibrated']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"reliable_threshold": RELIABLE_ALPHA, "rows": rows}, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
