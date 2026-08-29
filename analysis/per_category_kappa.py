#!/usr/bin/env python
"""Per-category kappa_w beside per-category rater agreement.

The paper's conclusion asks benchmarks to "report agreement per category beside performance
per category", and the submitted version did not do it: alpha per category was in the
supplementary figure, performance per category only as a Spearman correlation in the rank
appendix. This script closes that gap, and it costs no new quantity, because the reported
numbers are already per-category means.

`human_human_kappa` and `model_single_rater_kappa` are per-emotion functions, pooled over
rater pairs, averaged over a category set by the caller (see `e1_report.py`). So the mean of
the five alpha>=0.3 categories IS the 0.468 anchor and the mean over all 40 IS 0.204, and the
same holds per model against `verify_prefill_report.json`. Both identities are asserted below:
if the scoring path ever drifts, this script fails rather than drawing a figure that
contradicts the body.

The verification arm read here is the prefill arm for all eleven models, which is the arm the
paper reports (an `Answer:` prefill is applied to every model, sec:verification).

Usage:
    python analysis/per_category_kappa.py
    python analysis/per_category_kappa.py --hq-csv /tmp/hq.csv
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
    model_single_rater_kappa,
    parse_into_arrays,
    quantile_bin,
    scan_index,
)

ARM_DIR = "results_verify_prefill"

# What the body reports, and what this script has to reproduce from the per-category values.
ANCHOR_RELIABLE = 0.468
ANCHOR_ALL40 = 0.204
TOLERANCE = 0.001


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hq-csv", type=Path, default=Path("data/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--npz", type=Path, default=Path("data/verify_predictions.npz"))
    ap.add_argument("--arm-dir", default=ARM_DIR)
    ap.add_argument("--report", type=Path, default=Path("results/verify_prefill_report.json"),
                    help="per-model kappa to check the per-category means against")
    ap.add_argument("--out", type=Path, default=Path("results/per_category_kappa.json"))
    args = ap.parse_args()

    if not args.hq_csv.exists():
        print(f"missing {args.hq_csv} -- fetch hq.csv from the emonet-face repo", file=sys.stderr)
        return 1

    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, _ = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]
    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)
    n_emo = len(emotions)

    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j]) for j in range(n_emo)])
    reliable = alpha >= RELIABLE_ALPHA
    anchor = np.array([human_human_kappa(human[:, j, :]) for j in range(n_emo)])

    z = np.load(args.npz, allow_pickle=True)
    npz_emotions = list(z["__emotions__"])
    col = {e: k for k, e in enumerate(npz_emotions)}
    missing = [e for e in short if e not in col]
    if missing:
        raise SystemExit(f"{len(missing)} emotions absent from {args.npz}: {missing[:3]}")

    models = sorted(k for k in z.files
                    if k.startswith(f"{args.arm_dir}/") and not k.endswith("::idx"))
    if not models:
        raise SystemExit(f"no {args.arm_dir} matrices in {args.npz}")

    # Per model, per category: calibrate against that category's gold marginal, then score
    # against INDIVIDUAL raters -- the same two calls `e1_report.py` makes, in that order.
    per_model: dict[str, np.ndarray] = {}
    for key in models:
        idx = z[f"{key}::idx"]
        mat = np.full((len(images), len(npz_emotions)), np.nan, dtype=np.float64)
        mat[idx] = z[key]
        kap = np.full(n_emo, np.nan)
        for j in range(n_emo):
            pred = mat[:, col[short[j]]]
            kap[j] = model_single_rater_kappa(quantile_bin(pred, gold[:, j]), human[:, j, :])
        per_model[key.split("/", 1)[1]] = kap

    stack = np.vstack(list(per_model.values()))
    cohort_mean = np.nanmean(stack, axis=0)

    # --- identities. A per-category figure that does not average to the body is a wrong
    # figure, so fail here rather than downstream.
    checks: list[tuple[str, float, float]] = [
        ("anchor, alpha>=0.3", float(np.nanmean(anchor[reliable])), ANCHOR_RELIABLE),
        ("anchor, all 40", float(np.nanmean(anchor)), ANCHOR_ALL40),
    ]
    if args.report.exists():
        published = {r["arm"]: r["kappa_calibrated"] for r in json.load(open(args.report))["rows"]}
        for name, kap in per_model.items():
            if name in published:
                checks.append((f"{name}, alpha>=0.3",
                               float(np.nanmean(kap[reliable])), published[name]))

    failed = [(what, got, want) for what, got, want in checks if abs(got - want) > TOLERANCE]
    for what, got, want in checks:
        flag = "FAIL" if abs(got - want) > TOLERANCE else "ok"
        print(f"{what:44s} {got:.4f} vs {want:.4f}  {flag}")
    if failed:
        print(f"\n{len(failed)} identity check(s) failed -- the scoring path drifted; the "
              f"per-category values do not average to what the paper reports.", file=sys.stderr)
        return 1

    rows = [{
        "emotion": short[j],
        "alpha": None if np.isnan(alpha[j]) else float(alpha[j]),
        "anchor_kappa": None if np.isnan(anchor[j]) else float(anchor[j]),
        "verify_kappa_mean": None if np.isnan(cohort_mean[j]) else float(cohort_mean[j]),
        "verify_kappa_min": float(np.nanmin(stack[:, j])),
        "verify_kappa_max": float(np.nanmax(stack[:, j])),
        "verify_kappa_per_model": {n: (None if np.isnan(k[j]) else float(k[j]))
                                   for n, k in per_model.items()},
        "n_models_below_anchor": int(sum(k[j] < anchor[j] for k in per_model.values())),
        "reliable": bool(reliable[j]),
    } for j in range(n_emo)]
    rows.sort(key=lambda r: -(r["alpha"] if r["alpha"] is not None else -9))

    payload = {
        "arm_dir": args.arm_dir,
        "n_models": len(per_model),
        "models": sorted(per_model),
        "reliable_threshold": RELIABLE_ALPHA,
        "anchor_reliable": float(np.nanmean(anchor[reliable])),
        "anchor_all40": float(np.nanmean(anchor)),
        "verify_reliable": float(np.nanmean(cohort_mean[reliable])),
        "verify_all40": float(np.nanmean(cohort_mean)),
        "per_model_reliable": {n: float(np.nanmean(k[reliable])) for n, k in per_model.items()},
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=1))

    print(f"\n{'category':28s}{'alpha':>8}{'anchor':>9}{'verify':>9}{'range':>16}")
    for r in rows:
        rng = f"{r['verify_kappa_min']:.3f}-{r['verify_kappa_max']:.3f}"
        print(f"{r['emotion'][:28]:28s}{r['alpha']:8.3f}{r['anchor_kappa']:9.3f}"
              f"{r['verify_kappa_mean']:9.3f}{rng:>16s}")
    print(f"\nwrote {args.out}  ({len(per_model)} models, {n_emo} categories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
