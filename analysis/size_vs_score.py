#!/usr/bin/env python
"""Does parameter count predict the verification score in this cohort?

The models were chosen as the newest per family that fits one consumer GPU, which spreads
them over an order of magnitude (1.30B to 13.95B) without that spread being an experimental
variable. This script asks whether the spread explains anything, so the paper can say what it
found rather than leaving the reader to wonder.

It does not: Spearman rho is around 0.3 with p well above any threshold at n=11. The honest
reading is "no relationship detectable at this cohort size", NOT "size does not matter" --
eleven points have very little power, and the leakage-adjusted variant below moves rho UP,
so the data do not exclude a real effect either.

Two robustness points are computed rather than asserted:

  * The oracle calibration inflates two models appreciably (see sec:limitations). Since the
    smallest checkpoint is one of them, any "the little one keeps up" reading rests on the
    single most leakage-exposed number in the table. Subtracting each model's own measured
    leakage is the conservative variant, and it is reported beside the headline one.
  * Both category sets are reported, because the five and the forty rank the models
    differently and a correlation on one is not a correlation on the other.

Parameter counts are the HuggingFace `safetensors.total` for each repo, the same source as
`tab:models` and `docs/model_selection.md`; they are not recoverable from any results JSON,
so they are listed here and checked against the scored arms.

Usage:
    python analysis/size_vs_score.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

# HF safetensors.total, in billions. Mirrors tab:models and docs/model_selection.md.
PARAMS_B = {
    "GLM-4.6V-Flash": 10.29,
    "InternVL3_5-8B-HF": 8.53,
    "MiMo-VL-7B-RL-2508": 8.31,
    "MiniCPM-V-4.6": 1.30,
    "Ministral-3-14B-Instruct-2512": 13.95,
    "Qwen2.5-VL-3B-Instruct": 3.75,
    "Qwen3-VL-8B-Instruct": 8.77,
    "Qwen3.5-9B": 9.65,
    "gemma-3-4b-it": 4.30,
    "gemma-3-12b-it": 12.19,
    "gemma-4-12B-it": 11.96,
}


def report(label: str, xs, ys) -> dict:
    rho, p_rho = spearmanr(xs, ys)
    r, p_r = pearsonr(xs, ys)
    print(f"{label:34s} rho={rho:+.3f} (p={p_rho:.3f})   r={r:+.3f} (p={p_r:.3f})   n={len(xs)}")
    return {"label": label, "n": len(xs), "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "pearson_r": float(r), "pearson_p": float(p_r)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=Path("results/verify_prefill_report.json"),
                    help="per-model kappa on the five reliable categories")
    ap.add_argument("--percat", type=Path, default=Path("results/per_category_kappa.json"),
                    help="per-category kappa, used for the all-40 means")
    ap.add_argument("--heldout", type=Path,
                    default=Path("results/verify_prefill_heldout_calibration_all11.json"))
    ap.add_argument("--out", type=Path, default=Path("results/size_vs_score.json"))
    args = ap.parse_args()

    five = {r["arm"]: r["kappa_calibrated"] for r in json.load(open(args.report))["rows"]}
    leak = {r["arm"]: r["leakage"] for r in json.load(open(args.heldout))["rows"]}
    percat = json.load(open(args.percat))
    all40 = {m: sum(row["verify_kappa_per_model"][m] for row in percat["rows"]) / len(percat["rows"])
             for m in percat["models"]}

    missing = [m for m in PARAMS_B if m not in five or m not in all40 or m not in leak]
    if missing:
        print(f"scored arms do not match the parameter list: {missing}", file=sys.stderr)
        return 1

    models = sorted(PARAMS_B, key=lambda m: PARAMS_B[m])
    xs = [PARAMS_B[m] for m in models]
    print(f"{len(models)} models, {min(xs):.2f}B to {max(xs):.2f}B "
          f"({max(xs) / min(xs):.1f}x)\n")

    rows = [report("five reliable, as reported", xs, [five[m] for m in models]),
            report("five reliable, leakage-adjusted", xs, [five[m] - leak[m] for m in models]),
            report("all 40, as reported", xs, [all40[m] for m in models])]

    print(f"\n{'model':32s}{'params':>9}{'kappa(5)':>10}{'leakage':>9}")
    for m in models:
        print(f"{m[:32]:32s}{PARAMS_B[m]:8.2f}B{five[m]:10.3f}{leak[m]:9.3f}")

    json.dump({"params_b": PARAMS_B, "correlations": rows,
               "kappa_five": five, "kappa_all40": all40, "leakage": leak},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
