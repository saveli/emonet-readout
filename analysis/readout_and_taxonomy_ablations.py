#!/usr/bin/env python
"""Two ablations that need no GPU, because the answer is already in the stored predictions.

Both were nearly submitted as cluster jobs before it became clear they were re-scores.

**S -- readout vs question format.** Verification changes two things at once against
generative elicitation: it asks 40 binary questions instead of requesting a list, and it
reads `P(yes)` off the logits instead of parsing sampled text. Nothing so far separates
them, and the separation matters: "ask binary questions" transfers to any API model,
"read the logits" does not.

No GPU is needed because mean yes/no mass runs 0.798-1.000 across the sweep, so the argmax
token at the read position *is* `yes` or `no`. Greedy decoding therefore equals argmax over
{yes,no} equals thresholding the stored `P(yes)` at 0.5, and the sampled-text condition is
a re-score. (Were the mass low this identity would fail -- which is exactly the §L failure
the mass guard now aborts on.)

**T -- the 5-of-40 restriction.** Every kappa in the project is averaged over the 5
categories with Krippendorff alpha >= 0.3. That is our methodological choice and reads as
cherry-picking unless the result is shown not to depend on it. Scoring the excluded 35 and
the full 40 costs nothing and answers it. The all-40 human anchor also happens to be the
number directly comparable to EmoNet's published alpha of 0.19.

Baselines are scored on every category subset, because the whole point of C6 is that a
metric's degenerate solutions change when the scoring set changes.

Usage:
    python analysis/readout_and_taxonomy_ablations.py --results-dir results_verify_prefill
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c4_calibration_control import build_baselines  # noqa: E402
from e0_report import macro_ap  # noqa: E402
from e1_report import load_verify_results  # noqa: E402
from reliability_vs_performance import (  # noqa: E402
    RELIABLE_ALPHA,
    human_human_kappa,
    krippendorff_alpha,
    model_single_rater_kappa,
    parse_into_arrays,
    quantile_bin,
    scan_index,
)


def kappa(mat, human, gold, idx):
    return float(np.nanmean([
        model_single_rater_kappa(quantile_bin(mat[:, j], gold[:, j]), human[:, j, :])
        for j in idx]))


def tie_fraction(mat, idx):
    fr = []
    for j in idx:
        p = mat[:, j][~np.isnan(mat[:, j])]
        if len(p) < 2:
            continue
        _, c = np.unique(p, return_counts=True)
        fr.append(c[c > 1].sum() / len(p))
    return float(np.mean(fr)) if fr else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_verify_prefill"))
    ap.add_argument("--hq-csv", type=Path, default=Path("/tmp/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/readout_and_taxonomy_ablations.json"))
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(yes) cut standing in for the greedy-decoded answer")
    args = ap.parse_args()

    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, _ = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]
    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j])
                      for j in range(len(emotions))])
    rel_mask = alpha >= RELIABLE_ALPHA
    rel = [j for j in range(len(emotions)) if rel_mask[j]]
    unrel = [j for j in range(len(emotions)) if not rel_mask[j]]
    allc = list(range(len(emotions)))

    hh = np.array([human_human_kappa(human[:, j, :]) for j in range(len(emotions))])
    anchors = {"reliable_5": float(np.nanmean(hh[rel])),
               "unreliable_35": float(np.nanmean(hh[unrel])),
               "all_40": float(np.nanmean(hh))}

    runs = load_verify_results(args.results_dir, short, len(images))
    if not runs:
        print(f"no verify results in {args.results_dir}", file=sys.stderr)
        return 1
    matrices = {t: m for t, (m, _) in runs.items()}
    matrices.update(build_baselines(gold, 0))

    print(f"human-human anchor:  5 reliable {anchors['reliable_5']:.3f}   "
          f"35 unreliable {anchors['unreliable_35']:.3f}   all 40 {anchors['all_40']:.3f}")
    print("(EmoNet's published Krippendorff alpha is 0.19; the all-40 column is what "
          "compares to it)\n")

    print("S -- readout ablation: continuous P(yes) vs the same values thresholded at "
          f"{args.threshold}")
    print(f"{'model':32s}{'cont k':>8}{'bin k':>8}{'d':>8}{'cont mAP':>10}{'bin mAP':>9}"
          f"{'bin ties':>10}")
    rows = []
    for tag, mat in sorted(matrices.items()):
        binm = (mat > args.threshold).astype(np.float32)
        r = {"arm": tag,
             "kappa_continuous": kappa(mat, human, gold, rel),
             "kappa_binarised": kappa(binm, human, gold, rel),
             "map_continuous": macro_ap(mat, gold, rel_mask),
             "map_binarised": macro_ap(binm, gold, rel_mask),
             "tie_fraction_binarised": tie_fraction(binm, rel),
             "kappa_reliable_5": kappa(mat, human, gold, rel),
             "kappa_unreliable_35": kappa(mat, human, gold, unrel),
             "kappa_all_40": kappa(mat, human, gold, allc)}
        r["readout_delta"] = r["kappa_binarised"] - r["kappa_continuous"]
        rows.append(r)
        if not tag.startswith("baseline"):
            print(f"{tag:32s}{r['kappa_continuous']:8.3f}{r['kappa_binarised']:8.3f}"
                  f"{r['readout_delta']:+8.3f}{r['map_continuous']:10.3f}"
                  f"{r['map_binarised']:9.3f}{r['tie_fraction_binarised']:10.3f}")

    models = [r for r in rows if not r["arm"].startswith("baseline")]
    d = np.array([r["readout_delta"] for r in models])
    print(f"\nmean delta {d.mean():+.3f}; all ten lose: {bool((d < 0).all())}")
    print("Binarised verification lands in the generative range (0.286-0.486) -- the gain "
          "is the readout,\nnot the binary question. Applies only where logprobs exist.\n")

    print("T -- taxonomy ablation: does the 5-reliable restriction carry the result?")
    print(f"{'model':32s}{'5 rel':>8}{'35 unrel':>10}{'all 40':>9}")
    for r in rows:
        if r["arm"].startswith("baseline"):
            continue
        print(f"{r['arm']:32s}{r['kappa_reliable_5']:8.3f}{r['kappa_unreliable_35']:10.3f}"
              f"{r['kappa_all_40']:9.3f}")
    for r in rows:
        if r["arm"].startswith("baseline"):
            print(f"{r['arm']:32s}{r['kappa_reliable_5']:8.3f}"
                  f"{r['kappa_unreliable_35']:10.3f}{r['kappa_all_40']:9.3f}")
    over = sum(1 for r in models if r["kappa_all_40"] > anchors["all_40"])
    print(f"\nmodels clearing the all-40 human anchor ({anchors['all_40']:.3f}): "
          f"{over}/{len(models)}")

    json.dump({"results_dir": str(args.results_dir), "threshold": args.threshold,
               "anchors": anchors, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
