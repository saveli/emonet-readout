#!/usr/bin/env python
"""Score the benchmark's OWN published baselines through the same scoring path we use.

Everything else in this project evaluates eleven checkpoints the benchmark never ran, which
leaves a gap a reviewer names immediately: we would have shown "these models score well under
verification", not "the benchmark's models were mismeasured". The benchmark ships its
fourteen baselines' per-image predictions in `hq.csv`, so the gap closes without a GPU.

The zero-shot prompt demands "exactly 25 key-value pairs" with zeros omitted. Several
baselines complied exactly, on every image, with zero variance -- instruction-following
rather than perception. The decisive control is one the benchmark itself supplies: Gemini 2.5
Flash appears twice, under a zero-shot and a multi-step prompt. Same model, same images, same
metric; only the elicitation differs.

Cardinality is not the whole mechanism, and this script is built so that shows: Hume Face*
emits a constant 31 and still scores well, because continuous scores rank informatively
regardless of how many are nonzero. The defensible claim is not "emitting 25 is bad" but that
constant-cardinality compliance displaces the signal, which a rank-sensitive readout exposes.

IMPORTANT -- this reuses `reliability_vs_performance`'s scoring rather than reimplementing it.
An earlier version rolled its own calibration, pooling emotions and scoring against the
rounded rater mean, and produced numbers disagreeing with the rest of the project by up to
0.14 kappa. The project calibrates PER EMOTION and scores against INDIVIDUAL raters, then
averages over the reliable categories; anything else is a different estimator wearing the
same name.

Usage:
    python analysis/published_baselines.py --hq-csv data/hq.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv.field_size_limit(10**7)

from reliability_vs_performance import (  # noqa: E402
    RELIABLE_ALPHA,
    human_human_kappa,
    is_human,
    krippendorff_alpha,
    model_single_rater_kappa,
    parse_into_arrays,
    quantile_bin,
    scan_index,
)


def cardinality(hq_csv: Path) -> dict[str, list[int]]:
    """Nonzero emotions per image per annotator -- the compliance signature."""
    card: dict[str, list[int]] = {}
    for row in csv.DictReader(open(hq_csv)):
        try:
            ann = ast.literal_eval(row["annotations"])
        except (ValueError, SyntaxError):
            continue
        if not isinstance(ann, dict):
            continue
        name = "HUMAN" if is_human(row) else row["annotator"]
        card.setdefault(name, []).append(
            sum(1 for v in ann.values() if isinstance(v, (int, float)) and v > 0))
    return card


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hq-csv", type=Path, default=Path("data/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--out", type=Path, default=Path("analysis/published_baselines.json"))
    args = ap.parse_args()

    index_map = json.load(open(args.index_map)) if args.index_map.is_file() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, preds = parse_into_arrays(args.hq_csv, emotions, images, annotators)

    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j])
                      for j in range(len(emotions))])
    reliable = alpha >= RELIABLE_ALPHA
    anchor = float(np.nanmean([human_human_kappa(human[:, j, :])
                               for j in range(len(emotions)) if reliable[j]]))
    print(f"reliable categories (alpha >= {RELIABLE_ALPHA}): {int(reliable.sum())}")
    print(f"human-human anchor: {anchor:.3f}")

    gold = np.round(np.nanmean(human, axis=2)).astype(int)
    card = cardinality(args.hq_csv)

    rows = []
    for name, ai in annotators.items():
        mat = preds[ai]
        ks = [model_single_rater_kappa(quantile_bin(mat[:, j], gold[:, j]), human[:, j, :])
              for j in range(len(emotions)) if reliable[j]]
        c = card.get(name, [])
        rows.append({
            "annotator": name,
            "kappa_w": float(np.nanmean(ks)) if ks else float("nan"),
            "mean_nonzero": statistics.mean(c) if c else float("nan"),
            "sd_nonzero": statistics.pstdev(c) if len(c) > 1 else 0.0,
        })
    rows.sort(key=lambda r: -(r["kappa_w"] if np.isfinite(r["kappa_w"]) else -9))

    print(f"\n{'annotator':<34}{'kappa_w':>9}{'nonzero/img':>13}{'sd':>7}")
    for r in rows:
        print(f"{r['annotator']:<34}{r['kappa_w']:>9.3f}"
              f"{r['mean_nonzero']:>13.2f}{r['sd_nonzero']:>7.2f}")
    hc = card.get("HUMAN", [])
    print(f"{'human experts (anchor)':<34}{anchor:>9.3f}"
          f"{statistics.mean(hc):>13.2f}{statistics.pstdev(hc):>7.2f}")

    args.out.write_text(json.dumps(
        {"anchor": float(anchor), "reliable_alpha": RELIABLE_ALPHA,
         "n_reliable": int(reliable.sum()), "rows": rows,
         "human_nonzero_mean": statistics.mean(hc),
         "human_nonzero_sd": statistics.pstdev(hc)}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
