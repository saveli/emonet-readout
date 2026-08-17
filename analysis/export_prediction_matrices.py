#!/usr/bin/env python
"""Bundle the verification-arm predictions into one compressed archive.

The raw `*__verify_eval.json` files are 62 MB across the two arms and ~97% of that is
float repr and forty repeated key strings per image. Everything downstream of scoring
needs only the (2500 x 40) matrix of P(yes), so the archive is 3.7 MB and the whole
analysis stack -- report, held-out calibration, bootstrap, tie sensitivity -- reruns from
it on CPU with no cluster access.

Same reasoning as `data/faces_siglip2_emb.npz`, which is committed so §D2's control
reruns without a GPU. The raw JSONs stay on the compute cluster and stay gitignored: they are
regenerable from this archive plus the scoring code for every purpose except re-reading a
model's literal text output, which the verification protocol does not produce anyway
(`raw_response` is empty -- nothing is sampled).

Keys are `<arm-dir>/<model-tag>` for the matrices, `<arm-dir>/<model-tag>::idx` for the
image indices they correspond to, and `__emotions__` for the 40 column names. Indices are
stored rather than assumed contiguous, so a partial run round-trips honestly.

Usage:
    python analysis/export_prediction_matrices.py
    python analysis/export_prediction_matrices.py --dirs results_verify results_e1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", type=Path,
                    default=[Path("results_verify"), Path("results_verify_prefill")])
    ap.add_argument("--out", type=Path, default=Path("data/verify_predictions.npz"))
    args = ap.parse_args()

    bundle: dict[str, np.ndarray] = {}
    emotions = None
    for d in args.dirs:
        for path in sorted(d.glob("*__verify_eval.json")):
            payload = json.load(open(path))
            rows = sorted(payload["results"], key=lambda r: r["image_index"])
            if not rows:
                continue
            cols = list(rows[0]["predicted_emotions"])
            if emotions is None:
                emotions = cols
            elif cols != emotions:
                raise SystemExit(f"{path}: emotion column order differs from the first file")
            tag = path.name.replace("__verify_eval.json", "")
            key = f"{d.name}/{tag}"
            bundle[key] = np.array([[r["predicted_emotions"][e] for e in cols] for r in rows],
                                   dtype=np.float32)
            bundle[f"{key}::idx"] = np.array([r["image_index"] for r in rows], dtype=np.int32)
            print(f"{key:52s} {bundle[key].shape}")
    if not bundle:
        raise SystemExit("no verify_eval.json files found")
    bundle["__emotions__"] = np.array(emotions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **bundle)
    n = len([k for k in bundle if "::" not in k and not k.startswith("__")])
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB, {n} matrices)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
