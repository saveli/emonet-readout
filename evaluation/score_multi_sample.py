#!/usr/bin/env python
"""Aggregate an `ours_multi` run (K sampled completions per image) into an `ours`-format file.

Alternative explanation being tested: sample the generative model K times per image and
average, and the gap to the logit readout closes. e0_prompt_sweep.py --arm ours_multi keeps
all K parses per row under `samples`; this script reduces them per category per image and
writes a results file that analysis/e0_report.py (and everything else that reads `ours`
rows through its load_arm_results) consumes UNCHANGED:

  <out>/mean_k<K>/<tag>__ours_n<N>_eval.json   mean of the parsed samples (the multi arm)
  <out>/sample0/<tag>__ours_n<N>_eval.json     sample 0 alone (the paired single-sample
                                               reference: same images, same decoder)

batch_info.arm is "ours" so e0_report's ARMS filter accepts it; source_arm, aggregate,
n_samples and the image list say what the file really is. Samples that did not parse are
excluded from the mean, and each row records how many entered it (`n_samples_used`); an image
with no parsed sample is written parse_success=false, exactly as an unparsed `ours` row is.

Asserts, before anything is written:
  1. K=1 identity, on the multi file itself: the aggregate of sample 0 alone equals the
     matrix load_arm_results reads from the rows' own single-sample slots.
  2. Round trip: every written file, read back through analysis/e0_report.load_arm_results,
     gives the in-memory matrix bit for bit.
  3. --check-ours DIR: every `ours` file in DIR, pushed through the same aggregation as a
     K=1 input and restricted to --images (default: the multi run's image list), reproduces
     load_arm_results' matrix for that file on the same images exactly.

Usage:
    python evaluation/score_multi_sample.py --in-dir results_e0_multi --out-dir results_e0_multi_scored
    python evaluation/score_multi_sample.py --check-ours results_e0 --check-only
    python analysis/e0_report.py --results-dir results_e0_multi_scored/mean_k16 --hq-csv data/hq.csv \\
        --out results_e0_multi_scored/e0_report_mean_k16.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from e0_report import load_arm_results  # noqa: E402

N_IMAGES = 2500


def emotions_of(payload) -> list[str]:
    """The 40 bare names, from the first parsed row (every parsed row carries all 40 keys)."""
    for r in payload["results"]:
        for s in r.get("samples") or [r]:
            if s.get("parse_success") and s.get("predicted_emotions"):
                return sorted(s["predicted_emotions"])
    raise SystemExit("no parsed row anywhere; cannot recover the emotion list")


def samples_of(row) -> list[dict]:
    """K samples for a multi row; the row itself as one sample for a single-sample row."""
    if row.get("samples") is not None:
        return row["samples"]
    return [{"parse_success": row.get("parse_success"),
             "predicted_emotions": row.get("predicted_emotions")}]


def num(v) -> float:
    # Same coercion as load_arm_results: a non-numeric value counts as 0.
    return float(v) if isinstance(v, (int, float)) else 0.0


def aggregate_row(row, emotions, take: int | None, how: str):
    used = [s for s in samples_of(row)[:take] if s.get("parse_success")]
    if not used:
        return None, 0
    vals = np.array([[num(s["predicted_emotions"].get(e, 0)) for e in emotions] for s in used])
    agg = vals.mean(axis=0) if how == "mean" else np.median(vals, axis=0)
    return {e: float(v) for e, v in zip(emotions, agg)}, len(used)


def aggregate_file(payload, take, how, label):
    """Rows in the shared results/ schema, one per image, scores = aggregate of samples."""
    emotions = emotions_of(payload)
    rows = []
    for r in payload["results"]:
        scores, k_used = aggregate_row(r, emotions, take, how)
        rows.append({
            "image_index": r["image_index"], "image_path": r.get("image_path"),
            "model": r.get("model"), "success": True,
            "parse_success": scores is not None,
            "predicted_emotions": scores if scores is not None else {e: 0 for e in emotions},
            "n_samples_available": len(samples_of(r)[:take]),
            "n_samples_used": k_used,
        })
    info = dict(payload["batch_info"])
    src_arm = info.get("arm")
    nz = [sum(1 for v in r["predicted_emotions"].values() if v > 0)
          for r in rows if r["parse_success"]]
    info.update({
        "arm": "ours", "source_arm": src_arm, "aggregate": label,
        "aggregate_take": take, "successful_parses": sum(r["parse_success"] for r in rows),
        "successful_requests": len(rows),
        "mean_nonzero_emotions": float(np.mean(nz)) if nz else 0.0,
        "mean_samples_used": float(np.mean([r["n_samples_used"] for r in rows])) if rows else 0.0,
    })
    info["model"] = f"{info.get('model_id', '').split('/')[-1]} [ours<-{src_arm}:{label}]"
    return {"batch_info": info, "results": rows}, emotions


def matrix(payload, emotions) -> np.ndarray:
    """load_arm_results' matrix for one payload, via load_arm_results itself."""
    with tempfile.TemporaryDirectory() as td:
        p = dict(payload)
        p["batch_info"] = {**payload["batch_info"], "arm": "ours"}
        json.dump(p, open(Path(td) / "x__ours_eval.json", "w"))
        runs = load_arm_results(Path(td), emotions, N_IMAGES)
    assert len(runs) == 1, runs.keys()
    return next(iter(runs.values()))[0]


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def restrict(payload, images: set[int]):
    return {"batch_info": payload["batch_info"],
            "results": [r for r in payload["results"] if r["image_index"] in images]}


def check_ours(ours_dir: Path, images: set[int] | None) -> int:
    """Assertion 3: a single-sample `ours` file through the K=1 path reproduces its matrix."""
    n = 0
    for path in sorted(ours_dir.glob("*_eval.json")):
        payload = json.load(open(path))
        if payload["batch_info"].get("arm") != "ours":
            continue
        sub = restrict(payload, images) if images is not None else payload
        agg, emotions = aggregate_file(sub, 1, "mean", "mean_k1")
        ref = matrix(sub, emotions)
        got = matrix(agg, emotions)
        assert same(ref, got), f"K=1 aggregate of {path.name} differs from its ours matrix"
        rows = np.isfinite(ref).any(axis=1).sum()
        print(f"[check-ours] {path.name}: K=1 aggregate == ours matrix on {rows} parsed images")
        n += 1
    assert n, f"no `ours` files in {ours_dir}"
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, help="directory holding *__ours_multi_k*_eval.json")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--only-n", type=int, default=None,
                    help="aggregate only runs with this many images (500 = the subset, 2500 = "
                         "the full set); both kinds can sit in --in-dir at once")
    ap.add_argument("--take", type=int, default=None,
                    help="use only the first N samples of each image (K-curves); default all")
    ap.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    ap.add_argument("--check-ours", type=Path, default=None,
                    help="also assert the K=1 path reproduces every `ours` file in this dir")
    ap.add_argument("--check-only", action="store_true", help="run --check-ours and stop")
    args = ap.parse_args()

    if args.check_only:
        assert args.check_ours, "--check-only needs --check-ours"
        check_ours(args.check_ours, None)
        return 0
    assert args.in_dir and args.out_dir, "--in-dir and --out-dir are required"

    files = sorted(args.in_dir.glob("*__ours_multi*_eval.json"))
    assert files, f"no ours_multi results in {args.in_dir}"
    image_sets = []
    for path in files:
        payload = json.load(open(path))
        info = payload["batch_info"]
        assert info.get("prompt_arm") == "ours_multi", path
        if args.only_n is not None and info["total_images"] != args.only_n:
            continue
        k = info["n_samples"]
        bad = [r["image_index"] for r in payload["results"] if len(r.get("samples") or []) != k]
        assert not bad, f"{path.name}: rows without exactly K={k} samples: {bad[:10]}"
        take = min(args.take or k, k)
        label = f"{args.aggregate}_k{take}"
        n = info["total_images"]

        # 1. K=1 identity on this file.
        s0, emotions = aggregate_file(payload, 1, "mean", "sample0")
        assert same(matrix(s0, emotions), matrix(payload, emotions)), \
            f"{path.name}: sample-0 aggregate != the rows' single-sample matrix"

        agg, _ = aggregate_file(payload, take, args.aggregate, label)
        tag = info["model_id"].split("/")[-1]
        for sub, out in ((label, agg), ("sample0", s0)):
            d = args.out_dir / sub
            d.mkdir(parents=True, exist_ok=True)
            dest = d / f"{tag}__ours_n{n}_eval.json"
            json.dump(out, open(dest, "w"))
            # 2. Round trip through the consumer.
            back = load_arm_results(d, emotions, N_IMAGES)
            key = next(k_ for k_ in back if k_[0] == tag)
            assert same(back[key][0], matrix(out, emotions)), f"round trip failed: {dest}"
        used = [r["n_samples_used"] for r in agg["results"]]
        temp = info["sampling"].get("temperature")
        override = " (card override)" if info["sampling"].get("temperature_override") else ""
        print(f"{tag:32} K={k} take={take} images={len(used)} "
              f"parsed {agg['batch_info']['successful_parses']}/{len(used)} "
              f"mean samples used {np.mean(used):.2f} min {min(used)} "
              f"T={temp}{override}")
        image_sets.append(set(info["image_indices"]))

    if len({frozenset(s) for s in image_sets}) > 1:
        print("WARNING: the multi runs do not share one image list", file=sys.stderr)
    if args.check_ours:
        check_ours(args.check_ours, set.intersection(*image_sets))
    print(f"wrote {args.out_dir}/{{{args.aggregate}_k*,sample0}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
