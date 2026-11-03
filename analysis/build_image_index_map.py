#!/usr/bin/env python
"""Map EmoNet-Face-HQ filenames (as used in the paper's hq.csv) to dataset row indices.

The HQ parquet has a column named `path`, but it is typed as `Image`, so the original
filename was replaced by the decoded PNG and no join key survives. Local predictions in
`results/` are keyed by dataset row index while the paper's published baselines in
`hq.csv` are keyed by filename, and the two orderings do not agree -- so the two sources
cannot be compared until the mapping is recovered.

Both sources do carry the raw per-expert annotations in an identical
`'Family|Emotion': rating` format, keyed by rater name (`human-1` .. `human-8`). The
full set of (rater, 40-rating vector) pairs for one image is effectively unique, so it
serves as a content-based join key.

Writes {filename: row_index} to JSON.

Usage:
    python analysis/build_image_index_map.py --out data/hq_image_index_map.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import resource
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10**7)
resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, resource.RLIM_INFINITY))


def signature(per_rater: dict[str, dict[str, float]]) -> tuple:
    """Order-independent fingerprint of one image's expert annotations."""
    return tuple(sorted(
        (rater, tuple(sorted((e, round(float(v), 4)) for e, v in ratings.items())))
        for rater, ratings in per_rater.items()
    ))


def load_csv_signatures(path: Path) -> dict[tuple, str]:
    per_image: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if str(row.get("human", "")).strip().lower() not in ("true", "1", "yes"):
                continue
            try:
                ann = ast.literal_eval(row["annotations"])
            except (ValueError, SyntaxError):
                continue
            if isinstance(ann, dict):
                per_image[row["image"]][row["annotator"]] = ann
    out: dict[tuple, str] = {}
    for img, per_rater in per_image.items():
        out.setdefault(signature(per_rater), img)
    return out, len(per_image)


def load_dataset_signatures(dataset_path: str) -> dict[tuple, int]:
    from datasets import load_dataset

    # Select only `label` so the Image column is never decoded.
    ds = load_dataset(dataset_path)["train"].select_columns(["label"])
    out: dict[tuple, int] = {}
    dupes = 0
    for i, raw in enumerate(ds["label"]):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
        per_rater: dict[str, dict[str, float]] = {}
        for entry in parsed:
            for rater, ratings in entry.items():
                per_rater[rater] = ratings
        sig = signature(per_rater)
        if sig in out:
            dupes += 1
            continue
        out[sig] = i
    return out, dupes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hq-csv", type=Path, default=Path("/tmp/hq.csv"))
    ap.add_argument("--dataset", default="data/emonet-face-hq")
    ap.add_argument("--out", type=Path, default=Path("data/hq_image_index_map.json"))
    args = ap.parse_args()

    csv_sigs, n_csv = load_csv_signatures(args.hq_csv)
    print(f"hq.csv: {n_csv} images, {len(csv_sigs)} distinct annotation signatures")

    ds_sigs, dupes = load_dataset_signatures(args.dataset)
    print(f"dataset: {len(ds_sigs)} distinct signatures ({dupes} duplicate signatures skipped)")

    mapping = {csv_sigs[sig]: idx for sig, idx in ds_sigs.items() if sig in csv_sigs}
    print(f"matched: {len(mapping)}/{n_csv} images")

    if len(mapping) < n_csv:
        print(f"  unmatched: {n_csv - len(mapping)} (duplicate or altered annotations)")

    # Is the filename order the same as the row order? If so the whole problem was moot.
    ordered = sorted(mapping.items())
    identity = sum(1 for k, (name, idx) in enumerate(ordered) if k == idx)
    print(f"filename-sorted order == row order for {identity}/{len(ordered)} images")

    if not mapping:
        print("no matches -- annotation formats differ between sources", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(mapping, fh, indent=0, sort_keys=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
