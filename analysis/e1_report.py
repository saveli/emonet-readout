#!/usr/bin/env python
"""E1 report -- does correctly-formatted training help, and is preference signal doing it?

Scores the five E1 arms produced by `evaluation/verify_eval.py`, which queries all 40
emotions per image with the training prompt and records P(yes) as a continuous value.

  e1_0_base   untrained base, same protocol -- the only honest reference point
  e1_1        verify_sft
  e1_2        verify_dpo
  e1_3        compare_dpo          <- the honest preference test
  e1_4        compare_dpo_random   <- C1 control

The read-out the spec mandates is **E1.3 - E1.4**, not E1.3 - E1.0: if the random-pairing
control matches the real one, the gain was format exposure and not preference signal.
E1.3 must also clear the C4 calibration-only margin to be worth reporting.

P(yes) lives on 0..1 while the ground truth is 0..7, so `raw_bin` (clip-and-round the
model's own scale) collapses every score to 0 or 1 and is not a meaningful number here.
The calibrated column is the primary one; raw is printed only so the scale mismatch is
visible rather than hidden. Same oracle caveat as C4: quantile calibration uses the gold
marginal, so it is an upper bound, not a deployable procedure.

Usage:
    python analysis/e1_report.py --results-dir results_e1 --hq-csv /tmp/hq.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c4_calibration_control import build_baselines, raw_bin  # noqa: E402
from e0_report import macro_ap  # noqa: E402
from reliability_vs_performance import (  # noqa: E402
    RELIABLE_ALPHA,
    human_human_kappa,
    krippendorff_alpha,
    model_single_rater_kappa,
    parse_into_arrays,
    quantile_bin,
    scan_index,
)

# Display order, and the labels the spec uses.
ARM_ORDER = ["e1_0_base", "e1_1", "e1_2", "e1_3", "e1_4"]
ARM_DESC = {
    "e1_0_base": "untrained base",
    "e1_1": "verify_sft",
    "e1_2": "verify_dpo",
    "e1_3": "compare_dpo",
    "e1_4": "compare_dpo_random (C1)",
}

# P(yes) above this counts as an emitted emotion, matching verify_eval.py's own density
# accounting so metric 3 is the same number in both places.
DENSITY_THRESHOLD = 0.5


def load_verify_results(results_dir: Path, emotions_short, n_images):
    """{tag: (pred_matrix, batch_info)} from verify_eval.py output files.

    Keyed on batch_info["model"] (the --tag), not model_id: every E1 arm is the same base
    checkpoint with a different adapter, so model_id collides across all five.
    """
    out = {}
    for path in sorted(results_dir.glob("*__verify_eval.json")):
        try:
            payload = json.load(open(path))
            info = payload["batch_info"]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        if info.get("arm") != "verify":
            continue
        tag = info["model"].split(" ")[0]
        mat = np.full((n_images, len(emotions_short)), np.nan, dtype=np.float32)
        for row in payload.get("results", []):
            i = row.get("image_index")
            if i is None or i >= n_images or not row.get("parse_success"):
                continue
            emitted = row["predicted_emotions"]
            for j, e in enumerate(emotions_short):
                v = emitted.get(e, 0)
                mat[i, j] = v if isinstance(v, (int, float)) else 0.0
        out[tag] = (mat, info)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_e1"))
    ap.add_argument("--hq-csv", type=Path, default=Path("/tmp/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--out", type=Path, default=Path("results/e1_report.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, _ = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]
    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j]) for j in range(len(emotions))])
    reliable = alpha >= RELIABLE_ALPHA
    hh = np.nanmean([human_human_kappa(human[:, j, :])
                     for j in range(len(emotions)) if reliable[j]])

    runs = load_verify_results(args.results_dir, short, len(images))
    if not runs:
        print(f"no E1 verify results in {args.results_dir}", file=sys.stderr)
        return 1

    matrices = {tag: mat for tag, (mat, _) in runs.items()}
    matrices.update(build_baselines(gold, args.seed))

    rows = []
    for tag, mat in matrices.items():
        info = runs.get(tag, (None, {}))[1]
        raw_k, cal_k = [], []
        for j in range(len(emotions)):
            if not reliable[j]:
                continue
            raw_k.append(model_single_rater_kappa(raw_bin(mat[:, j]), human[:, j, :]))
            cal_k.append(model_single_rater_kappa(quantile_bin(mat[:, j], gold[:, j]), human[:, j, :]))
        # Density recomputed here rather than trusted from batch_info: baselines have no
        # batch_info, and a scorer that reports the run's own self-declared number cannot
        # catch a density collapse introduced between generation and scoring.
        dens = float(np.nanmean((mat > DENSITY_THRESHOLD).sum(axis=1)))
        rows.append({
            "arm": tag,
            "desc": ARM_DESC.get(tag, ""),
            "kappa_raw": float(np.nanmean(raw_k)) if raw_k else float("nan"),
            "kappa_calibrated": float(np.nanmean(cal_k)) if cal_k else float("nan"),
            "map": macro_ap(mat, gold, reliable),
            "emit": dens,
            "emit_reported": float(info.get("mean_nonzero_emotions", float("nan"))),
            "parsed": info.get("successful_parses", 0),
            "total": info.get("total_images", 0),
            "mean_yesno_mass": float(info.get("mean_yesno_mass", float("nan"))),
        })

    order = {a: i for i, a in enumerate(ARM_ORDER)}
    rows.sort(key=lambda r: (order.get(r["arm"], 99), r["arm"]))

    by_arm = {r["arm"]: r for r in rows}
    n_img = {r["total"] for r in rows if r["arm"] in ARM_ORDER}
    if len(n_img) > 1:
        print(f"WARNING: arms cover different image counts {sorted(n_img)}; "
              "deltas below are not like-for-like.\n", file=sys.stderr)

    print(f"human-human anchor (reliable categories): kappa_w = {hh:.3f}")
    print(f"human experts emit 8.14 emotions/image; EmoNet ground truth 12.1\n")
    print(f"{'arm':12}{'what':26}{'raw':>8}{'cal':>8}{'mAP':>8}{'emit':>7}{'parsed':>8}{'yes/no':>8}")
    for r in rows:
        if r["arm"].startswith("baseline:"):
            continue
        print(f"{r['arm']:12}{r['desc'][:26]:26}{r['kappa_raw']:8.3f}{r['kappa_calibrated']:8.3f}"
              f"{r['map']:8.3f}{r['emit']:7.1f}{r['parsed']:8d}{r['mean_yesno_mass']:8.3f}")
    print()
    for r in rows:
        if r["arm"].startswith("baseline:"):
            print(f"{r['arm']:38}{r['kappa_raw']:8.3f}{r['kappa_calibrated']:8.3f}{r['map']:8.3f}")

    # The gate. E1.3 - E1.4 is the number the whole training track hangs on.
    gate = {}
    if "e1_3" in by_arm and "e1_4" in by_arm:
        for metric in ("kappa_calibrated", "map"):
            gate[f"e1_3_minus_e1_4_{metric}"] = by_arm["e1_3"][metric] - by_arm["e1_4"][metric]
        if "e1_0_base" in by_arm:
            for metric in ("kappa_calibrated", "map"):
                gate[f"e1_3_minus_base_{metric}"] = by_arm["e1_3"][metric] - by_arm["e1_0_base"][metric]
        print("\nGATE (spec: E1.3 must beat both E1.0 and E1.4 by more than the C4 margin)")
        for k, v in gate.items():
            print(f"  {k:38}{v:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"human_human_kappa": float(hh), "reliable_threshold": RELIABLE_ALPHA,
                   "density_threshold": DENSITY_THRESHOLD, "gate": gate, "rows": rows}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
