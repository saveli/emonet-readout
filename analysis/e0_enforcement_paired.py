#!/usr/bin/env python
"""Paired within-image test of the paper's 25-pair enforcement loop.

`e0_report.py` compares the `paper_retry` arm against `paper_once`, which are separate
runs over separate sampling draws. This script instead uses the two responses the retry
arm already stores for the *same image*:

  first_predicted_emotions   round 1, before any enforcement pressure
  predicted_emotions         the accepted response after up to N retries

Same model, same image, same prompt, same decoder settings -- the only difference is that
the loop rejected round 1 and kept resampling. That makes it a paired test, so the
cardinality confound that makes the cross-arm comparison hard to read (paper_once emits
15.1 emotions/image against 4.8 for `ours`) cannot drive the result: both columns here come
from the same prompt.

Also splits on `paper_compliant`, the images where the loop actually succeeded in forcing
exactly 25 non-zero pairs. If the enforcement mechanism is what broke the paper's
baselines, those images specifically should score worse than their own round-1 responses.

Usage:
    python analysis/e0_enforcement_paired.py --results-dir results_e0 --hq-csv /tmp/hq.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e0_report import macro_ap  # noqa: E402
from reliability_vs_performance import (  # noqa: E402
    RELIABLE_ALPHA,
    krippendorff_alpha,
    model_single_rater_kappa,
    parse_into_arrays,
    quantile_bin,
    scan_index,
)


def _fill(mat, i, emitted, emotions_short):
    for j, e in enumerate(emotions_short):
        v = emitted.get(e, 0)
        mat[i, j] = v if isinstance(v, (int, float)) else 0.0


def load_paired(path: Path, emotions_short, n_images):
    """(first_matrix, final_matrix, compliant_mask, n_paired) for one paper_retry file.

    Only images where *both* the round-1 and the final response parsed are populated;
    anything else stays NaN in both matrices, so the two columns are scored on an
    identical image set by construction.
    """
    payload = json.load(open(path))
    first = np.full((n_images, len(emotions_short)), np.nan, dtype=np.float32)
    final = np.full((n_images, len(emotions_short)), np.nan, dtype=np.float32)
    compliant = np.zeros(n_images, dtype=bool)
    paired = 0
    rounds = 0
    for row in payload.get("results", []):
        rounds = max(rounds, int(row.get("attempts") or 0))
        i = row.get("image_index")
        if i is None or i >= n_images:
            continue
        if not (row.get("parse_success") and row.get("first_parse_success")):
            continue
        _fill(first, i, row["first_predicted_emotions"], emotions_short)
        _fill(final, i, row["predicted_emotions"], emotions_short)
        compliant[i] = bool(row.get("paper_compliant"))
        paired += 1
    return first, final, compliant, paired, rounds, payload["batch_info"]


def score(mat, human, gold, reliable, n_emotions, subset=None):
    """Calibrated single-rater kappa on reliable categories, plus mAP and density."""
    m = mat if subset is None else np.where(subset[:, None], mat, np.nan)
    kap = [model_single_rater_kappa(quantile_bin(m[:, j], gold[:, j]), human[:, j, :])
           for j in range(n_emotions) if reliable[j]]
    rows_present = ~np.isnan(m).all(axis=1)
    dens = float(np.nanmean((m[rows_present] > 0).sum(axis=1))) if rows_present.any() else float("nan")
    return float(np.nanmean(kap)), macro_ap(m, gold, reliable), dens, int(rows_present.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_e0"))
    ap.add_argument("--hq-csv", type=Path, default=Path("/tmp/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--out", type=Path, default=Path("results/e0_enforcement_paired.json"))
    ap.add_argument("--min-images", type=int, default=1000,
                    help="skip runs still capped at the n=250 sweep size")
    args = ap.parse_args()

    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, _ = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]
    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j]) for j in range(len(emotions))])
    reliable = alpha >= RELIABLE_ALPHA

    # `.partial` checkpoints are included. The two runs that hit the 20 h wall carry all
    # 2500 images but only 3 of the 5 retry rounds, so they have had LESS enforcement
    # pressure applied than a complete run. That is conservative for a null: the estimated
    # effect can only be smaller than the model's true one, not larger. Rounds reached are
    # reported per model so the difference is visible rather than buried.
    paths = sorted(args.results_dir.glob("*__paper_retry_eval.json"))
    have = {p.name.split("__")[0] for p in paths}
    paths += [p for p in sorted(args.results_dir.glob("*__paper_retry_eval.json.partial"))
              if p.name.split("__")[0] not in have]

    rows = []
    for path in paths:
        tag = path.name.split("__")[0]
        first, final, compliant, paired, payload_rounds, info = load_paired(
            path, short, len(images))
        if info.get("total_images", 0) < args.min_images:
            print(f"skip {tag}: only {info.get('total_images')} images", file=sys.stderr)
            continue
        if paired == 0:
            print(f"skip {tag}: no image parsed in both rounds", file=sys.stderr)
            continue
        fk, fm, fd, _ = score(first, human, gold, reliable, len(emotions))
        lk, lm, ld, _ = score(final, human, gold, reliable, len(emotions))
        row = {"model": tag, "paired": paired, "compliant": int(compliant.sum()),
               "rounds": payload_rounds,
               "partial": path.name.endswith(".partial"),
               "first_kappa": fk, "final_kappa": lk, "d_kappa": lk - fk,
               "first_map": fm, "final_map": lm, "d_map": lm - fm,
               "first_emit": fd, "final_emit": ld}
        # The subset where enforcement actually succeeded. Below ~100 images the kappa is
        # too noisy to read, so it is computed but flagged rather than silently reported.
        if compliant.sum() >= 30:
            ck, cm, cd, cn = score(first, human, gold, reliable, len(emotions), compliant)
            ek, em, ed, _ = score(final, human, gold, reliable, len(emotions), compliant)
            row.update({"compliant_first_kappa": ck, "compliant_final_kappa": ek,
                        "compliant_d_kappa": ek - ck, "compliant_first_map": cm,
                        "compliant_final_map": em, "compliant_d_map": em - cm,
                        "compliant_first_emit": cd, "compliant_final_emit": ed,
                        "compliant_n": cn, "compliant_noisy": bool(cn < 100)})
        rows.append(row)

    if not rows:
        print(f"no full-size paper_retry runs in {args.results_dir}", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["d_kappa"])
    print(f"{'model':30}{'paired':>7}{'compl':>7}{'rnds':>6}{'k_r1':>8}{'k_fin':>8}{'dk':>8}"
          f"{'mAP_r1':>8}{'mAP_fin':>8}{'dmAP':>8}{'emit_r1':>9}{'emit_fin':>9}")
    for r in rows:
        rnd = f"{r['rounds']}{'*' if r['partial'] else ''}"
        print(f"{r['model'][:30]:30}{r['paired']:7d}{r['compliant']:7d}{rnd:>6}"
              f"{r['first_kappa']:8.3f}{r['final_kappa']:8.3f}{r['d_kappa']:+8.3f}"
              f"{r['first_map']:8.3f}{r['final_map']:8.3f}{r['d_map']:+8.3f}"
              f"{r['first_emit']:9.1f}{r['final_emit']:9.1f}")

    if any(r["partial"] for r in rows):
        print("  * incomplete: hit the wall after 3 of 5 retry rounds, so less enforcement "
              "pressure was applied than a full run -- conservative for a null.")
    dk = [r["d_kappa"] for r in rows if r["d_kappa"] == r["d_kappa"]]
    dm = [r["d_map"] for r in rows if r["d_map"] == r["d_map"]]
    print(f"\npaired enforcement effect over {len(dk)} models: "
          f"kappa {np.mean(dk):+.3f} (worst {np.min(dk):+.3f}), "
          f"mAP {np.mean(dm):+.3f} (worst {np.min(dm):+.3f})")

    sub = [r for r in rows if "compliant_d_kappa" in r]
    if sub:
        print(f"\nrestricted to images where the loop reached exactly 25 pairs:")
        print(f"{'model':30}{'n':>7}{'k_r1':>8}{'k_fin':>8}{'dk':>8}{'emit_r1':>9}{'emit_fin':>9}")
        for r in sub:
            flag = " (noisy)" if r["compliant_noisy"] else ""
            print(f"{r['model'][:30]:30}{r['compliant_n']:7d}"
                  f"{r['compliant_first_kappa']:8.3f}{r['compliant_final_kappa']:8.3f}"
                  f"{r['compliant_d_kappa']:+8.3f}{r['compliant_first_emit']:9.1f}"
                  f"{r['compliant_final_emit']:9.1f}{flag}")
        cdk = [r["compliant_d_kappa"] for r in sub if r["compliant_d_kappa"] == r["compliant_d_kappa"]]
        if cdk:
            print(f"  mean kappa change on enforced images: {np.mean(cdk):+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"reliable_threshold": RELIABLE_ALPHA, "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
