#!/usr/bin/env python
"""E0 report -- does the paper's 25-pair enforcement explain the reported VLM failure?

Compares three arms per model on EmoNet-Face HQ:

  paper_retry  the paper's prompt + their rejection loop (any response whose value dict
               is not exactly 25 pairs, or contains a zero, is regenerated)
  paper_once   the same prompt, single shot
  ours         free-cardinality prompt

If the failure is a capability limit, all three arms should score similarly. If it is the
enforcement mechanism, paper_retry should collapse toward the Random Baseline while
paper_once and ours do not -- which is what their own published predictions imply, where
five baselines emit exactly 25.00 emotions with std 0.00 and score at chance, against 8.14
(std 4.00) for human experts.

Scores use single-rater weighted kappa on reliable categories, the same path as
analysis/reliability_vs_performance.py, so numbers are comparable to the human-human
anchor of 0.468.

Usage:
    python analysis/e0_report.py --results-dir results_e0 --hq-csv /tmp/hq.csv
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

ARMS = ["paper_retry", "paper_once", "ours"]


def macro_ap(pred: np.ndarray, gold: np.ndarray, reliable: np.ndarray) -> float:
    """Mean average precision over reliable emotions, against binarised gold.

    Reported alongside weighted kappa as a second, rank-based view. Note that neither
    metric rewards emitting more emotions: injecting random false positives into
    gemma-3-4b's predictions drives kappa 0.389 -> -0.016 and mAP 0.530 -> 0.391,
    monotonically, as emission rises 7.8 -> 39.7. The observed +0.75 correlation between
    emission count and kappa across models is therefore a property of the models -- the
    better ones name roughly as many emotions as the experts do -- and not a scoring
    artifact.
    """
    from sklearn.metrics import average_precision_score

    aps = []
    for j in range(pred.shape[1]):
        if not reliable[j]:
            continue
        g, p = gold[:, j], pred[:, j]
        ok = ~(np.isnan(g) | np.isnan(p))
        y = (g[ok] > 0).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aps.append(average_precision_score(y, p[ok]))
    return float(np.mean(aps)) if aps else float("nan")


def load_arm_results(results_dir: Path, emotions_short, n_images):
    """{(model_tag, arm): (pred_matrix, batch_info)} from the E0 output files."""
    out = {}
    for path in sorted(results_dir.glob("*_eval.json")):
        try:
            payload = json.load(open(path))
            info = payload["batch_info"]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        arm = info.get("arm")
        if arm not in ARMS:
            continue
        tag = info.get("model_id", info["model"]).split("/")[-1]
        mat = np.full((n_images, len(emotions_short)), np.nan, dtype=np.float32)
        for row in payload.get("results", []):
            i = row.get("image_index")
            if i is None or i >= n_images or not row.get("parse_success"):
                continue
            emitted = row["predicted_emotions"]
            for j, e in enumerate(emotions_short):
                v = emitted.get(e, 0)
                mat[i, j] = v if isinstance(v, (int, float)) else 0.0
        # The n=250 sweep writes `__paper_retry_n250_eval.json` alongside the full-2500
        # `__paper_retry_eval.json`, and both carry arm="paper_retry". Sorted order puts
        # the full run first, so a plain assignment lets the 250-image file clobber it.
        # Keep whichever covers more images.
        prev = out.get((tag, arm))
        if prev and prev[1].get("total_images", 0) >= info.get("total_images", 0):
            continue
        out[(tag, arm)] = (mat, info)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_e0"))
    ap.add_argument("--hq-csv", type=Path, default=Path("/tmp/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--out", type=Path, default=Path("results/e0_report.json"))
    args = ap.parse_args()

    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, _ = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]
    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j]) for j in range(len(emotions))])
    reliable = alpha >= RELIABLE_ALPHA

    # Rater-vs-rater, not mean-of-raters-vs-rater: the 4-rater mean is denoised, and
    # scoring against it inflates the anchor (0.740 rather than the correct 0.468).
    hh = np.nanmean([human_human_kappa(human[:, j, :])
                     for j in range(len(emotions)) if reliable[j]])

    runs = load_arm_results(args.results_dir, short, len(images))
    if not runs:
        print(f"no E0 results in {args.results_dir}", file=sys.stderr)
        return 1

    rows = []
    for (tag, arm), (mat, info) in runs.items():
        kap = [model_single_rater_kappa(quantile_bin(mat[:, j], gold[:, j]), human[:, j, :])
               for j in range(len(emotions)) if reliable[j]]
        rows.append({
            "model": tag, "arm": arm,
            "kappa": float(np.nanmean(kap)),
            "map": macro_ap(mat, gold, reliable),
            "emit": float(info.get("mean_nonzero_emotions", float("nan"))),
            "parsed": info.get("successful_parses", 0),
            "total": info.get("total_images", 0),
            "compliant": info.get("paper_compliant", 0),
            "attempts": float(info.get("mean_attempts", 1.0)),
        })

    by_model: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["arm"]] = r

    # Arms of the same model must cover the same images before their columns can be
    # subtracted. They did not on 2026-08-06: paper_retry ran at --limit 250 while
    # paper_once and ours ran the full 2500, so the "retry effect" below was a difference
    # between two different image subsets. Warn loudly rather than printing a clean number.
    mixed = {m: sorted({a: d[a]["total"] for a in d}.items())
             for m, d in by_model.items() if len({d[a]["total"] for a in d}) > 1}
    if mixed:
        print(f"WARNING: {len(mixed)} model(s) have arms measured on DIFFERENT image counts.",
              file=sys.stderr)
        for m, tot in sorted(mixed.items()):
            print(f"  {m}: " + ", ".join(f"{a}={n}" for a, n in tot), file=sys.stderr)
        print("  Cross-arm deltas below are not like-for-like; restrict to a shared subset "
              "before quoting them.\n", file=sys.stderr)

    print(f"human-human anchor (reliable categories): kappa_w = {hh:.3f}")
    print(f"human experts emit 8.14 emotions/image (std 4.00)\n")
    hdr = f"{'model':30}" + "".join(f"{a:>26}" for a in ARMS) + f"{'retry effect':>14}"
    print(hdr)
    print(f"{'':30}" + "".join(f"{'kappa    mAP  emit parse':>26}" for _ in ARMS))
    for model in sorted(by_model, key=lambda m: -by_model[m].get("ours", {}).get("kappa", -9)):
        cells = ""
        for arm in ARMS:
            r = by_model[model].get(arm)
            cells += (f"{r['kappa']:7.3f}{r['map']:7.3f}{r['emit']:6.1f}{r['parsed']:6d} " if r
                      else f"{'--':>26}")
        pr, po = by_model[model].get("paper_retry"), by_model[model].get("paper_once")
        delta = f"{pr['map'] - po['map']:+.3f}" if pr and po else "--"
        print(f"{model[:30]:30}{cells}{delta:>14}")

    # Headline: does enforcing 25 pairs cost accuracy, and by how much?
    # Drop NaN pairs rather than averaging through them. A model that parsed 0 rows in an arm
    # scores NaN mAP, and a single NaN turned the whole headline into "mean +nan" -- which is
    # at least loud, but it also silently discards the nine models that did produce a number.
    pairs = [(m, by_model[m]["paper_retry"]["map"] - by_model[m]["paper_once"]["map"])
             for m in by_model
             if "paper_retry" in by_model[m] and "paper_once" in by_model[m]]
    dropped = [m for m, d in pairs if d != d]
    deltas = [d for _, d in pairs if d == d]
    if deltas:
        print(f"\nretry enforcement effect on mAP over {len(deltas)} models: "
              f"mean {np.mean(deltas):+.3f}, worst {np.min(deltas):+.3f}")
    if dropped:
        print(f"  excluded (no parsed rows in one arm): {', '.join(sorted(dropped))}")
    emits = {a: [r["emit"] for r in rows if r["arm"] == a and r["emit"] == r["emit"]] for a in ARMS}
    for a in ARMS:
        if emits[a]:
            print(f"  {a:12} mean emitted emotions/image = {np.mean(emits[a]):.2f} "
                  f"(human 8.14)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"human_human_kappa": float(hh), "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
