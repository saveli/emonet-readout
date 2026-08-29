#!/usr/bin/env python
"""Does model performance track per-emotion inter-rater reliability?

EmoNet-Face HQ carries four expert ratings per image across all 40 emotions. Agreement
varies enormously by category -- Elation reaches Krippendorff alpha 0.58 while Interest,
Concentration and Contemplation sit at or below zero. If per-emotion model performance
tracks per-emotion reliability, then a benchmark score averaged over all 40 categories
is diluted by categories that are not reliably measurable in the first place.

This script computes, for every model, a per-emotion Spearman correlation against the
expert mean rating, then correlates that 40-vector against the per-emotion alpha vector.

Two prediction sources are combined:
  * `hq.csv` from the EmoNet-Face repo -- the paper's own baselines (EmpathicInsight-Face,
    Claude 3.7, GPT-4o, Gemini 2.0/2.5 Flash, Nova Pro, Pixtral Large, Qwen Plus).
  * local `results/` JSONs -- the open models evaluated in this project.

Spearman is used throughout because the two sources are on incomparable scales: the
paper's baselines emit mean-subtracted continuous scores, ours emit 0-7 integers.

Memory note: hq.csv is ~91 MB and holds 44,763 rows whose `annotations` field is a dict
literal of 40 entries. Materialising those as Python dicts costs several GB, because
every row mints 40 fresh key strings. An earlier version of this script did exactly that
and then forked the heap across a process pool, which triggered a system-wide OOM. The
parse below streams each row straight into preallocated arrays and discards the dict, so
peak usage stays in the low tens of MB. The arithmetic is trivially vectorised and needs
no parallelism.

Usage:
    python analysis/reliability_vs_performance.py --hq-csv /tmp/hq.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import pickle
import resource
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

csv.field_size_limit(10**7)

# Hard ceiling on address space, so a future regression kills this process rather than
# the desktop session.
MEMORY_LIMIT_GB = 4
resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_GB * 1024**3, resource.RLIM_INFINITY))

# hq.csv carries 22 annotators: 8 human experts (human-1..human-8, 10,000 rows) and 14
# model baselines (34,763 rows). The `human` column is authoritative -- do NOT identify
# humans by name, since an incomplete model list silently pollutes the rater pool and
# collapses the agreement estimate.
def is_human(row) -> bool:
    return str(row.get("human", "")).strip().lower() in ("true", "1", "yes")

MIN_IMAGES = 100      # emotions need this many rated images to enter a correlation
MAX_RATERS = 8        # per-image rater slots to preallocate
RELIABLE_ALPHA = 0.3  # threshold separating measurable from unmeasurable categories


def scan_index(path: Path, index_map: dict[str, int] | None = None):
    """First pass: emotion keys, image order, annotator names. One literal_eval total.

    `index_map` maps hq.csv filenames to dataset row indices (see
    analysis/build_image_index_map.py). It is required for the local predictions, which
    are keyed by row index: filename order and row order are an arbitrary permutation of
    each other, so without the map every local correlation is scrambled.
    """
    emotions: list[str] | None = None
    images: dict[str, int] = {}
    annotators: dict[str, int] = {}

    with open(path) as fh:
        for row in csv.DictReader(fh):
            if emotions is None:
                try:
                    parsed = ast.literal_eval(row["annotations"])
                except (ValueError, SyntaxError):
                    continue
                if isinstance(parsed, dict):
                    emotions = sorted(parsed)
            if is_human(row):
                images.setdefault(row["image"], 0)
            else:
                annotators.setdefault(row["annotator"], len(annotators))
    if emotions is None:
        raise ValueError("no parsable annotations column found")
    if index_map:
        missing = [im for im in images if im not in index_map]
        if missing:
            raise ValueError(f"{len(missing)} images absent from the index map")
        images = {img: index_map[img] for img in images}
    else:
        print("WARNING: no index map -- local model rows will be misaligned", file=sys.stderr)
        images = {img: i for i, img in enumerate(sorted(images))}
    return emotions, images, annotators


def parse_into_arrays(path: Path, emotions, images, annotators):
    """Second pass: stream rows into preallocated arrays, discarding each parsed dict."""
    n_img, n_emo = len(images), len(emotions)
    col = {e: j for j, e in enumerate(emotions)}

    human = np.full((n_img, n_emo, MAX_RATERS), np.nan, dtype=np.float32)
    n_raters = np.zeros((n_img, n_emo), dtype=np.int8)
    preds = np.full((len(annotators), n_img, n_emo), np.nan, dtype=np.float32)

    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                ann = ast.literal_eval(row["annotations"])
            except (ValueError, SyntaxError):
                continue
            if not isinstance(ann, dict):
                continue
            i = images.get(row["image"])
            if i is None:
                continue

            if not is_human(row):
                m = annotators[row["annotator"]]
                for k, v in ann.items():
                    j = col.get(k)
                    if j is not None and isinstance(v, (int, float)):
                        preds[m, i, j] = v   # duplicate rows: last write wins
            else:
                for k, v in ann.items():
                    j = col.get(k)
                    if j is None or not isinstance(v, (int, float)):
                        continue
                    slot = n_raters[i, j]
                    if slot < MAX_RATERS:
                        human[i, j, slot] = v
                        n_raters[i, j] = slot + 1
            del ann  # keep peak memory flat
    return human, n_raters, preds


def krippendorff_alpha(values: np.ndarray, counts: np.ndarray) -> float:
    """Interval-scale Krippendorff's alpha for one emotion.

    values: (n_images, MAX_RATERS) with NaN padding; counts: (n_images,) raters per image.
    Units with fewer than two ratings carry no disagreement information and are dropped.
    """
    keep = counts > 1
    if keep.sum() < 2:
        return float("nan")
    v = values[keep]
    c = counts[keep].astype(np.float64)

    # Within-unit squared difference, summed over ordered pairs, normalised by (m_u - 1).
    s = np.nansum(v, axis=1)
    sq = np.nansum(v * v, axis=1)
    within = (c * sq - s * s) / (c - 1)          # == sum_{i<j}(x_i-x_j)^2 / (m_u-1) * 1
    observed = within.sum() / c.sum()

    flat = v[~np.isnan(v)]
    m = flat.size
    if m < 2:
        return float("nan")
    expected = (m * (flat * flat).sum() - flat.sum() ** 2) / (m * (m - 1))
    return float(1 - observed / expected) if expected > 0 else float("nan")


N_LEVELS = 8  # ordinal rating scale 0..7


def quantile_bin(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Map continuous predictions onto the 0..7 scale by matching the gold marginal.

    Weighted kappa is defined on ordinal categories, but the paper's baselines emit
    continuous mean-subtracted scores (Empathic Insight, Hume) while ours emit 0-7
    integers. Rank-matching each model's scores to the gold marginal for that emotion
    puts both on the same scale without rewarding or punishing a model for its output
    range. This is a calibration choice and has to be reported as one.
    """
    out = np.full(pred.shape, np.nan)
    ok = ~(np.isnan(pred) | np.isnan(gold))
    if ok.sum() < 2:
        return out
    p, g = pred[ok], gold[ok]
    # Target counts per level, taken from the gold ratings for this emotion.
    levels, counts = np.unique(np.clip(np.round(g), 0, N_LEVELS - 1), return_counts=True)
    order = np.argsort(p, kind="stable")
    assigned = np.empty(len(p))
    start = 0
    for lvl, cnt in zip(levels, counts):
        assigned[order[start:start + cnt]] = lvl
        start += cnt
    if start < len(p):                     # rounding slack -> top level
        assigned[order[start:]] = levels[-1]
    out[ok] = assigned
    return out


def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa with quadratic weights over N_LEVELS ordinal categories."""
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 2:
        return float("nan")
    x = np.clip(a[ok], 0, N_LEVELS - 1).astype(int)
    y = np.clip(np.round(b[ok]), 0, N_LEVELS - 1).astype(int)
    obs = np.zeros((N_LEVELS, N_LEVELS))
    np.add.at(obs, (x, y), 1)
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    idx = np.arange(N_LEVELS)
    w = (idx[:, None] - idx[None, :]) ** 2 / (N_LEVELS - 1) ** 2
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom > 0 else float("nan")


def _kappa_from_pairs(x: np.ndarray, y: np.ndarray) -> float:
    """Quadratic weighted kappa from already-paired ordinal observations."""
    if x.size < 2:
        return float("nan")
    xi = np.clip(np.round(x), 0, N_LEVELS - 1).astype(int)
    yi = np.clip(np.round(y), 0, N_LEVELS - 1).astype(int)
    obs = np.zeros((N_LEVELS, N_LEVELS))
    np.add.at(obs, (xi, yi), 1)
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    idx = np.arange(N_LEVELS)
    w = (idx[:, None] - idx[None, :]) ** 2 / (N_LEVELS - 1) ** 2
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom > 0 else float("nan")


def human_human_kappa(values: np.ndarray) -> float:
    """Mean pairwise weighted kappa between expert raters for one emotion.

    This is the paper's own reference quantity (they report 0.20 pooled). Computing it
    through the same code path as the model scores is what makes the model numbers
    interpretable -- an absolute kappa means nothing without the matching anchor.
    """
    xs, ys = [], []
    for a in range(values.shape[1]):
        for b in range(a + 1, values.shape[1]):
            m = ~(np.isnan(values[:, a]) | np.isnan(values[:, b]))
            if m.any():
                xs.append(values[m, a])
                ys.append(values[m, b])
    if not xs:
        return float("nan")
    return _kappa_from_pairs(np.concatenate(xs), np.concatenate(ys))


def model_single_rater_kappa(pred_binned: np.ndarray, values: np.ndarray) -> float:
    """Weighted kappa between a model and individual raters, pooled over raters.

    Matches the human-human protocol: one noisy judgement against another, rather than
    against the denoised 4-rater mean. Scores are markedly lower than mean-target kappa
    and are the only ones comparable to the published 0.20.
    """
    xs, ys = [], []
    for a in range(values.shape[1]):
        m = ~(np.isnan(values[:, a]) | np.isnan(pred_binned))
        if m.any():
            xs.append(pred_binned[m])
            ys.append(values[m, a])
    if not xs:
        return float("nan")
    return _kappa_from_pairs(np.concatenate(xs), np.concatenate(ys))


def per_emotion_kappa(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    out = np.full(pred.shape[1], np.nan)
    for j in range(pred.shape[1]):
        binned = quantile_bin(pred[:, j], gold[:, j])
        out[j] = quadratic_weighted_kappa(binned, gold[:, j])
    return out


def load_local_models(dirs, emotions_short, n_images):
    """Cached open-model predictions, keyed by image_index. Unparsed rows stay NaN."""
    out, seen = {}, set()
    for d in dirs:
        if not d.exists():
            continue
        for path in sorted(d.glob("*.json")):
            if path.name in seen:
                continue
            seen.add(path.name)
            try:
                with open(path) as fh:
                    payload = json.load(fh)
                name = payload["batch_info"]["model"]
            except (json.JSONDecodeError, KeyError, OSError, TypeError):
                continue
            mat = np.full((n_images, len(emotions_short)), np.nan, dtype=np.float32)
            for row in payload.get("results", []):
                idx = row.get("image_index")
                if idx is None or idx >= n_images:
                    continue
                if not row.get("parse_success") or not row.get("predicted_emotions"):
                    continue
                emitted = row["predicted_emotions"]
                for j, e in enumerate(emotions_short):
                    v = emitted.get(e, 0)
                    mat[idx, j] = v if isinstance(v, (int, float)) else 0.0
            out[name] = mat
            del payload
    return out


def per_emotion_spearman(pred: np.ndarray, gold: np.ndarray) -> np.ndarray:
    out = np.full(pred.shape[1], np.nan)
    for j in range(pred.shape[1]):
        p, g = pred[:, j], gold[:, j]
        ok = ~(np.isnan(p) | np.isnan(g))
        if ok.sum() < MIN_IMAGES or np.std(p[ok]) == 0 or np.std(g[ok]) == 0:
            continue
        out[j] = spearmanr(p[ok], g[ok]).statistic
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hq-csv", type=Path, default=Path("/tmp/hq.csv"))
    ap.add_argument("--results-dir", type=Path, nargs="+",
                    default=[Path("results"), Path("results_dpo")])
    ap.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth_emotions.pkl"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"),
                    help="filename -> dataset row index, from build_image_index_map.py")
    ap.add_argument("--out", type=Path, default=Path("analysis/reliability_vs_performance"))
    args = ap.parse_args()

    if not args.hq_csv.exists():
        print(f"missing {args.hq_csv} -- fetch data/hq.csv from the emonet-face repo", file=sys.stderr)
        return 1

    index_map = None
    if args.index_map.exists():
        with open(args.index_map) as fh:
            index_map = json.load(fh)
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    print(f"hq.csv: {len(images)} images, {len(emotions)} emotions, {len(annotators)} model annotators")

    human, n_raters, preds = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]

    gold = np.where(n_raters > 0, np.nanmean(human, axis=2), np.nan)

    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j])
                      for j in range(len(emotions))])
    print(f"alpha: mean={np.nanmean(alpha):.3f} min={np.nanmin(alpha):.3f} max={np.nanmax(alpha):.3f}")

    matrices = {f"[paper] {n}": preds[m] for n, m in annotators.items()}
    local = load_local_models(args.results_dir, short, len(images))
    matrices.update({f"[ours] {n}": m for n, m in local.items()})
    print(f"models: {len(matrices)} ({len(annotators)} from hq.csv, {len(local)} local)")

    # Local JSONs key on dataset row index, hq.csv on filename. If the orderings disagree
    # every local correlation is meaningless, so verify against the cached pkl gold.
    if args.ground_truth.exists() and local:
        with open(args.ground_truth, "rb") as fh:
            rows = pickle.load(fh)
        pkl_emos = list(rows[0]["gold_scores"].keys())
        pkl_gold = np.array([[r["gold_scores"][e] for e in pkl_emos] for r in rows], dtype=np.float32)
        common = [e for e in short if e in pkl_emos]
        a = gold[:, [short.index(e) for e in common]].ravel()
        b = pkl_gold[:len(images)][:, [pkl_emos.index(e) for e in common]].ravel()
        ok = ~(np.isnan(a) | np.isnan(b))
        # The cached pkl stores round(rater mean); hq.csv keeps the raw mean. Round before
        # comparing, otherwise the quantisation alone caps the correlation near 0.88.
        exact = float((np.round(a[ok]) == b[ok]).mean())
        r = float(spearmanr(np.round(a[ok]), b[ok]).statistic)
        flag = "OK" if exact > 0.99 else "*** MISALIGNED -- local results not comparable ***"
        print(f"alignment check: {exact:.2%} exact cell match, spearman={r:.3f}  {flag}")

    reliable = alpha >= RELIABLE_ALPHA
    print(f"\nreliable categories (alpha>={RELIABLE_ALPHA}): {int(reliable.sum())}/{len(emotions)}")

    hh = np.array([human_human_kappa(human[:, j, :]) for j in range(len(emotions))])
    print(f"HUMAN-HUMAN kappa_w: all-40={np.nanmean(hh):.3f}  "
          f"alpha>=.3={np.nanmean(hh[reliable & ~np.isnan(hh)]):.3f}   "
          f"(paper reports 0.20 pooled)")

    table = []
    for name, mat in matrices.items():
        perf = per_emotion_spearman(mat, gold)
        kap = per_emotion_kappa(mat, gold)
        kap1 = np.array([model_single_rater_kappa(quantile_bin(mat[:, j], gold[:, j]),
                                                  human[:, j, :])
                         for j in range(len(emotions))])
        ok = ~(np.isnan(perf) | np.isnan(alpha))
        if ok.sum() < 10:
            continue
        kok = ~np.isnan(kap)
        table.append({
            "model": name,
            "rho_alpha_vs_perf": float(spearmanr(alpha[ok], perf[ok]).statistic),
            "perf_all": float(np.nanmean(perf)),
            "perf_reliable": float(np.nanmean(perf[reliable & ok])),
            "perf_unreliable": float(np.nanmean(perf[~reliable & ok])),
            "kappa_all": float(np.nanmean(kap)),
            "kappa_reliable": float(np.nanmean(kap[reliable & kok])),
            "kappa_single_rater_all": float(np.nanmean(kap1)),
            "kappa_single_rater_reliable": float(np.nanmean(kap1[reliable & ~np.isnan(kap1)])),
            "per_emotion": {short[j]: (None if np.isnan(perf[j]) else float(perf[j]))
                            for j in range(len(emotions))},
        })
    table.sort(key=lambda r: -r["perf_reliable"])

    print(f"\n{'model':40}{'rhoRel':>8}{'kwRel':>8}{'kw1r40':>9}{'kw1rRel':>9}")
    for r in table:
        print(f"{r['model'][:40]:40}{r['perf_reliable']:8.3f}{r['kappa_reliable']:8.3f}"
              f"{r['kappa_single_rater_all']:9.3f}{r['kappa_single_rater_reliable']:9.3f}")
    print(f"{'HUMAN-HUMAN (anchor)':40}{'':8}{'':8}"
          f"{np.nanmean(hh):9.3f}{np.nanmean(hh[reliable & ~np.isnan(hh)]):9.3f}")
    print("  rhoRel  = mean per-emotion Spearman vs 4-rater mean, alpha>=0.3")
    print("  kwRel   = weighted kappa vs the DENOISED 4-rater mean -- inflated, not")
    print("            comparable to the published 0.20")
    print("  kw1r*   = weighted kappa vs INDIVIDUAL raters, same protocol as human-human")

    rhos = [r["rho_alpha_vs_perf"] for r in table]
    print(f"\nrho across {len(rhos)} models: mean={np.mean(rhos):.3f} "
          f"min={np.min(rhos):.3f} max={np.max(rhos):.3f}")

    # Does dropping unmeasurable categories reorder the leaderboard?
    by_all = [r["model"] for r in sorted(table, key=lambda r: -r["perf_all"])]
    by_rel = [r["model"] for r in table]
    moved = sum(1 for i, m in enumerate(by_rel) if by_all.index(m) != i)
    print(f"rank changes when restricting to alpha>={RELIABLE_ALPHA}: {moved}/{len(table)} models move")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".json"), "w") as fh:
        json.dump({"alpha": {short[j]: (None if np.isnan(alpha[j]) else float(alpha[j]))
                             for j in range(len(emotions))},
                   "reliable_threshold": RELIABLE_ALPHA,
                   "models": table}, fh, indent=2)
    print(f"wrote {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
