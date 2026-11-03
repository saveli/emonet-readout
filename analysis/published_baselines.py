#!/usr/bin/env python
"""Score the benchmark's OWN published baselines through the same scoring path we use.

Everything else in this project evaluates eleven checkpoints the benchmark never ran, which
leaves an obvious gap: we would have shown "these models score well under
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
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv.field_size_limit(10**7)

from reliability_vs_performance import is_human  # noqa: E402


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
    ap.add_argument("--out", type=Path, default=Path("results/published_baselines.json"))
    ap.add_argument("-B", "--replicates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=20)
    ap.add_argument("--tie-rule", choices=("midpoint", "fractional"), default="fractional")
    ap.add_argument("--target", choices=("mean_halfup", "mean_halfeven", "rater_pooled"),
                    default="rater_pooled")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--gen-dir", type=Path, default=Path("results_e0"))
    ap.add_argument("--ver-dir", type=Path, default=Path("results_verify_prefill"))
    args = ap.parse_args()
    return compute(args)


DISPLAY = {"Empathic Insight Face Small": r"EIF Small$^{\dagger}$",
           "Empathic Insight Face Large": r"EIF Large$^{\dagger}$",
           "Gemini 2.5 Pro 03-25 ZS": "Gemini 2.5 Pro ZS",
           "Random Baseline": r"\emph{Random Baseline}"}
BOLD = {"Gemini 2.5 Flash MS", "Gemini 2.5 Flash ZS"}


def compute(args) -> int:
    """The fourteen published baselines, both estimators, each with a paired
    image-bootstrap test against the expert reference (mean pairwise kappa_w), B=500. Replaces
    published_baselines.json + published_anchor_significance.json for the paper.
    Output: results/published_baselines.json, tables/published.tex.
    """
    import scoring as s2
    from controls_common import ci, pmap

    D, _, _, head = s2.headline(args)
    s2.assert_headline(head)
    rel, n = D["rel"], D["n"]
    mats = {name: D["preds"][ai].astype(np.float64) for name, ai in D["annotators"].items()}
    names = sorted(mats)

    def score(Sm):
        o = {"anchor": Sm.anchor(rel, "mean")}
        for nm in names:
            for est in s2.ESTIMATORS:
                o[(nm, est)] = Sm.model(mats[nm], rel, est)
        return o

    point = score(s2.Sample(D, np.arange(n), S=args.S))
    dr = s2.draws(n, args.replicates, args.seed)
    print(f"paired image bootstrap B={args.replicates}, S={args.S}", flush=True)
    reps = pmap(lambda b: score(s2.Sample(D, dr[b], rep=b, S=args.S)),
                list(range(args.replicates)), args.workers)
    A = np.array([r["anchor"] for r in reps])
    card = cardinality(args.hq_csv)
    # Mapping gain = held-out kappa - kappa on the system's own 0-7 output with no mapping
    # (clip(round(x), 0, 7); continuous scores such as EIF's and Hume's lose most under it).
    full = s2.Sample(D, np.arange(n), S=args.S)

    def raw_kappa(m):
        return float(np.mean([s2.kappa_slots(np.where(np.isnan(m[:, j]), np.nan,
                                                       np.clip(np.round(m[:, j]), 0, 7)),
                                              full.H[:, j, :]) for j in rel]))
    raw = {nm: raw_kappa(mats[nm]) for nm in names}
    # Constant-prediction check: a constant column is one tie block, which the fractional rule
    # spreads over the target marginal exactly as a random tie-break would, so its expected
    # agreement is chance and its kappa must sit at zero (held-out) / exactly zero (in-sample).
    const = np.zeros((n, D["gold"].shape[1]))
    const_k = {est: full.model(const, rel, est) for est in s2.ESTIMATORS}
    print(f"constant-prediction check: held-out {const_k['heldout']:+.4f}, in-sample "
          f"{const_k['oracle']:+.2e}")
    if abs(const_k["heldout"]) > 0.002 or abs(const_k["oracle"]) > 1e-9:
        raise SystemExit("STOP: a constant prediction does not score at chance")
    rows = []
    for nm in names:
        c = card.get(nm, [])
        row = {"annotator": nm, "mean_nonzero": statistics.mean(c) if c else float("nan"),
               "sd_nonzero": statistics.pstdev(c) if len(c) > 1 else 0.0,
               "n_scored": int((~np.isnan(mats[nm][:, rel[0]])).sum()),
               "kappa_raw_no_mapping": raw[nm]}
        for est in s2.ESTIMATORS:
            R = np.array([r[(nm, est)] for r in reps])
            lo, hi = ci(R - A)
            row[est] = {"kappa_w": point[(nm, est)], "ci": ci(R),
                        "minus_anchor": point[(nm, est)] - point["anchor"],
                        "minus_anchor_ci": [lo, hi], "verdict": s2.verdict(lo, hi)}
        rows.append(row)
    for r in rows:
        r["mapping_gain"] = r["heldout"]["kappa_w"] - r["kappa_raw_no_mapping"]
    rows.sort(key=lambda r: -r["heldout"]["kappa_w"])
    big = sorted(rows, key=lambda r: -r["mapping_gain"])[:2]
    gains = {"mean_all": float(np.mean([r["mapping_gain"] for r in rows])),
             "n_all": len(rows),
             "largest_two": {r["annotator"]: r["mapping_gain"] for r in big},
             "mean_without_largest_two": float(np.mean([r["mapping_gain"] for r in rows
                                                        if r not in big])),
             "n_without": len(rows) - 2,
             "definition": "held-out kappa minus kappa of clip(round(x), 0, 7), no mapping",
             "constant_prediction_kappa": const_k}
    print(f"mapping gain: mean {gains['mean_all']:+.3f} over {len(rows)}; "
          + ", ".join(f"{k} {v:+.3f}" for k, v in gains["largest_two"].items())
          + f"; mean over the other {gains['n_without']} {gains['mean_without_largest_two']:+.3f}")
    hc = card.get("HUMAN", [])
    print(f"\n{'annotator':32}{'held-out':>10}{'oracle':>9}{'n/img':>7}   vs reference (held-out)")
    for r in rows:
        h = r["heldout"]
        print(f"{r['annotator']:32}{h['kappa_w']:10.3f}{r['oracle']['kappa_w']:9.3f}"
              f"{r['mean_nonzero']:7.1f}   {h['minus_anchor']:+.3f} [{h['minus_anchor_ci'][0]:+.3f},"
              f"{h['minus_anchor_ci'][1]:+.3f}] {h['verdict']}")
    print(f"{'experts (reference)':32}{point['anchor']:10.3f}{'':9}{statistics.mean(hc):7.1f}")

    # body table: two halves side by side, sorted by held-out kappa, experts in sorted position
    entries = [(DISPLAY.get(r["annotator"], r["annotator"]), r["mean_nonzero"],
                r["heldout"]["kappa_w"], r["annotator"] in BOLD) for r in rows]
    entries.append((r"\emph{experts (reference)}", statistics.mean(hc), point["anchor"], False))
    entries.sort(key=lambda e: -e[2])
    half = (len(entries) + 1) // 2
    left, right = entries[:half], entries[half:]

    def cell(e):
        if e is None:
            return "& &"
        nm, k, kap, bold = e
        nm = f"\\textbf{{{nm}}}" if bold else nm
        kv = f"$\\mathbf{{{kap:.3f}}}$" if bold else f"${kap:.3f}$"
        return f"{nm} & ${k:.1f}$ & {kv}"
    lines = ["% generated by analysis/published_baselines.py -- do not edit by hand",
             "% Five categories, held-out mapping; sorted by kappa_w with the expert reference "
             "(mean pairwise) in its sorted position.",
             r"\begin{tabular}{@{}lcc@{\hspace{8pt}}lcc@{}}", r"\toprule",
             r"System & n/img & $\kappa_w$ & System & n/img & $\kappa_w$ \\", r"\midrule"]
    for i in range(half):
        lines.append(f"{cell(left[i])} & {cell(right[i] if i < len(right) else None)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    s2.write_json("published_baselines.json", {
        "anchor": point["anchor"], "anchor_ci": ci(A), "rows": rows, "mapping_gain": gains,
        "human_nonzero_mean": statistics.mean(hc), "human_nonzero_sd": statistics.pstdev(hc),
        "provenance": s2.provenance("published_baselines.py", args,
                                    sources=[args.hq_csv])})
    s2.write_tex(s2.TABLES_DIR / "published.tex", "\n".join(lines))
    s2.write_tex(Path("tables/published.tex"), "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
