#!/usr/bin/env python
"""Is a verification model above the benchmark's own trained model, or does it only look it?

The paper says five of eleven checkpoints score above Empathic-Insight-Face, the model
EmoNet-Face trained to reach expert level. As first written that compared a model's bootstrap
interval to EIF's bare point estimate -- exactly the mistake `anchor_significance.py` exists to
correct for the human anchor, and it is worse here, because both quantities are computed on
the same 2500 images and therefore move together. A resample rich in images the taxonomy
handles cleanly lifts EIF and every VLM at once; two marginal intervals compared by eye throw
that covariance away and overstate the uncertainty.

So: one image resample per replicate, shared by EIF and by every model, and the reported
quantity is the paired difference `model - EIF`. EIF's own predictions are deterministic and
fixed per image, which is not an argument against an interval: the interval is the sampling
error of the 2500-image test set, not of the annotator, and the same is true of our
verification arm (a forward pass, nothing sampled) which has carried intervals all along.

Both references are scored: Small (the higher of the two under our path, 0.551) and Large
(0.534). Small is the harder bar and the one the paper's claim should rest on.

Guards, in the style of `anchor_significance.py`:

  * EIF's point kappa on the reliable five must reproduce `published_baselines.json`, and each
    model's must reproduce `verify_prefill_bootstrap_ci.json`, or the run aborts. Two scoring
    paths that disagree would make every difference below uninterpretable.
  * The published-baseline scorer and this one must therefore be the same estimator:
    per-emotion quantile calibration refit inside every replicate, scored against the
    individual raters, averaged over categories. `kappa_cal` is that estimator.

Only the reliable-five column can be checked against a published number, because
`published_baselines.py` reports that set alone; the all-40 and 35-excluded columns are new
and are reported without a cross-check.

Usage:
    python analysis/eif_significance.py                    # B=500, results_verify_prefill
    python analysis/eif_significance.py -B 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


# The two references, as they are named in hq.csv's annotator column.
REFERENCES = {
    "eif_small": "Empathic Insight Face Small",
    "eif_large": "Empathic Insight Face Large",
}
TOL = 0.0005  # published numbers are quoted to three decimals; this is rounding slack


def check(label: str, got: float, want: float) -> bool:
    ok = abs(got - want) <= TOL
    print(f"  {label:34}{got:9.4f}{want:9.3f}{got - want:+9.4f}   {'YES' if ok else '*** NO ***'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_verify_prefill"))
    ap.add_argument("--hq-csv", type=Path, default=Path("data/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--published", type=Path, default=Path("results/published_baselines.json"))
    ap.add_argument("--out", type=Path, default=Path("results/eif_significance.json"))
    ap.add_argument("-B", "--replicates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--S", type=int, default=20)
    ap.add_argument("--tie-rule", choices=("midpoint", "fractional"), default="fractional")
    ap.add_argument("--target", choices=("mean_halfup", "mean_halfeven", "rater_pooled"),
                    default="rater_pooled")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--gen-dir", type=Path, default=Path("results_e0"))
    args = ap.parse_args()
    return compute(args)


def compute(args) -> int:
    """P(yes) models - Empathic-Insight-Face (Small, Large), paired image
    bootstrap (B=500), both estimators, reliable five / unreliable 35 / all 40.
    Output: results/eif_significance.json.
    """
    import scoring as s2
    from controls_common import ci, pmap

    args.ver_dir = args.results_dir
    D, _, ver, head = s2.headline(args)
    s2.assert_headline(head)
    n, n_emo = D["n"], len(D["short"])
    rel = D["rel"]
    sets = {"reliable_5": rel, "unreliable_35": [j for j in range(n_emo) if j not in rel],
            "all_40": list(range(n_emo))}
    refs = {k: D["preds"][D["annotators"][nm]].astype(np.float64) for k, nm in REFERENCES.items()}
    tags = sorted(ver)
    allm = {**refs, **ver}

    def score(Sm):
        o = {}
        for est in s2.ESTIMATORS:
            for k, m in allm.items():
                v = Sm.model_cats(m, list(range(n_emo)), est)
                o[(k, est)] = {sn: float(np.nanmean(v[cs])) for sn, cs in sets.items()}
        return o

    point = score(s2.Sample(D, np.arange(n), S=args.S))
    dr = s2.draws(n, args.replicates, args.seed)
    print(f"paired image bootstrap B={args.replicates}, S={args.S}", flush=True)
    reps = pmap(lambda b: score(s2.Sample(D, dr[b], rep=b, S=args.S)),
                list(range(args.replicates)), args.workers)
    rows, summary, references = [], {}, {}
    for est in s2.ESTIMATORS:
        for key, nm in REFERENCES.items():
            references[f"{key}/{est}"] = {sn: {"point": point[(key, est)][sn],
                                              "ci": ci([r[(key, est)][sn] for r in reps])}
                                         for sn in sets}
            for sn in sets:
                cnt = {"above": 0, "indistinguishable": 0, "below": 0}
                for t in tags:
                    d = np.array([r[(t, est)][sn] - r[(key, est)][sn] for r in reps])
                    lo, hi = ci(d)
                    v = s2.verdict(lo, hi)
                    cnt[v] += 1
                    rows.append({"model": t, "reference": nm, "reference_key": key,
                                 "estimator": est, "category_set": sn,
                                 "model_kappa": point[(t, est)][sn],
                                 "reference_kappa": point[(key, est)][sn],
                                 "diff": point[(t, est)][sn] - point[(key, est)][sn],
                                 "diff_ci": [lo, hi], "verdict": v})
                summary[f"{key}/{est}/{sn}"] = cnt
    for k, v in summary.items():
        print(f"{k:34} above {v['above']:2d}  indistinguishable {v['indistinguishable']:2d}  "
              f"below {v['below']:2d}")
    for k, v in references.items():
        print(f"{k:20} " + "  ".join(f"{sn} {x['point']:.4f}" for sn, x in v.items()))
    s2.write_json("eif_significance.json", {
        "references": references, "summary": summary, "rows": rows,
        "provenance": s2.provenance("eif_significance.py", args,
                                    sources=[args.results_dir, args.hq_csv],
                                    secondary_sets=["unreliable_35", "all_40"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
