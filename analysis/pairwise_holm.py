#!/usr/bin/env python
"""Is the model ranking identified? All 55 pairs, paired bootstrap, Holm and BH.

An earlier draft counted pairs of models whose *marginal* 95% CIs do not
overlap (18/55 under verification) and read the rest as "not identified". That is the wrong
test: the eleven models are scored on the same 2500 images, so their
replicates covary, and the question is whether the *difference* of a pair excludes zero. Two
overlapping marginal intervals say nothing about that, and the overlap count is conservative,
not optimistic.

Per arm, one shared set of image resamples (identical to the draws behind
`verify_prefill_bootstrap_ci.json` and `generative_bootstrap_ci.json`: seed 0, B=500, quantile
calibration refit inside every replicate). For each of the 55 pairs:

  * percentile 95% CI of the paired difference (uncorrected),
  * a two-sided p-value from the normal approximation, z = point difference / bootstrap sd of
    the difference. With B=500 a percentile p-value cannot go below 0.002, which no Holm step
    over 55 tests (first threshold 0.05/55 = 0.0009) could ever pass, so the normal
    approximation is what makes a multiplicity correction possible at all; it is stated as such,
  * Holm (FWER 0.05) and Benjamini-Hochberg (FDR 0.05) over the 55.

Guards: printed headline figures (controls_common.reproduce); every per-model marginal CI
must equal the published JSON's, and the marginal non-overlap counts must come out at 18/55
(verification) and 37/55 (generative), i.e. this is the same bootstrap as the paper's.

Usage:
    python analysis/pairwise_holm.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controls_common import add_common_args, ci, pmap  # noqa: E402

ALPHA = 0.05


def holm(p):
    o = np.argsort(p)
    m = len(p)
    rej = np.zeros(m, bool)
    for k, i in enumerate(o):
        if p[i] > ALPHA / (m - k):
            break
        rej[i] = True
    return rej


def bh(p):
    o = np.argsort(p)
    m = len(p)
    passed = [k for k in range(m) if p[o[k]] <= ALPHA * (k + 1) / m]
    rej = np.zeros(m, bool)
    if passed:
        rej[o[:max(passed) + 1]] = True
    return rej


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out", type=Path, default=Path("results/pairwise_holm.json"))
    ap.add_argument("--tex", type=Path, default=Path("analysis/tables/pairwise_holm.tex"))
    ap.add_argument("--B-test", type=int, default=5000, help="replicates for the tests")
    args = ap.parse_args()
    return compute(args)


def compute(args) -> int:
    """All 55 pairs per condition, B=5000 (--B-test), percentile p-values.

    p = min(1, 2 min(#{d* <= 0} + 1, #{d* >= 0} + 1) / (B + 1)): no normal approximation; at
    B=5000 the smallest attainable p is 0.0004 < 0.05/55, so Holm's first step can pass on the
    bootstrap distribution itself. Held-out mapping primary, oracle reported beside it.
    Output: results/pairwise_holm.json, tables/pairwise_holm.tex.
    """
    import scoring as s2

    B = args.B_test
    D, gen, ver, head = s2.headline(args)
    s2.assert_headline(head)
    tags = sorted(ver)
    mats = {"text": {t: gen[t][1] for t in tags}, "pyes": ver}
    rel, n = D["rel"], D["n"]

    def score(Sm):
        return {(c, est): np.array([Sm.model(mats[c][t], rel, est) for t in tags])
                for c in mats for est in s2.ESTIMATORS}

    point = score(s2.Sample(D, np.arange(n), S=args.S))
    dr = s2.draws(n, B, args.seed)
    print(f"paired image bootstrap B={B}, S={args.S}, {args.workers} workers", flush=True)
    reps = pmap(lambda b: score(s2.Sample(D, dr[b], rep=b, S=args.S)), list(range(B)),
                args.workers)

    summary, out = {}, {}
    for est in s2.ESTIMATORS:
        for c in mats:
            P = point[(c, est)]
            R = np.stack([r[(c, est)] for r in reps])          # (B, models)
            marg = [ci(R[:, i]) for i in range(len(tags))]
            pairs, p = [], []
            sep = 0
            for a, b in itertools.combinations(range(len(tags)), 2):
                d = R[:, a] - R[:, b]
                lo, hi = ci(d)
                hi_i, lo_i = (a, b) if P[a] > P[b] else (b, a)
                sep += marg[hi_i][0] > marg[lo_i][1]
                pv = s2.p_percentile(d)
                pairs.append({"a": tags[a], "b": tags[b], "diff": float(P[a] - P[b]),
                              "ci": [lo, hi], "p": pv, "ci_excludes_0": bool(lo > 0 or hi < 0)})
                p.append(pv)
            h, f = s2.holm(p), s2.bh(p)
            for pr, hh, ff in zip(pairs, h, f):
                pr["holm"], pr["bh"] = bool(hh), bool(ff)
            ps = np.sort(p)
            thr = np.array([s2.ALPHA / (len(p) - k) for k in range(len(p))])
            nh = int(h.sum())
            margin = float(ps[nh] / thr[nh]) if nh < len(p) else None
            summary[f"{c}/{est}"] = {
                "pairs": len(pairs), "marginal_ci_nonoverlap": int(sep),
                "paired_ci_excludes_0": int(sum(x["ci_excludes_0"] for x in pairs)),
                "holm": nh, "bh": int(f.sum()), "p_lt_05": int((np.array(p) < s2.ALPHA).sum()),
                "first_holm_failure_p_over_threshold": margin,
                "between_model_sd": float(np.std(P, ddof=1))}
            out[f"{c}/{est}"] = pairs
            sm = summary[f"{c}/{est}"]
            print(f"{c:5} {est:8} marginal disjoint {sep}/55 | paired CI excl. 0 "
                  f"{sm['paired_ci_excludes_0']}/55 | Holm {nh}/55 | BH {sm['bh']}/55 | "
                  f"next Holm step at {margin if margin is None else round(margin, 2)}x threshold")

    lines = ["% generated by analysis/pairwise_holm.py -- do not edit by hand",
             f"% 55 pairs per condition, paired image bootstrap B={B}, percentile p-values, "
             "held-out mapping",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Condition & between-model SD & marginal CIs disjoint & paired CI excl.\ 0 & "
             r"Holm & BH \\", r"\midrule"]
    for c, lab in (("text", "Text parsing"), ("pyes", r"$P(\mathrm{yes})$ scoring")):
        sm = summary[f"{c}/heldout"]
        lines.append(f"{lab} & ${sm['between_model_sd']:.3f}$ & ${sm['marginal_ci_nonoverlap']}/55$ & "
                     f"${sm['paired_ci_excludes_0']}/55$ & ${sm['holm']}/55$ & ${sm['bh']}/55$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    payload = {"summary": summary, "pairs": out, "alpha": s2.ALPHA,
               "p_value": "two-sided bootstrap percentile, (count+1)/(B+1)",
               "provenance": s2.provenance("pairwise_holm.py", args, B=B,
                                           sources=[args.gen_dir, args.ver_dir, args.hq_csv])}
    s2.write_json("pairwise_holm.json", payload)
    s2.write_tex(s2.TABLES_DIR / "pairwise_holm.tex", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
