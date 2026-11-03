#!/usr/bin/env python
"""Detection vs intensity: where does the verification gain come from?

On the reliable five, an expert marks a category present (> 0) on only a minority of images and
the rounded four-rater mean is 0 on most of them, so quadratic-weighted kappa on the 0--7 scale
is dominated by the present/absent decision. This script reports the two parts apart.

  detection   Cohen's kappa on presence (rating > 0). Human pair: every unordered pair of the
              four rating slots. Model: calibrated prediction > 0 against each rater.
  intensity   quadratic-weighted kappa on co-present cells only. Human pair: ordered pairs
              where both raters gave > 0. Model: cells where the calibrated prediction and
              the rater are both > 0, pooled over raters.

The intensity conditioning differs between the human and model columns (a model's "present"
set is fixed by calibration to the gold marginal), so that row is descriptive only.

Per category, beside the per-category calibrated kappa_w: prevalence (fraction of individual
ratings > 0) and the fraction of images whose four-rater mean rounds to 0 -- which, because
calibration rank-matches to that marginal, is also the fraction every model is forced to call
level 0.

Guard: printed headline figures (controls_common.reproduce), then the audit's three
detection means.

Usage:
    python analysis/detection_intensity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controls_common import add_common_args, ci, fci, pmap  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out", type=Path, default=Path("results/detection_intensity.json"))
    ap.add_argument("--tex", type=Path, default=Path("analysis/tables/detection_intensity.tex"))
    args = ap.parse_args()
    return compute(args)


def prevalence_threshold(p, Hb):
    """Threshold tau on P(yes) such that #{p > tau} = round(prevalence * #images), prevalence =
    the experts' rate of ratings > 0 on these images. Ties at tau go to absent (order-free)."""
    ok = ~np.isnan(p)
    if ok.sum() < 2:
        return np.nan
    q = np.sort(p[ok])[::-1]
    k = int(round(Hb[ok].mean() * ok.sum()))
    return q[min(k, len(q) - 1)] if k > 0 else np.inf


def compute(args) -> int:
    """Detection and intensity, image-bootstrap CIs (B=500), reliable five.

    detection (Cohen kappa on present vs absent, mean over slots; human: mean over six pairs),
    three readouts per condition:
      calibrated   level > 0 after the map; held-out (primary) and oracle
      native       text: emitted value > 0 (no fitting at all); P(yes): > 0.5 (no fitting) and
                   > a prevalence-matched threshold (expert present-rate), fit held-out
    intensity      quadratic kappa_w on cells where both sides are > 0 (calibrated levels; human:
                   mean over six pairs). Descriptive: the conditioning set differs per model.
    Prevalence (called-present rate) is reported beside every detection kappa, with the
    experts' rate. Output: results/detection_intensity.json, tables/detection_intensity.tex.
    """
    import scoring as s2

    D, gen, ver, head = s2.headline(args)
    s2.assert_headline(head)
    tags = sorted(ver)
    mats = {"text": {t: gen[t][1] for t in tags}, "pyes": ver}
    rel, n = D["rel"], D["n"]
    READ = ("cal_heldout", "cal_oracle", "native", "native_prev_heldout")

    def intensity(lv, H):
        if lv.ndim == 2:
            return np.nan        # intensity on co-present cells is undefined for level weights
        ok = ~np.isnan(lv)
        ks = []
        for k in range(H.shape[1]):
            m = ok & (H[:, k] > 0) & (np.nan_to_num(lv) > 0)
            ks.append(s2.kappa_w(lv[m].astype(np.int64), H[m, k]) if m.sum() > 1 else np.nan)
        return float(np.nanmean(ks))

    def intensity_hh(H):
        K = H.shape[1]
        ks = []
        for a in range(K):
            for b in range(a + 1, K):
                m = (H[:, a] > 0) & (H[:, b] > 0)
                ks.append(s2.kappa_w(H[m, a], H[m, b]))
        return float(np.nanmean(ks))

    def present(lv):
        return s2.present_weight(lv)

    def model_cat(Sm, c, mat, j):
        p, g, H = mat[Sm.rows, j], Sm.tgt(j), Sm.H[:, j, :]
        Hb = H > 0
        o = {}
        # calibrated, held-out: mean over splits of detection and intensity
        det, inten, prev = [], [], []
        for m in Sm.masks:
            lv = s2.map_heldout(p, g, m)
            det.append(s2.detection_model(present(lv), Hb))
            inten.append(intensity(lv, H))
            prev.append(np.nanmean(present(lv)))
        o["cal_heldout"] = (np.mean(det), np.mean(inten), np.mean(prev))
        lv = s2.map_oracle(p, g)
        o["cal_oracle"] = (s2.detection_model(present(lv), Hb), intensity(lv, H),
                           np.nanmean(present(lv)))
        thr = 0.0 if c == "text" else 0.5
        pres = np.where(np.isnan(p), np.nan, p > thr)
        o["native"] = (s2.detection_model(pres, Hb), np.nan, np.nanmean(pres))
        if c == "pyes":
            det, prev = [], []
            for m in Sm.masks:
                pr = np.full(p.shape, np.nan)
                for fit_, test in ((~m, m), (m, ~m)):
                    tau = prevalence_threshold(p[fit_], Hb[fit_])
                    pr[test] = np.where(np.isnan(p[test]), np.nan, p[test] > tau)
                det.append(s2.detection_model(pr, Hb))
                prev.append(np.nanmean(pr))
            o["native_prev_heldout"] = (np.mean(det), np.nan, np.mean(prev))
        else:
            o["native_prev_heldout"] = o["native"]
        return o

    def score(Sm):
        out = {"hh": np.array([[s2.detection_hh(Sm.H[:, j, :] > 0), intensity_hh(Sm.H[:, j, :]),
                                float((Sm.H[:, j, :] > 0).mean())] for j in rel])}
        for c in mats:
            for t in tags:
                per = [model_cat(Sm, c, mats[c][t], j) for j in rel]
                out[(c, t)] = {r: np.array([x[r] for x in per]) for r in READ}   # (cats, 3)
        return out

    point = score(s2.Sample(D, np.arange(n), S=args.S))
    dr = s2.draws(n, args.replicates, args.seed)
    print(f"image bootstrap B={args.replicates}, S={args.S}, {args.workers} workers", flush=True)
    reps = pmap(lambda b: score(s2.Sample(D, dr[b], rep=b, S=args.S)),
                list(range(args.replicates)), args.workers)

    hh_p = point["hh"].mean(0)
    hh_r = np.stack([r["hh"].mean(0) for r in reps])           # (B, 3)
    res = {"models": tags, "category_set": "reliable_5",
           "human": {"detection": float(hh_p[0]), "detection_ci": ci(hh_r[:, 0]),
                     "intensity": float(hh_p[1]), "intensity_ci": ci(hh_r[:, 1]),
                     "prevalence": float(hh_p[2])},
           "readouts": {}, "categories": []}
    for r_ in READ:
        blk = {}
        for c in mats:
            P = np.array([point[(c, t)][r_].mean(0) for t in tags])                 # (m, 3)
            R = np.stack([[r[(c, t)][r_].mean(0) for t in tags] for r in reps])     # (B, m, 3)
            dd = R[:, :, 0] - hh_r[:, [0]]
            blk[c] = {"detection": float(P[:, 0].mean()), "detection_ci": ci(R[:, :, 0].mean(1)),
                      "intensity": float(np.nanmean(P[:, 1])) if not np.isnan(P[:, 1]).all() else None,
                      "intensity_ci": (ci(np.nanmean(R[:, :, 1], 1))
                                       if not np.isnan(P[:, 1]).all() else None),
                      "prevalence": float(P[:, 2].mean()),
                      "detection_above_human_point": int((P[:, 0] > hh_p[0]).sum()),
                      "detection_above_human_ci": int(sum(ci(dd[:, i])[0] > 0 for i in range(len(tags)))),
                      "detection_below_human_ci": int(sum(ci(dd[:, i])[1] < 0 for i in range(len(tags)))),
                      "per_model": {t: {"detection": float(P[i, 0]), "intensity": float(P[i, 1]),
                                        "prevalence": float(P[i, 2]),
                                        "detection_ci": ci(R[:, i, 0])}
                                    for i, t in enumerate(tags)},
                      "_R": R}
        Rg, Rv = blk["text"].pop("_R"), blk["pyes"].pop("_R")
        blk["delta_detection"] = float(blk["pyes"]["detection"] - blk["text"]["detection"])
        blk["delta_detection_ci"] = ci((Rv[:, :, 0] - Rg[:, :, 0]).mean(1))
        res["readouts"][r_] = blk
    for k, j in enumerate(rel):
        row = {"category": D["short"][j], "alpha": float(D["alpha"][j]),
               "expert_prevalence": float(point["hh"][k, 2]),
               "hh_detection": float(point["hh"][k, 0]), "hh_intensity": float(point["hh"][k, 1])}
        for r_ in READ:
            for c in mats:
                v = np.array([point[(c, t)][r_][k] for t in tags])
                row[f"{c}_{r_}_detection"] = float(v[:, 0].mean())
                row[f"{c}_{r_}_prevalence"] = float(v[:, 2].mean())
        res["categories"].append(row)

    h = res["human"]
    print(f"\nhuman pair: detection {h['detection']:.4f} [{h['detection_ci'][0]:.3f},"
          f"{h['detection_ci'][1]:.3f}]  intensity {h['intensity']:.4f}  prevalence {h['prevalence']:.3f}")
    for r_ in READ:
        b = res["readouts"][r_]
        print(f"{r_:20}" + "  ".join(
            f"{c} det {b[c]['detection']:.4f} [{b[c]['detection_ci'][0]:.3f},{b[c]['detection_ci'][1]:.3f}]"
            f" prev {b[c]['prevalence']:.3f}"
            + (f" int {b[c]['intensity']:.4f}" if b[c]["intensity"] is not None else "")
            + f" >hh {b[c]['detection_above_human_point']}({b[c]['detection_above_human_ci']})"
            for c in mats)
            + f"  Delta {b['delta_detection']:+.4f} [{b['delta_detection_ci'][0]:+.4f},"
              f"{b['delta_detection_ci'][1]:+.4f}]")

    # B4: a calibrated "present" call is level > 0 under the map, so its prevalence is fixed by
    # the target marginal (mean_halfup 0.31, mean_halfeven 0.25, rater_pooled 0.20 on the five,
    # experts 0.20) and its kappa moves with that choice; the label states the target.
    tg = {"mean_halfup": "mean, half-up", "mean_halfeven": "mean, half-even",
          "rater_pooled": "pooled ratings"}[s2.CONFIG["target"]]
    lab = {"cal_heldout": f"calibrated (held-out map, target {tg}), level $>0$",
           "cal_oracle": f"calibrated (in-sample map, target {tg}), level $>0$",
           "native": r"native: text $>0$, $P(\mathrm{yes})>0.5$",
           "native_prev_heldout": r"native: text $>0$, $P(\mathrm{yes})$ at expert prevalence"}
    lines = ["% generated by analysis/detection_intensity.py -- do not edit by hand",
             f"% reliable five; Cohen kappa on present vs absent, image bootstrap B={args.replicates};"
             " prevalence = called-present rate",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Readout & text $\kappa$ (prev.) & $P(\mathrm{yes})$ $\kappa$ (prev.) & $\Delta$ [95\% CI] \\",
             r"\midrule",
             f"experts, pairwise & \\multicolumn{{2}}{{c}}{{${h['detection']:.3f}$ "
             f"(${h['prevalence']:.2f}$)}} & \\\\"]
    for r_ in READ:
        b = res["readouts"][r_]
        lines.append(f"{lab[r_]} & ${b['text']['detection']:.3f}$ (${b['text']['prevalence']:.2f}$) & "
                     f"${b['pyes']['detection']:.3f}$ (${b['pyes']['prevalence']:.2f}$) & "
                     f"${b['delta_detection']:+.3f}$ {fci(*b['delta_detection_ci'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    res["provenance"] = s2.provenance("detection_intensity.py", args,
                                      sources=[args.gen_dir, args.ver_dir, args.hq_csv])
    s2.write_json("detection_intensity.json", res)
    s2.write_tex(s2.TABLES_DIR / "detection_intensity.tex", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
