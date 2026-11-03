#!/usr/bin/env python
"""Every headline under five (tie rule, target marginal) settings of the scoring path, held-out map.

The primary tie rule and target are open (J2 in scoring.py): the default
midpoint rule does not marginal-match the heavily tied text condition, and the rounding of the
target moves every mapped number. This script runs the same estimator under

  midpoint   / mean_halfup    (current default)
  fractional / mean_halfup
  midpoint   / rater_pooled
  fractional / rater_pooled
  midpoint   / mean_halfeven

and reports, per setting: text, P(yes), Delta [CI]; P(yes) and text above one expert (point /
CI); P(yes) above the three-expert panel (point / CI above / CI below); the matched-resolution
ordering share [CI]; calibrated detection kappa text / P(yes) (level > 0); and the in-sample
total-variation distance between realised and target histograms. Reliable five, held-out
estimator (S=20), paired image bootstrap B=500, same draws and splits in every setting. The
reported text arm is re-selected under each setting (J6); a switch is printed.

Output: results/tie_rule_sensitivity.{md,json}. Does not touch any other output.

Usage:
    python analysis/tie_rule_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoring as s2  # noqa: E402
from controls_common import ci, pmap  # noqa: E402
from matched_resolution import match_tieaware  # noqa: E402

SETTINGS = [("fractional", "rater_pooled"), ("midpoint", "mean_halfup"),
            ("fractional", "mean_halfup"), ("midpoint", "rater_pooled"),
            ("midpoint", "mean_halfeven")]


def tv_distance(mat, D, rel):
    """Mean over (model, category) of the in-sample TV between realised and target histogram."""
    tv = []
    for j in rel:
        p = mat[:, j]
        Sm = s2.Sample(D, np.arange(D["n"]), S=0)
        tg = Sm.tgt(j)
        lv = s2.map_oracle(p, tg)
        ok = s2.scored(lv)
        C = s2.target_cumulative(tg[ok])
        target = np.diff(np.concatenate([[0], C])) / ok.sum()
        got = (lv[ok].sum(0) if lv.ndim == 2
               else np.bincount(lv[ok].astype(int), minlength=s2.N_LEVELS)) / ok.sum()
        tv.append(0.5 * np.abs(target - got).sum())
    return float(np.mean(tv))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    s2.add_args(ap)
    args = ap.parse_args()
    D = s2.load(args)
    rel, n = D["rel"], D["n"]
    dr = s2.draws(n, args.replicates, args.seed)
    default_arms = None
    results = {}

    for tie, tgt in SETTINGS:
        s2.configure(tie, tgt)
        gen, ver, _ = s2.select_gen(args, D)
        tags = sorted(ver)
        arms = {t: gen[t][0] for t in tags}
        if default_arms is None:
            default_arms = arms
        switched = {t: (default_arms[t], a) for t, a in arms.items() if a != default_arms[t]}
        text = {t: gen[t][1] for t in tags}
        parsed = {t: ~np.isnan(text[t][:, rel[0]]) for t in tags}
        verc = {}
        for t in tags:
            v = ver[t].copy()
            v[~parsed[t]] = np.nan
            verc[t] = v

        def score(Sm):
            o = {"anchor": Sm.anchor(rel, "mean"), "panel": Sm.panel(rel, "heldout")}
            for c, ms in (("text", text), ("pyes", ver)):
                k, det = [], []
                for t in tags:
                    kk, dd = [], []
                    for j in rel:
                        p, g, H = ms[t][Sm.rows, j], Sm.tgt(j), Sm.H[:, j, :]
                        ks, ds = [], []
                        for m in Sm.masks:
                            lv = s2.map_heldout(p, g, m)
                            ks.append(s2.kappa_slots(lv, H))
                            ds.append(s2.detection_model(s2.present_weight(lv), H > 0))
                        kk.append(np.mean(ks))
                        dd.append(np.mean(ds))
                    k.append(np.mean(kk))
                    det.append(np.mean(dd))
                o[c] = np.array(k)
                o[f"det_{c}"] = np.array(det)
            o["pyes_panel"] = np.array([Sm.model_panel(ver[t], rel, "heldout") for t in tags])
            mt = {}
            for t in tags:
                m = np.full(ver[t].shape, np.nan)
                for j in rel:
                    col = np.full(n, np.nan)
                    col[Sm.rows] = match_tieaware(ver[t][Sm.rows, j], text[t][Sm.rows, j])
                    m[:, j] = col
                mt[t] = m
            o["matched"] = np.array([Sm.model(mt[t], rel, "heldout") for t in tags])
            o["pyes_same"] = np.array([Sm.model(verc[t], rel, "heldout") for t in tags])
            return o

        print(f"\n[{tie} / {tgt}] B={args.replicates}, S={args.S}"
              + (f"; text arm switches {switched}" if switched else ""), flush=True)
        P = score(s2.Sample(D, np.arange(n), S=args.S))
        R = pmap(lambda b: score(s2.Sample(D, dr[b], rep=b, S=args.S)),
                 list(range(args.replicates)), args.workers)

        def st(k):
            return np.stack([r[k] for r in R])

        A = np.array([r["anchor"] for r in R])
        PN = np.array([r["panel"] for r in R])
        Rt, Rv = st("text"), st("pyes")
        out = {"text": float(P["text"].mean()), "pyes": float(P["pyes"].mean()),
               "delta": float((P["pyes"] - P["text"]).mean()), "delta_ci": ci((Rv - Rt).mean(1)),
               "anchor": P["anchor"], "panel": P["panel"]}
        for c, Rc in (("pyes", Rv), ("text", Rt)):
            lo_hi = [ci(Rc[:, i] - A) for i in range(len(tags))]
            out[f"{c}_above_point"] = int((P[c] > P["anchor"]).sum())
            out[f"{c}_above_ci"] = int(sum(x[0] > 0 for x in lo_hi))
            out[f"{c}_below_ci"] = int(sum(x[1] < 0 for x in lo_hi))
        Rp = st("pyes_panel")
        pp = [ci(Rp[:, i] - PN) for i in range(len(tags))]
        out["panel_above_point"] = int((P["pyes_panel"] > P["panel"]).sum())
        out["panel_above_ci"] = int(sum(x[0] > 0 for x in pp))
        out["panel_below_ci"] = int(sum(x[1] < 0 for x in pp))
        Rm, Rs = st("matched"), st("pyes_same")
        num = (P["matched"] - P["text"]).mean()
        den = (P["pyes_same"] - P["text"]).mean()
        out["matched"] = float(P["matched"].mean())
        out["share"] = float(num / den)
        out["share_ci"] = ci((Rm - Rt).mean(1) / (Rs - Rt).mean(1))
        out["det_text"] = float(P["det_text"].mean())
        out["det_pyes"] = float(P["det_pyes"].mean())
        out["tv_text"] = float(np.mean([tv_distance(text[t], D, rel) for t in tags]))
        out["tv_pyes"] = float(np.mean([tv_distance(ver[t], D, rel) for t in tags]))
        out["text_arms"] = arms
        out["arm_switches"] = {t: list(v) for t, v in switched.items()}
        out["per_model"] = {t: {"text": float(P["text"][i]), "pyes": float(P["pyes"][i])}
                            for i, t in enumerate(tags)}
        # regression: a setting with a frozen headline set must reproduce it (scoring.EXPECT)
        frozen = s2.EXPECT.get(f"{tie}/{tgt}")
        if frozen:
            ok = s2.check(f"{tie}/{tgt} text heldout", out["text"], frozen["gen_heldout"], 1e-6)
            ok &= s2.check(f"{tie}/{tgt} P(yes) heldout", out["pyes"], frozen["ver_heldout"], 1e-6)
            if not ok:
                raise SystemExit("STOP: sensitivity row does not reproduce its frozen headline")
            out["frozen_reproduced"] = True
        results[f"{tie}/{tgt}"] = out
        print(json.dumps({k: v for k, v in out.items() if k not in ("per_model", "text_arms")}))

    s2.configure(**s2.DEFAULT_CONFIG)
    dh = results["fractional/rater_pooled"]
    lines = ["# Tie rule and target marginal: sensitivity of the headline", "",
             "Generated by `analysis/tie_rule_sensitivity.py`; do not edit by hand. Reliable five, "
             f"held-out map (two-fold cross-fit, S={args.S}), paired image bootstrap B="
             f"{args.replicates} (same draws and splits in every row). One expert = mean pairwise "
             f"kappa_w {dh['anchor']:.3f}; three-expert panel {dh['panel']:.3f} (primary row; "
             "panel per row in the JSON). Kappa = mean over the four rater slots (J5). Rows with a "
             "frozen headline in scoring.EXPECT are asserted to reproduce it. Text arm re-selected per row (J6). TV = in-sample "
             "total-variation distance between the realised and the target level histogram "
             "(0 = marginal-matched). Detection = Cohen kappa on mapped level > 0 (experts 0.388).",
             "",
             "| tie rule / target | text | P(yes) | Delta [95% CI] | P(yes) > 1 expert (pt / CI) | "
             "text > 1 expert (pt / CI) | P(yes) vs panel (pt / CI above / CI below) | "
             "matched share [CI] | detection text / P(yes) | TV text / P(yes) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, o in results.items():
        lines.append(
            f"| {key}{' (PRIMARY)' if key == 'fractional/rater_pooled' else ''}"
            f"{' (first setting)' if key == 'midpoint/mean_halfup' else ''} | {o['text']:.3f} | "
            f"{o['pyes']:.3f} | {o['delta']:+.3f} [{o['delta_ci'][0]:+.3f}, {o['delta_ci'][1]:+.3f}] | "
            f"{o['pyes_above_point']}/11 / {o['pyes_above_ci']}/11 | "
            f"{o['text_above_point']}/11 / {o['text_above_ci']}/11 | "
            f"{o['panel_above_point']} / {o['panel_above_ci']} / {o['panel_below_ci']} | "
            f"{o['share']:.2f} [{o['share_ci'][0]:.2f}, {o['share_ci'][1]:.2f}] | "
            f"{o['det_text']:.3f} / {o['det_pyes']:.3f} | {o['tv_text']:.3f} / {o['tv_pyes']:.3f} |")
    sw = {k: o["arm_switches"] for k, o in results.items() if o["arm_switches"]}
    lines += ["", "Text-arm switches relative to the primary: "
              + (json.dumps(sw) if sw else "none.")]
    md = "\n".join(lines) + "\n"
    (s2.RESULTS_DIR / "tie_rule_sensitivity.md").write_text(md)
    s2.write_json("tie_rule_sensitivity.json", {
        "settings": results,
        "provenance": {**s2.provenance("tie_rule_sensitivity.py", args,
                                       sources=[args.gen_dir, args.ver_dir]),
                       "tie_rule": "varies (see settings)", "target": "varies (see settings)"}})
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
