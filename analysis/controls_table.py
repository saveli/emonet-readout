#!/usr/bin/env python
"""One table, one row per control: does the generative-vs-verification contrast survive it?

Assembles `paper/tables/controls.tex` from the control JSONs, and computes the rows that had a
point estimate on disk but no paired fleet interval. Every row reports, on the reliable five:

  kappa_w generative / verification (fleet means), Delta = verification - generative with a
  paired image-bootstrap 95% CI of the fleet mean (B=500, seed-0 draws, calibration refit in
  every replicate), and how many models clear that row's anchor: point estimate, with the
  number whose paired CI of (model - anchor) lies entirely above it in parentheses.

Rows and their sources:

  as reported            computed here; point = contrast_as_reported.json
  no answer prefix       computed here from results_verify/ (8 models whose readout is defined
                         without the prefix); verification point = verify_nopf_bootstrap_ci.json
  coverage-matched       computed here, both arms on the images the generative arm parsed;
                         point = contrast_coverage_matched_coverage.json
  retry arm included     computed here, generative = best of paper_once / ours / paper_retry
                         (full 2500-image retry runs only); A_settings: Qwen3-VL 0.488,
                         Gemma-3-12b 0.480
  held-out calibration   calibration_both_arms.json, tie-aware held-out fit, both arms
  thresholded P(yes)     computed here, P(yes) > 0.5; point = readout_and_taxonomy_ablations.json
  matched resolution     matched_resolution.json (verification column = matched)
  same raw mapping       calibration_both_arms.json, no label information in either arm
  3-rater anchor         anchor_three_rater.json, both arms on the leave-one-rater-out path
  multi-sample gen.      multi_sample_control.json (text = mean of K=16 sampled parses, 500
                         images); labelled 'k of 11' until all eleven runs land, pending if absent

Guards: printed headline figures (controls_common.reproduce) and every per-row point estimate
listed above against its source JSON.

Usage:
    python analysis/controls_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controls_common import Drift, add_common_args, ci, pmap  # noqa: E402
from e0_report import load_arm_results  # noqa: E402
from e1_report import load_verify_results  # noqa: E402

A = Path("analysis")
EXACT = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--nopf-dir", type=Path, default=Path("results_verify"))
    ap.add_argument("--out", type=Path, default=Path("results/controls_table.json"))
    ap.add_argument("--tex", type=Path, default=Path("paper/tables/controls.tex"))
    args = ap.parse_args()
    return compute(args)


def compute(args) -> int:
    """The controls table: held-out map primary, in-sample (oracle) Delta as a
    secondary column. Computed here (one shared bootstrap): as reported, no answer prefix,
    coverage-matched (anchor recomputed on each model's own parsed images), retry loop included,
    in-sample map. Imported, each after `scoring.check_provenance` (same module version, B,
    seed, S, split seed, category set, unchanged inputs): matched resolution, raw scale, single
    expert through the mapping and three-expert panel (anchor_three_rater), binary detection
    without mapping (detection_intensity), multi-sample (multi_sample_control). The binary
    detection rows take the place of a thresholded-P(yes) row. Output: results/controls_table.json,
    tables/controls.tex (and tables/controls.tex).
    """
    import scoring as s2

    D, gen, ver, head = s2.headline(args)
    s2.assert_headline(head)
    tags = sorted(ver)
    rel, n = D["rel"], D["n"]
    B = args.replicates

    imp = {}
    # Multi-sample: the full 2,500-image set (all eleven models, full-set reference) replaces
    # the 500-image subset in the table when it exists; the subset stays in its own JSON.
    ms_name = ("multi_sample_control_full"
               if (s2.RESULTS_DIR / "multi_sample_control_full.json").exists()
               else "multi_sample_control")
    for name in ("fleet", "matched_resolution", "anchor_three_rater", "calibration_both_arms",
                 "detection_intensity", ms_name):
        path = s2.RESULTS_DIR / f"{name}.json"
        if not path.exists():
            if name.startswith("multi_sample_control"):
                imp[name] = None
                continue
            raise Drift(f"STOP: {path} missing; run its script first")
        imp[name] = json.load(open(path))
        s2.check_provenance(imp[name]["provenance"], B=B, seed=args.seed, S=args.S, label=name)
    print("imported JSONs: provenance matches (module sha, B, seed, S, split seed, category set, "
          "input hashes)")

    nopf = {t: m.astype(np.float64) for t, (m, _) in
            load_verify_results(args.nopf_dir, D["short"], n).items()}
    arms_all = load_arm_results(args.gen_dir, D["short"], n)
    full = s2.Sample(D, np.arange(n), S=args.S)
    retry = {}
    for t in tags:
        cands = [(gen[t][0], gen[t][1])]
        r = arms_all.get((t, "paper_retry"))
        if r and r[1].get("total_images") == n:
            cands.append(("paper_retry", r[0].astype(np.float64)))
        retry[t] = max(cands, key=lambda c: full.model(c[1], rel, s2.PRIMARY))
    parsed = {t: ~np.isnan(gen[t][1][:, rel[0]]) for t in tags}
    computed = {
        "as_reported": {t: (gen[t][1], ver[t]) for t in tags},
        "no_prefix": {t: (gen[t][1], nopf[t]) for t in sorted(nopf)},
        "retry": {t: (retry[t][1], ver[t]) for t in tags},
        # J6: the benchmark's own prompt for every model (no selection between prompts)
        "own_prompt": {t: (arms_all[(t, "paper_once")][0].astype(np.float64), ver[t]) for t in tags},
        # off-table: P(yes) > 0.5 through the map (two tie blocks -> at most two levels, no
        # index artefact any more); feeds the supplementary readout figure only
        "threshold": {t: (ver[t], (ver[t] > 0.5).astype(np.float64)
                          + np.where(np.isnan(ver[t]), np.nan, 0.0)) for t in tags},
    }

    def score(Sm, rep=None):
        o = {"anchor": Sm.anchor(rel, "mean")}
        for key, ms in computed.items():
            for t, (g, v) in ms.items():
                for est in s2.ESTIMATORS:
                    o[(key, t, est)] = (Sm.model(g, rel, est), Sm.model(v, rel, est))
        for t in tags:
            St = Sm.restrict(parsed[t], rep=rep, S=args.S)
            o[("coverage_anchor", t)] = St.anchor(rel, "mean")
            for est in s2.ESTIMATORS:
                o[("coverage", t, est)] = (St.model(gen[t][1], rel, est), St.model(ver[t], rel, est))
        return o

    point = score(full)
    dr = s2.draws(n, B, args.seed)
    print(f"paired image bootstrap B={B}, S={args.S}, {args.workers} workers", flush=True)
    reps = pmap(lambda b: score(s2.Sample(D, dr[b], rep=b, S=args.S), rep=b), list(range(B)),
                args.workers)
    anc_r = np.array([r["anchor"] for r in reps])

    def prov(source, **kw):
        return {"source": source, "B": B, "seed": args.seed, "S": args.S,
                "category_set": "reliable_5", **kw}

    table = []

    def computed_row(key, label, note="", in_table=True):
        ms = list(computed[key]) if key != "coverage" else tags
        row = {"control": key, "label": label, "n_models": len(ms), "note": note,
               "in_table": in_table,
               "per_model": {t: {est: list(point[(key, t, est)]) for est in s2.ESTIMATORS}
                             for t in ms},
               "provenance": prov("computed by analysis/controls_table.py")}
        for est in s2.ESTIMATORS:
            g = np.array([point[(key, t, est)][0] for t in ms])
            v = np.array([point[(key, t, est)][1] for t in ms])
            G = np.array([[r[(key, t, est)][0] for t in ms] for r in reps])
            V = np.array([[r[(key, t, est)][1] for t in ms] for r in reps])
            if key == "coverage":
                a = np.array([point[("coverage_anchor", t)] for t in ms])
                A = np.array([[r[("coverage_anchor", t)] for t in ms] for r in reps])
            else:
                a = np.full(len(ms), point["anchor"])
                A = np.repeat(anc_r[:, None], len(ms), 1)
            e = {"anchor": float(a.mean()), "text": float(g.mean()), "pyes": float(v.mean()),
                 "delta": float((v - g).mean()), "delta_ci": ci((V - G).mean(1)),
                 "pyes_above": int((v > a).sum()),
                 "pyes_above_ci": int(sum(ci(V[:, i] - A[:, i])[0] > 0 for i in range(len(ms)))),
                 "text_above": int((g > a).sum()),
                 "text_above_ci": int(sum(ci(G[:, i] - A[:, i])[0] > 0 for i in range(len(ms))))}
            row[est] = e
        table.append(row)

    computed_row("as_reported", "As reported")
    # cross-check against fleet.json: same draws and splits -> identical
    F = imp["fleet"]["sets"]["reliable_5"]["estimators"]
    for est in s2.ESTIMATORS:
        a, b = table[0][est], F[est]
        if not (abs(a["delta"] - b["fleet"]["delta"]) < 1e-12
                and np.allclose(a["delta_ci"], b["fleet"]["delta_ci"], atol=1e-12)
                and a["pyes_above_ci"] == b["conditions"]["pyes"]["above_ci"]
                and a["text_above_ci"] == b["conditions"]["text"]["above_ci"]):
            raise Drift(f"STOP: as-reported row ({est}) differs from results/fleet.json")
    print("as-reported row identical to results/fleet.json (both estimators)")
    computed_row("no_prefix", "No answer prefix",
                 f"{len(nopf)} models; GLM, MiMo, Qwen3.5 give no readable answer without it")
    computed_row("coverage", "Coverage-matched images",
                 "both conditions and the expert reference on each model's parsed images")
    computed_row("own_prompt", "Benchmark's own prompt only",
                 "text = paper_once for every model; the reported text is the better of two "
                 "prompts per model (a text-favourable upper envelope)")
    n_retry = sum(retry[t][0] == "paper_retry" for t in tags)
    computed_row("retry", "Retry loop included",
                 f"text = best of paper_once/ours/paper_retry; retry chosen for {n_retry} models")

    computed_row("threshold", "P(yes) thresholded at 0.5 (first column: continuous P(yes))",
                 "supplementary readout figure only; 'text' fields hold continuous P(yes)",
                 in_table=False)

    def imported(control, label, src, ho, orc, note=""):
        row = {"control": control, "label": label, "n_models": ho.pop("n_models", 11),
               "note": note, "heldout": ho, "oracle": orc, "in_table": True,
               "provenance": prov(f"results/{src}.json",
                                  module_sha=imp[src]["provenance"]["code_sha"])}
        table.append(row)

    # in-sample map (upper bound): the as-reported computation under the oracle estimator
    o = table[0]["oracle"]
    imported("oracle_mapping", "In-sample mapping (upper bound)", "fleet", dict(o), None,
             "map fit on the scored images; uses their gold marginal")
    table[-1]["provenance"]["source"] = "computed by analysis/controls_table.py"

    mr = imp["matched_resolution"]["estimators"]

    def mr_cols(est):
        f = mr[est]["fleet"]
        return {"anchor": float(np.mean([r["anchor_same_images"] for r in mr[est]["rows"]])),
                "text": f["kappa_text"], "pyes": f["kappa_matched"],
                "delta": f["matched_minus_text"], "delta_ci": f["matched_minus_text_ci"],
                "pyes_above": f["matched_point_above_anchor"],
                "pyes_above_ci": f["matched_ci_above_anchor"],
                "text_above": None, "text_above_ci": None}
    f = mr["heldout"]["fleet"]
    imported("matched_resolution", "Matched resolution", "matched_resolution",
             mr_cols("heldout"), mr_cols("oracle"),
             f"P(yes) column = P(yes) order with the text histogram (tie-aware); ordering share "
             f"{f['ordering_share']:.2f} [{f['ordering_share_ci'][0]:.2f}, "
             f"{f['ordering_share_ci'][1]:.2f}]")

    cb = imp["calibration_both_arms"]["fleet"]["raw"]
    imported("raw_scale", "No mapping, raw scale", "calibration_both_arms",
             {"anchor": imp["calibration_both_arms"]["anchor"], "text": cb["text"],
              "pyes": cb["pyes"], "delta": cb["delta"], "delta_ci": cb["delta_ci"],
              "pyes_above": cb["pyes_point_above_anchor"], "pyes_above_ci": cb["pyes_ci_above_anchor"],
              "text_above": cb["text_point_above_anchor"], "text_above_ci": cb["text_ci_above_anchor"]},
             None, "text as emitted; P(yes) as round(7 P(yes)); no label information")

    a3 = imp["anchor_three_rater"]["sets"]["reliable_5"]

    def a3_cols(est, ref, comp, mk):
        fl = a3["fleet"][f"{mk}/{est}"]
        tp, tg = a3["tests"][f"pyes/{est}"][comp], a3["tests"][f"text/{est}"][comp]
        return {"anchor": a3["references"][f"{ref}/{est}"]["point"], "text": fl["text"],
                "pyes": fl["pyes"], "delta": fl["delta"], "delta_ci": fl["delta_ci"],
                "pyes_above": tp["point_above"], "pyes_above_ci": tp["ci_above"],
                "text_above": tg["point_above"], "text_above_ci": tg["ci_above"]}
    imported("single_mapped", "One expert through the mapping", "anchor_three_rater",
             a3_cols("heldout", "single_mapped", "std_vs_single_mapped", "std"),
             a3_cols("oracle", "single_mapped", "std_vs_single_mapped", "std"),
             "reference = one expert's ratings mapped like the models, vs the other three")
    imported("three_expert_panel", "Three-expert panel, held-out expert", "anchor_three_rater",
             a3_cols("heldout", "panel", "loro_vs_panel", "loro"),
             a3_cols("oracle", "panel", "loro_vs_panel", "loro"),
             "reference = mean of three experts vs the fourth; models mapped to the same marginal")

    # B4: only the two native detection rows enter the table; the calibrated detection kappa
    # depends on the target marginal's forced prevalence (see detection_intensity.py) and stays
    # in that script's table, labelled with its target.
    di = imp["detection_intensity"]
    for r_, lab, note in (("native", "Binary detection, no mapping",
                           "Cohen kappa on present; text > 0, P(yes) > 0.5; reference = expert pairs"),
                          ("native_prev_heldout", "Binary detection, prevalence-matched",
                           "P(yes) threshold at the experts' present rate, fit held-out")):
        b = di["readouts"][r_]
        imported(f"detection_{r_}", lab, "detection_intensity",
                 {"anchor": di["human"]["detection"], "text": b["text"]["detection"],
                  "pyes": b["pyes"]["detection"], "delta": b["delta_detection"],
                  "delta_ci": b["delta_detection_ci"],
                  "pyes_above": b["pyes"]["detection_above_human_point"],
                  "pyes_above_ci": b["pyes"]["detection_above_human_ci"],
                  "text_above": b["text"]["detection_above_human_point"],
                  "text_above_ci": b["text"]["detection_above_human_ci"]}, None, note)

    ms = imp[ms_name]
    full_ms = ms_name.endswith("_full")
    if ms is None:
        table.append({"control": "multi_sample", "label": "Multi-sample text parsing (pending)",
                      "n_models": None, "note": "pending", "heldout": None, "oracle": None,
                      "in_table": True,
                      "provenance": prov(None)})
    else:
        def ms_cols(est):
            f = ms["results"]["reliable_5"]["estimators"][est]["fleet"]
            return {"n_models": ms["n_models"], "anchor": ms["results"]["reliable_5"]["anchor_subset"],
                    "text": f["kappa_mean16"], "pyes": f["kappa_pyes"],
                    "delta": f["pyes_minus_mean16"], "delta_ci": f["pyes_minus_mean16_ci"],
                    "pyes_above": f["pyes_above_anchor_point"], "pyes_above_ci": f["pyes_above_anchor_ci"],
                    "text_above": f["mean16_above_anchor_point"], "text_above_ci": f["mean16_above_anchor_ci"]}
        m, tot = ms["n_models"], ms["n_models_total"]
        fh = ms["results"]["reliable_5"]["estimators"]["heldout"]["fleet"]
        oc = ms_cols("oracle")
        oc.pop("n_models")
        base = "Mean of 16 sampled answers" + ("" if full_ms else f" ({ms['n_images']} subset)")
        imported("multi_sample",
                 f"{base} ({m} of {tot})" if m < tot else base,
                 ms_name, ms_cols("heldout"), oc,
                 f"{ms['n_images']} images, "
                 + ("full-set reference" if full_ms else "reference on those images")
                 + "; mean16 - 1 sample "
                 f"{fh['mean16_minus_sample0']:+.3f} [{fh['mean16_minus_sample0_ci'][0]:+.3f}, "
                 f"{fh['mean16_minus_sample0_ci'][1]:+.3f}]")
        table[-1]["provenance"].update(n_images=ms["n_images"])

    # Graded-digit control (analysis/rate_control.py): two rows when its JSON exists, none
    # otherwise -- a pending control is not shown as a number.
    rc_path = s2.RESULTS_DIR / "rate_control.json"
    if rc_path.exists():
        rc = json.load(open(rc_path))
        s2.check_provenance(rc["provenance"], B=B, seed=args.seed, S=args.S, label="rate_control")
        imp["rate_control"] = rc
        for a in ("argmax", "expected"):
            ho = dict(rc["controls_rows"]["heldout"][a])
            orc = dict(rc["controls_rows"]["oracle"][a])
            lab = ho.pop("label")
            orc.pop("label")
            orc.pop("n_models")
            m, tot = ho["n_models"], 11
            imported(f"rate_{a}", lab if m == tot else f"{lab} ({m} of {tot})", "rate_control",
                     ho, orc, f"one 0-7 digit query per category; {rc['n_images']} images, "
                              f"reference on those images; text column = {a} digit")
            table[-1]["provenance"].update(n_images=rc["n_images"])
    else:
        print("rate_control.json absent: graded-digit rows skipped")

    # provenance: every row the same B, seed, S, category set
    keys = ("B", "seed", "S", "category_set")
    ref = {k: table[0]["provenance"][k] for k in keys}
    bad = [r["control"] for r in table if {k: r["provenance"][k] for k in keys} != ref]
    if bad:
        raise Drift(f"STOP: rows with different provenance: {bad}")
    print(f"all {len(table)} rows: {ref}")

    print(f"\n{'control':40}{'n':>3}{'ref':>7}{'text':>7}{'P(yes)':>8}{'Delta':>8}  95% CI"
          f"          P>ref  T>ref  Delta(in-sample)")
    for r in table:
        e = r["heldout"]
        if e is None:
            print(f"{r['label'][:40]:40}  -- pending --")
            continue
        o = r["oracle"]
        ta = "--" if e["text_above"] is None else f"{e['text_above']}({e['text_above_ci']})"
        print(f"{r['label'][:40]:40}{r['n_models']:3d}{e['anchor']:7.3f}{e['text']:7.3f}"
              f"{e['pyes']:8.3f}{e['delta']:+8.3f}  [{e['delta_ci'][0]:+.3f},{e['delta_ci'][1]:+.3f}]"
              f"  {e['pyes_above']:2d}({e['pyes_above_ci']:2d})  {ta:6}"
              + (f"  {o['delta']:+.3f}" if o else "  --"))

    def cnt(a, s_, m):
        return "--" if a is None else f"${a}\\,({s_})/{m}$"

    lines = [
        "% generated by analysis/controls_table.py -- do not edit by hand",
        f"% Five categories. Held-out mapping (two-fold cross-fit, {args.S} splits). Delta = "
        f"P(yes) - text, paired image bootstrap of the fleet mean (B={B}).",
        "% '> ref.': models above the row's reference by point estimate, in parentheses those whose "
        "paired CI of (model - reference) excludes 0. Last column: Delta under the in-sample map.",
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r"Control & ref.\ & text & $P(\mathrm{yes})$ & $\Delta$ [95\% CI] & "
        r"$P(\mathrm{yes})$ $>$ ref.\ & text $>$ ref.\ & $\Delta$, in-sample map \\",
        r"\midrule"]
    for r in table:
        if not r["in_table"]:
            continue
        e = r["heldout"]
        if e is None:
            lines.append(f"{r['label']} & -- & -- & -- & -- & -- & -- & -- \\\\")
            continue
        m = r["n_models"]
        od = "--" if r["oracle"] is None else f"${r['oracle']['delta']:+.3f}$"
        lines.append(f"{r['label']} & ${e['anchor']:.3f}$ & ${e['text']:.3f}$ & ${e['pyes']:.3f}$ & "
                     f"${e['delta']:+.3f}$ $[{e['delta_ci'][0]:+.3f}, {e['delta_ci'][1]:+.3f}]$ & "
                     f"{cnt(e['pyes_above'], e['pyes_above_ci'], m)} & "
                     f"{cnt(e['text_above'], e['text_above_ci'], m)} & {od} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    tex = "\n".join(lines)
    s2.write_json("controls_table.json", {
        "rows": table, "retry_arms": {t: retry[t][0] for t in tags},
        "provenance": s2.provenance("controls_table.py", args,
                                    sources=[args.gen_dir, args.ver_dir, args.nopf_dir, args.hq_csv])})
    s2.write_tex(s2.TABLES_DIR / "controls.tex", tex)
    s2.write_tex(Path("tables/controls.tex"), tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
