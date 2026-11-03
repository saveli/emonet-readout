#!/usr/bin/env python
"""E4 report -- kappa_w against training step, per model and arm, with the label control.

E1 asked "does correctly-formatted training help?" and answered from the END of each run.
E4 asks the sharper version -- does training in the READ-OUT'S OWN format close the
elicitation gap, and how many images does it take to see it -- so the deliverable is a
curve, not a number:

    step 0        the UNTRAINED base under the identical protocol and the identical images
    step 75..608  every checkpoint LLaMA-Factory wrote, in order

and four arms per model:

    a1_sft        verify_sft, LoRA            prompt-matched SFT, the real thing
    a2_dpo        verify_dpo, LoRA            prompt-matched preference objective
    a3_sft_shuf   verify_sft_shuffled, LoRA   THE CONTROL -- same images, same prompts, same
                                              row count, yes/no permuted
    a4_sft_oft    verify_sft, OFT             same data, different adapter family

**Read a1 - a3, not a1 - step 0.** A curve that rises in a1 and rises equally in a3 has
measured format familiarity or regularisation: the model has learned that the answer to this
prompt is a single lowercase token, which is worth points under a P(yes) read-out and is not
affect recognition. Only the gap between the two arms is evidence about labels. The C4
calibration-only margin is +0.011 kappa_w (findings.md §7); a gap below that is noise.

**The margin is a noise floor, not a significance test, and this script no longer issues a
verdict against it.** It used to take the MAXIMUM gap over the nine checkpoints and declare
the arm clear if that single value beat +0.011 -- nine looks at the same contrast with no
multiplicity correction, so gemma-3-4b was reported as clearing on a +0.032 spike at step 225
whose end-of-training value is +0.007. Under a paired bootstrap that gemma contrast spans zero
at eight of nine checkpoints. What is printed here is now descriptive: the end-of-training gap,
how many checkpoints sit above the margin, and where the maximum is. For intervals, run
`analysis/e4_bootstrap.py`, which resamples the images and recomputes both arms together.

Note also that `a4_sft_oft - a3_sft_shuf`, which the table below shows because a3 is the only
control that exists, differs in adapter family AND label validity at once. It is not an OFT
result. The clean adapter contrast is `a1_sft - a4_sft_oft`.

Two things it would be WRONG to conclude:

  * that a null means the training did not take. E1 already measured 94.8% held-out
    preference accuracy on these objectives while the 40-way score did not move, so
    "undertrained" is not available as an explanation.
  * that a rise at step 75 means 2,400 images suffice. It means 2,400 images suffice TO
    MOVE THIS METRIC, which quantile calibration maps onto the gold marginal -- an oracle
    step, as C4 says, and an upper bound rather than a deployable procedure.

Every arm and every checkpoint is scored on the SAME image subset (verify_eval.py's
--limit-mode random with a recorded seed). That is checked here rather than assumed: a curve
assembled from different subsets is not a curve, and the check is cheap.

Usage:
    python analysis/e4_report.py --results-dir $SHARED/results_e4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
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

# Same threshold verify_eval.py uses for its own density accounting, so "emitted emotions
# per image" is one number with one definition rather than two that nearly agree.
DENSITY_THRESHOLD = 0.5

# The C4 calibration-only margin: mean kappa_w that quantile calibration alone buys, over 27
# models. Any arm-minus-control gap under this is not a result.
C4_MARGIN = 0.011

ARM_DESC = {
    "base": "untrained base (step 0)",
    "a1_sft": "verify_sft, LoRA",
    "a2_dpo": "verify_dpo, LoRA",
    "a3_sft_shuf": "verify_sft SHUFFLED (control)",
    "a4_sft_oft": "verify_sft, OFT",
}
ARM_ORDER = ["a1_sft", "a2_dpo", "a4_sft_oft", "a3_sft_shuf"]

TAG_RE = re.compile(r"^(?P<model>.+?)__(?P<arm>[a-z0-9_]+)__step(?P<step>\d+)$")


def load_runs(results_dir: Path):
    """{(model, arm, step): (scores_by_image_index, batch_info)}.

    Keyed on the --tag, never on model_id: every checkpoint of every arm is the same base
    weights with a different adapter, so model_id collides across the entire experiment.
    """
    runs = {}
    for path in sorted(results_dir.glob("*__verify_eval.json")):
        try:
            payload = json.load(open(path))
            info = payload["batch_info"]
        except (json.JSONDecodeError, KeyError, OSError):
            print(f"unreadable, skipped: {path.name}", file=sys.stderr)
            continue
        if info.get("arm") != "verify":
            continue
        tag = info["model"].split(" ")[0]
        m = TAG_RE.match(tag)
        if not m:
            print(f"tag does not parse as model__arm__stepN, skipped: {tag}", file=sys.stderr)
            continue
        scores = {r["image_index"]: r["predicted_emotions"]
                  for r in payload.get("results", []) if r.get("parse_success")}
        runs[(m["model"], m["arm"], int(m["step"]))] = (scores, info)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path,
                    default=Path("$SHARED/results_e4"))
    # The PAPER's hq.csv (one row per rater, `annotations` / `annotator` / `image`), which
    # is the in-repo copy at data/hq.csv -- the same default anchor_significance.py uses.
    # NOT the NAS file at data/emonet-face-hq/data/emonet-face-hq.csv: that
    # one is the dataset's own export, has a different schema (`label`, one row per image),
    # and scan_index cannot read it. Several older scripts default to /tmp/hq.csv, which is
    # a scratch copy that may not exist on a clean machine.
    ap.add_argument("--hq-csv", type=Path, default=Path("data/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--out", type=Path, default=Path("results/e4_report.json"))
    ap.add_argument("--seed", type=int, default=0, help="seed for the C6 trivial baselines")
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        print(f"no E4 verify results in {args.results_dir}", file=sys.stderr)
        return 1

    # --- the subset check, before any scoring -------------------------------------------
    # Files are compared on the SET of image indices they cover, not on a count and not on
    # the recorded seed: a matching seed with a mismatched limit, or a run that died at 380
    # of 400, would both pass a count check and both put two different image sets on one
    # axis. The intersection is what gets scored, so a partially-finished checkpoint
    # degrades the whole table's n rather than silently biasing one point.
    subsets = {k: set(v[0]) for k, v in runs.items()}
    common = set.intersection(*subsets.values())
    ragged = {k: len(s) for k, s in subsets.items() if s != common}
    if ragged:
        print(f"WARNING: {len(ragged)} of {len(runs)} runs cover a different image set; "
              f"scoring the {len(common)}-image intersection.", file=sys.stderr)
        for k, n in sorted(ragged.items())[:8]:
            print(f"  {k[0]} {k[1]} step {k[2]}: {n} images", file=sys.stderr)
    seeds = {(v[1].get("limit_mode"), v[1].get("subset_seed")) for v in runs.values()}
    if len(seeds) > 1:
        print(f"WARNING: mixed subset definitions {sorted(map(str, seeds))}", file=sys.stderr)
    sel = sorted(common)
    if not sel:
        print("no images common to all runs -- nothing comparable to report", file=sys.stderr)
        return 1

    # --- ground truth --------------------------------------------------------------------
    index_map = json.load(open(args.index_map)) if args.index_map.exists() else None
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, _ = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    short = [e.split("|")[-1] for e in emotions]

    # Reliability is computed over ALL 2500 images, not over the 400 scored here. It is a
    # property of the annotation instrument, and pinning it to the project-wide definition
    # keeps E4's five reliable categories the same five every other table in this repo uses.
    # Only the kappa itself is restricted to the evaluated subset.
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j])
                      for j in range(len(emotions))])
    reliable = alpha >= RELIABLE_ALPHA

    human_sel = human[sel, :, :]
    gold_sel = np.where(n_raters[sel] > 0, np.nanmean(human_sel, axis=2), np.nan)
    hh = float(np.nanmean([human_human_kappa(human_sel[:, j, :])
                           for j in range(len(emotions)) if reliable[j]]))

    def score(mat):
        raw_k, cal_k = [], []
        for j in range(len(emotions)):
            if not reliable[j]:
                continue
            raw_k.append(model_single_rater_kappa(raw_bin(mat[:, j]), human_sel[:, j, :]))
            cal_k.append(model_single_rater_kappa(quantile_bin(mat[:, j], gold_sel[:, j]),
                                                  human_sel[:, j, :]))
        return (float(np.nanmean(raw_k)), float(np.nanmean(cal_k)),
                macro_ap(mat, gold_sel, reliable),
                float(np.nanmean((mat > DENSITY_THRESHOLD).sum(axis=1))))

    rows = []
    for (model, arm, step), (scores, info) in sorted(runs.items()):
        mat = np.zeros((len(sel), len(short)), dtype=np.float32)
        for r, i in enumerate(sel):
            emitted = scores[i]
            for j, e in enumerate(short):
                v = emitted.get(e, 0)
                mat[r, j] = v if isinstance(v, (int, float)) else 0.0
        raw_k, cal_k, ap_, dens = score(mat)
        rows.append({"model": model, "arm": arm, "step": step,
                     "kappa_raw": raw_k, "kappa_calibrated": cal_k, "map": ap_,
                     "emit": dens,
                     "mean_yesno_mass": float(info.get("mean_yesno_mass", float("nan"))),
                     "images": len(sel)})

    # C6 trivial baselines on THIS subset. The protocol changes what the metric's degenerate
    # solutions are -- an all-zero prediction beat every model on MAE once already -- so the
    # floor is recomputed here rather than quoted from the full-2500 table.
    baselines = []
    for tag, mat in build_baselines(gold_sel, args.seed).items():
        raw_k, cal_k, ap_, dens = score(np.asarray(mat, dtype=np.float32))
        baselines.append({"arm": tag, "kappa_raw": raw_k, "kappa_calibrated": cal_k,
                          "map": ap_, "emit": dens})

    # --- the curves ----------------------------------------------------------------------
    by_model = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_model[r["model"]][r["arm"]][r["step"]] = r

    print(f"E4 -- kappa_w (quantile-calibrated) against training step")
    print(f"images: {len(sel)} of EmoNet-Face-HQ, identical across every cell")
    print(f"human-human anchor on these images: kappa_w = {hh:.3f} "
          f"(full-set anchor 0.468)")
    print(f"reliable categories: {int(reliable.sum())} of {len(emotions)} at alpha >= "
          f"{RELIABLE_ALPHA}\n")

    for model in sorted(by_model):
        arms = by_model[model]
        base = arms.get("base", {}).get(0)
        steps = sorted({s for a, d in arms.items() if a != "base" for s in d})
        present = [a for a in ARM_ORDER if a in arms] + \
                  [a for a in sorted(arms) if a not in ARM_ORDER and a != "base"]
        print(f"== {model}")
        # The base gets its own line, not just a row spread across the arm columns. While
        # the training jobs are still running `present` is EMPTY, and a table keyed only on
        # arm columns would then print the step-0 baseline nowhere at all -- which is the
        # one number that is already available and worth looking at mid-flight.
        if base is None:
            print("   NO STEP-0 BASELINE -- every number below is uninterpretable on its own.")
        else:
            print(f"   step 0, untrained base: kappa_w {base['kappa_calibrated']:.3f} "
                  f"(raw {base['kappa_raw']:.3f}), mAP {base['map']:.3f}, "
                  f"emit {base['emit']:.1f}, yes/no mass {base['mean_yesno_mass']:.3f}")
        if not present:
            print("   no trained arms scored yet\n")
            continue
        print(f"   {'step':>6}" + "".join(f"{a:>16}" for a in present) + f"{'a1-a3':>10}")
        if base is not None:
            print(f"   {0:>6}" + "".join(f"{base['kappa_calibrated']:>16.3f}" for _ in present)
                  + f"{0.0:>10.3f}")
        for s in steps:
            cells = []
            for a in present:
                r = arms[a].get(s)
                cells.append(f"{r['kappa_calibrated']:>16.3f}" if r else f"{'-':>16}")
            real = arms.get("a1_sft", {}).get(s)
            ctl = arms.get("a3_sft_shuf", {}).get(s)
            delta = (f"{real['kappa_calibrated'] - ctl['kappa_calibrated']:>+10.3f}"
                     if real and ctl else f"{'-':>10}")
            print(f"   {s:>6}" + "".join(cells) + delta)

        # Density is a mandatory read-out (spec §3 metric 3): it is the single number that
        # would have caught the void run's 0.38-emotions-per-image collapse on the day.
        # Reported at the LAST step of each arm, against the human 8.14 and the GT 12.1.
        cells = []
        for a in present:
            last = max(arms[a].items())[1] if arms.get(a) else None
            cells.append(f"{last['emit']:>16.1f}" if last else f"{'-':>16}")
        tail = f"   (base {base['emit']:.1f}, human 8.14, GT 12.1)" if base else ""
        print(f"   {'emit@end':>6}" + "".join(cells) + tail)
        print()

    print("C6 trivial baselines on the same images (raw / calibrated / mAP):")
    for b in sorted(baselines, key=lambda b: -b["kappa_calibrated"]):
        print(f"   {b['arm']:26}{b['kappa_raw']:8.3f}{b['kappa_calibrated']:8.3f}"
              f"{b['map']:8.3f}")

    # --- the read-out --------------------------------------------------------------------
    print(f"\nREAD-OUT (a1_sft - a3_sft_shuf, C4 calibration-only margin {C4_MARGIN:+.3f})")
    verdict = {}
    for model in sorted(by_model):
        arms = by_model[model]
        pairs = [(s, arms["a1_sft"][s]["kappa_calibrated"] - arms["a3_sft_shuf"][s]["kappa_calibrated"])
                 for s in sorted(arms.get("a1_sft", {}))
                 if s in arms.get("a3_sft_shuf", {})]
        if not pairs:
            print(f"   {model}: control or real arm missing -- no read-out")
            continue
        best_step, best = max(pairs, key=lambda p: p[1])
        last_step, last = pairs[-1]
        n_above = sum(1 for _, g in pairs if g > C4_MARGIN)
        verdict[model] = {"best_step": best_step, "best_gap": best,
                          "final_step": last_step, "final_gap": last,
                          "n_checkpoints": len(pairs), "n_above_c4_margin": n_above,
                          "final_above_c4_margin": bool(last > C4_MARGIN)}
        print(f"   {model}: end-of-training {last:+.3f} at step {last_step}"
              f" ({'above' if last > C4_MARGIN else 'below'} the margin); "
              f"{n_above} of {len(pairs)} checkpoints above it; "
              f"max {best:+.3f} at step {best_step}")
    print("   No verdict is issued here. The margin is a noise proxy, not a test, and the"
          "\n   maximum over checkpoints is the wrong statistic to compare to it -- see the"
          "\n   note above. Run analysis/e4_bootstrap.py for CIs on these contrasts.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"images": len(sel), "image_indices": sel,
                   "human_human_kappa": hh, "reliable_threshold": RELIABLE_ALPHA,
                   "density_threshold": DENSITY_THRESHOLD, "c4_margin": C4_MARGIN,
                   "arm_desc": ARM_DESC, "verdict": verdict,
                   "rows": rows, "baselines": baselines}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
