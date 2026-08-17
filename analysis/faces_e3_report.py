#!/usr/bin/env python
"""E3 read-out: does the elicitation contrast survive on real photographs?

Scores the paired arms written by `evaluation/faces_eval.py`. The claim being tested is
narrow and must stay narrow: FACES has one forced-choice gold label per image, so there is
no human anchor and no calibration here. Accuracy over six classes against a chance of
1/6 is the whole metric, and the only question is whether verification beats generative
elicitation on the same images and the same models -- as it does on EmoNet.

Three things this reports that a bare accuracy table would hide:

1. **Paired bootstrap over PERSONS, not images.** Each person contributes 12 images (6
   expressions x 2 sets), so resampling images treats correlated rows as independent and
   produces a CI roughly sqrt(12) too narrow. Same unit choice as §D.

2. **The unparsed count for the generative arm.** A model that answers "I cannot tell" is
   scored wrong here, which is the right call for accuracy but conflates refusal with
   misperception -- so the rate is printed rather than buried.

3. **Per-age and per-gender accuracy.** Free from the filename metadata, and §D2's rule
   applies: a spread across groups is only interpretable against models at matched
   accuracy, so the spread is reported next to the accuracy it belongs to and not on its
   own.

Usage:
    python analysis/faces_e3_report.py --results-dir results_faces
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EMOTION_NAMES = {"a": "anger", "d": "disgust", "f": "fear",
                 "h": "happy", "n": "neutral", "s": "sad"}


def load(results_dir: Path):
    """{(tag, arm): records}. Keyed on the tag so the two arms of one model pair up."""
    runs = {}
    for path in sorted(results_dir.glob("*__faces_*.json")):
        payload = json.load(open(path))
        info = payload["batch_info"]
        tag = info["model"].split(" ")[0]
        arm = info["arm"].replace("faces_", "")
        runs[(tag, arm)] = (payload["results"], info)
    return runs


def acc(records):
    return float(np.mean([r["correct"] for r in records])) if records else float("nan")


def numerical_churn(records_a, records_b):
    """Label disagreement between two runs of the SAME arm under different numerics.

    The person bootstrap resamples people; it never perturbs the arithmetic, so it cannot
    see this. Measured on FACES/gemma-4-12B, changing only the batch width -- same code
    path, same inputs, no cache -- moves ~3% of labels, because bf16 logit quantisation
    interacts with batch composition (section F) and this arm scores an argmax over six
    near-tied values (section P/S). Flips largely cancel in aggregate: on 60 images two
    batch widths gave identical accuracy while disagreeing on 3.3% of labels. But "largely"
    is not "always", and a delta smaller than the churn is not a finding.

    Returns the label disagreement rate and the accuracy difference it actually produced,
    which is the number to compare a reported delta against.
    """
    # Key on the BASENAME. The two clusters mount FACES at different absolute paths, so a
    # rerun produced on one and a baseline on the other share every image and no `path`
    # string. Matching on the full path returned an empty intersection and this function
    # then returned None, which the caller reads as "no churn file" -- three models dropped
    # out of the churn table silently. Failing loud is not an option here (a genuinely
    # absent file is legitimate), so the join has to be right.
    by_path_b = {Path(r["path"]).name: r for r in records_b}
    shared = [(a, by_path_b[Path(a["path"]).name])
              for a in records_a if Path(a["path"]).name in by_path_b]
    if not shared:
        return None
    flips = sum(1 for a, b in shared if a["pred"] != b["pred"])
    acc_a = float(np.mean([a["correct"] for a, _ in shared]))
    acc_b = float(np.mean([b["correct"] for _, b in shared]))
    # Whether the two runs read the same mount, which is a proxy for the same cluster and
    # therefore the same GPU architecture. A rerun on different silicon bounds MORE sources
    # of variation than a rerun on the same node, and mixing the two into one "churn floor"
    # would quietly inflate the floor for models that never crossed clusters. Report the
    # two groups apart and let the reader see which is which.
    #
    # None, not True, when the prefix has been stripped. build_release.py reduces `path` to a
    # bare basename, so on the release tree both parents are "." and a naive equality test
    # would report every cross-cluster rerun as same-cluster, collapsing precisely the two
    # error floors this flag exists to keep apart. Unknown must not read as reassuring.
    parents = {Path(records_a[0]["path"]).parent, Path(records_b[0]["path"]).parent}
    same_mount = None if Path(".") in parents else len(parents) == 1
    return {"n": len(shared), "label_churn": flips / len(shared),
            "acc_a": acc_a, "acc_b": acc_b, "acc_delta": abs(acc_a - acc_b),
            "same_mount": same_mount}


def paired_person_bootstrap(ver, gen, B, seed):
    """CI on (verify - generative) accuracy, resampling persons with replacement.

    Both arms are indexed by the same person set and resampled together, so the CI is on
    the paired difference rather than on two independent accuracies.
    """
    by_person_v, by_person_g = defaultdict(list), defaultdict(list)
    for r in ver:
        by_person_v[r["person"]].append(r["correct"])
    for r in gen:
        by_person_g[r["person"]].append(r["correct"])
    people = sorted(set(by_person_v) & set(by_person_g))
    if not people:
        return None
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        draw = rng.choice(people, size=len(people), replace=True)
        v = np.concatenate([by_person_v[p] for p in draw])
        g = np.concatenate([by_person_g[p] for p in draw])
        diffs.append(v.mean() - g.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n_persons": len(people), "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def group_table(records, key):
    out = {}
    for r in records:
        out.setdefault(r[key], []).append(r["correct"])
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


def gate_verdict(gen_payload) -> tuple[str, float | None]:
    """Run the generative-validity gate on one arm. Returns (verdict, gap).

    Imported rather than reimplemented so this report and the standalone gate cannot
    drift apart: an aggregate that silently included an arm the gate rejects is exactly
    the disagreement this is here to prevent.
    """
    from generative_validity_check import analyse
    r = analyse(gen_payload)
    return r["verdict"], r["gap"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_faces"))
    ap.add_argument("--out", type=Path, default=Path("analysis/faces_e3_report.json"))
    ap.add_argument("-B", "--replicates", type=int, default=2000)
    ap.add_argument("--churn-dir", type=Path, default=None,
                    help="a second run of the SAME arm under different numerics (e.g. "
                         "--max-rows 3). Its label disagreement with --results-dir is the "
                         "error component the person bootstrap cannot see; deltas smaller "
                         "than the accuracy difference it produces are not reportable.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", choices=["on", "off"], default="on",
                    help="on: exclude generative arms the validity gate rejects from the "
                         "aggregate, and report what including them would give. A rejected "
                         "arm's accuracy is set by where truncation landed, not by the "
                         "model, so averaging it in is averaging in a parser artifact.")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    runs = load(args.results_dir)
    if not runs:
        print(f"no FACES results in {args.results_dir}")
        return 1
    tags = sorted({t for t, _ in runs})

    print("E3 -- FACES, six-way forced choice, chance 0.167")
    print("Categorical replication: no human anchor, no calibration (see faces_eval.py).\n")
    print(f"{'model':30s}{'generative':>12}{'verify':>9}{'delta':>8}"
          f"{'95% CI (persons)':>22}{'unparsed':>10}")

    rows = []
    for tag in tags:
        ver = runs.get((tag, "verify"))
        gen = runs.get((tag, "generative"))
        row = {"model": tag,
               "acc_verify": acc(ver[0]) if ver else None,
               "acc_generative": acc(gen[0]) if gen else None,
               "unparsed_generative": (sum(1 for r in gen[0] if r["pred"] is None)
                                       if gen else None)}
        if gen:
            row["gate"], row["gate_gap"] = gate_verdict(
                {"results": gen[0], "batch_info": gen[1]})
        if ver and gen:
            row["delta"] = row["acc_verify"] - row["acc_generative"]
            row["paired_ci"] = paired_person_bootstrap(ver[0], gen[0],
                                                       args.replicates, args.seed)
            ci = row["paired_ci"]
            cis = f"[{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]" if ci else "-"
            star = " *" if ci and ci["excludes_zero"] else ""
            print(f"{tag:30s}{row['acc_generative']:12.3f}{row['acc_verify']:9.3f}"
                  f"{row['delta']:+8.3f}{cis:>20}{star:2s}"
                  f"{row['unparsed_generative']:10d}")
        else:
            have = "verify only" if ver else "generative only"
            print(f"{tag:30s}  {have} -- no pair, skipped in the read-out")
        for arm, run in (("verify", ver), ("generative", gen)):
            if run:
                row[f"{arm}_by_age"] = group_table(run[0], "age")
                row[f"{arm}_by_gender"] = group_table(run[0], "gender")
                row[f"{arm}_by_emotion"] = {EMOTION_NAMES.get(k, k): v for k, v in
                                            group_table(run[0], "gold").items()}
        if args.churn_dir and ver:
            alt = args.churn_dir / f"{tag}__faces_verify.json"
            if alt.exists():
                row["numerical"] = numerical_churn(
                    ver[0], json.load(open(alt))["results"])
        rows.append(row)

    all_paired = [r for r in rows if r.get("delta") is not None]
    rejected = [r for r in all_paired if args.gate == "on" and r.get("gate") == "INVALID"]
    paired = [r for r in all_paired if r not in rejected]
    if paired:
        d = np.array([r["delta"] for r in paired])
        sig = sum(1 for r in paired if r["paired_ci"] and r["paired_ci"]["excludes_zero"])
        print(f"\n* = paired CI excludes zero ({sig}/{len(paired)} models)")
        print(f"mean delta {d.mean():+.3f}; verification wins in "
              f"{int((d > 0).sum())}/{len(d)}")
        clear = sum(1 for r in paired
                    if r["delta"] > 0 and r["paired_ci"] and r["paired_ci"]["excludes_zero"])
        null_ = sum(1 for r in paired if r["paired_ci"] and not r["paired_ci"]["excludes_zero"])
        rev = sum(1 for r in paired
                  if r["delta"] < 0 and r["paired_ci"] and r["paired_ci"]["excludes_zero"])
        print(f"read this as {clear} clear / {null_} indistinguishable / {rev} reversed, "
              f"not as {int((d > 0).sum())}/{len(d)}")
    if rejected:
        # State the direction. An exclusion that happens to help the hypothesis has to be
        # visible in the tool that performs it, not only in the paper that reports it.
        da = np.array([r["delta"] for r in all_paired])
        print(f"\nEXCLUDED by the generative-validity gate ({len(rejected)}):")
        for r in rejected:
            print(f"  {r['model']:28s} delta {r['delta']:+.3f}  "
                  f"single-mention gap {r['gate_gap']:+.3f}")
        print(f"  including them: mean {da.mean():+.3f} over {len(da)}, "
              f"verification wins {int((da > 0).sum())}/{len(da)}")
        d_excl = np.array([r["delta"] for r in rejected])
        direction = ("TOWARD" if d_excl.mean() < np.array([r["delta"] for r in paired]).mean()
                     else "AWAY FROM")
        print(f"  the exclusion therefore moves the aggregate {direction} "
              f"a verification advantage -- report this.")
    # Dedented out of `if rejected:` on 2026-08-17. Both of these belong to every run,
    # not to runs where the gate happened to reject something: nested one level deeper,
    # a --churn-dir was read, stored in the JSON, and never printed whenever the gate
    # passed every arm. Same family as the join bugs found the same day, a computation
    # that silently produces nothing rather than failing.
    print("\nEmoNet comparison: verification beat generative for 10/10 models there. "
          "A null here\nwould mean the effect is a property of EmoNet, not of "
          "elicitation -- report either way.")

    withnum = [r for r in paired if r.get("numerical")]
    if withnum:
        print("\nnumerical error component (§W) -- the person bootstrap cannot see this")
        for r in withnum:
            nm = r["numerical"]
            verdict = ("SURVIVES" if abs(r["delta"]) > 2 * nm["acc_delta"]
                       else "NOT SEPARABLE from numerical noise")
            sm = nm.get("same_mount")
            where = ("same cluster" if sm is True else
                     "ACROSS CLUSTERS" if sm is False else "mount unknown")
            print(f"  {r['model']:26s} label churn {nm['label_churn']:5.1%}  "
                  f"accuracy moved {nm['acc_delta']:.3f} on rerun  "
                  f"vs reported delta {r['delta']:+.3f}  -> {verdict}  [{where}]")
        same = [r["numerical"]["acc_delta"] for r in withnum
                if r["numerical"].get("same_mount") is True]
        cross = [r["numerical"]["acc_delta"] for r in withnum
                 if r["numerical"].get("same_mount") is False]
        unknown = [r["numerical"]["acc_delta"] for r in withnum
                   if r["numerical"].get("same_mount") is None]
        if same:
            print(f"  same-cluster rerun  n={len(same)}  accuracy moves up to "
                  f"{max(same):.3f}")
        if cross:
            print(f"  cross-cluster rerun n={len(cross)}  accuracy moves up to "
                  f"{max(cross):.3f}  (different GPU as well as batch width and cache)")
        if unknown:
            print(f"  mount unknown       n={len(unknown)}  accuracy moves up to "
                  f"{max(unknown):.3f}  (paths stripped; cannot tell the groups apart)")
        worst = min(abs(r["delta"]) / r["numerical"]["acc_delta"]
                    for r in withnum if r["numerical"]["acc_delta"])
        print(f"  smallest clearance of a model's delta over its own churn: "
              f"{worst:.1f}x")
        print("  Report both components. A delta inside the churn band is a code-path "
              "artifact, not a model difference.")

        print("\nper-age accuracy (§D2: spread is only interpretable at matched accuracy)")
        for r in paired:
            for arm in ("generative", "verify"):
                g = r.get(f"{arm}_by_age", {})
                if g:
                    sd = float(np.std(list(g.values()), ddof=1))
                    cells = "  ".join(f"{k} {v:.3f}" for k, v in g.items())
                    print(f"  {r['model']:26s} {arm:11s} {cells}   sd {sd:.3f} "
                          f"@ acc {r['acc_' + arm]:.3f}")

    json.dump({"results_dir": str(args.results_dir), "replicates": args.replicates,
               "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
