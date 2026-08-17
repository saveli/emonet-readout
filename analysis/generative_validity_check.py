#!/usr/bin/env python
"""Is a FACES generative arm measuring the model, or measuring where truncation landed?

Section Z0c cost two rounds of wrong answers because the success criterion was wrong twice.
`--max-new-tokens 12` scored the reasoning models at 0.000 with 100% unparsed, which at
least looked broken. Raising the cap to 128 dropped unparsed to ~2% and produced *plausible*
numbers -- and those were worse, because nothing flagged them. "The parser found an emotion"
is not "the model reached a conclusion".

Run this on every generative arm before quoting a single accuracy from it.

THE TELL, and the only metric here that does not rest on a heuristic:

    accuracy | reply names exactly one emotion     vs     accuracy over all replies

`parse_generative` resolves a multi-mention reply by last-mention. When a model recites the
prompt's own option list ("Let's check the options: anger, disgust, fear, ...") every class
appears, so last-mention returns whatever the truncation happened to land near. That is a
coin flip dressed as a prediction, and it shows up as a large gap between the two columns.
Section Z0c measured MiniCPM at 0.989 single-mention against 0.615 overall, GLM 0.868 vs
0.691, MiMo 0.821 vs 0.644, Qwen3.5 0.862 vs 0.523. A gap that size is the parser, not
the model.

The supporting columns (mean chars, mid-word rate, multi-mention rate) are heuristics
and are printed to explain *why* an arm fails, never to decide it on their own.

Note that a clean bill of health here does NOT mean the arm was rendered as intended --
that is a separate axis. Section AA0's inverted `--thinking off` produced replies that were
long and truncated for exactly the reason this script detects, but the fix was in the
prompt rendering, not the token cap. Check `batch_info["enable_thinking"]` too.

Usage:
    python analysis/generative_validity_check.py --results-dir results_faces
    python analysis/generative_validity_check.py --results-dir results_faces --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# The six option words as the prompt lists them, matched on these surface forms rather than
# on `parse_generative`'s full synonym table: the failure being detected is the model echoing
# the prompt back, not the model using a synonym.
#
# `multi` counts replies naming MORE THAN ONE of them, which is broader than section Z0c's
# "recites the option list" and deliberately so -- last-mention parsing is ambiguous for any
# multi-mention reply, whatever produced it. Do not read the two numbers as the same column
# (Z0c put MiMo at 11.4% recitation; it is 64.2% multi-mention).
OPTION_WORDS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]

# A reply that stops without terminal punctuation and is long enough to have been going
# somewhere. Deliberately loose: single-word answers ("neutral") have no punctuation either
# and must not count, hence the length floor.
MID_WORD_MIN_CHARS = 40
TERMINAL = re.compile(r"""[.!?"')\]]\s*$""")

# Gap between single-mention accuracy and overall accuracy above which the arm is called on
# the parser rather than on the model. Section Z0c's four invalid arms sat at 0.17-0.37; its
# five clean ones are single-mention by construction and have no gap at all.
GAP_INVALID = 0.10


def count_option_mentions(text: str) -> int:
    low = text.lower()
    return sum(1 for w in OPTION_WORDS if re.search(rf"\b{w}\b", low))


def analyse(payload: dict) -> dict:
    rows = payload.get("results", [])
    raws = [r.get("raw_response") or "" for r in rows]
    lens = [len(s) for s in raws]

    mentions = [count_option_mentions(s) for s in raws]
    single = [i for i, m in enumerate(mentions) if m == 1]
    multi = [i for i, m in enumerate(mentions) if m > 1]

    def acc(idx):
        if not idx:
            return None
        return sum(1 for i in idx if rows[i].get("correct")) / len(idx)

    overall = acc(range(len(rows)))
    single_acc = acc(single)
    gap = None if (overall is None or single_acc is None) else single_acc - overall

    mid_word = sum(1 for s in raws
                   if len(s) >= MID_WORD_MIN_CHARS and not TERMINAL.search(s))

    info = payload.get("batch_info", {})
    return {
        "n": len(rows),
        "mean_chars": statistics.mean(lens) if lens else 0.0,
        "median_chars": statistics.median(lens) if lens else 0.0,
        "mid_word_frac": mid_word / len(raws) if raws else 0.0,
        "multi_mention_frac": len(multi) / len(raws) if raws else 0.0,
        "single_mention_frac": len(single) / len(raws) if raws else 0.0,
        "unparsed": sum(1 for r in rows if r.get("pred") is None),
        "acc_overall": overall,
        "acc_single_mention": single_acc,
        "gap": gap,
        "max_new_tokens": info.get("max_new_tokens"),
        "thinking": info.get("thinking"),
        # None on files written before section AA0 added the field -- and those are exactly
        # the files whose rendering cannot be confirmed from the file itself.
        "enable_thinking": info.get("enable_thinking", None),
        "verdict": ("INVALID" if (gap is not None and gap > GAP_INVALID) else "CLEAN"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_faces"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.results_dir.glob("*__faces_generative.json"))
    if not files:
        raise SystemExit(f"no generative arms under {args.results_dir}")

    out = {}
    print(f"{'model':<26} {'chars':>6} {'midwd':>6} {'multi':>7} {'1-ment':>7} "
          f"{'acc':>6} {'acc|1m':>7} {'gap':>7}  think  verdict")
    for path in files:
        tag = path.name.split("__")[0]
        try:
            payload = json.load(open(path))
        except (json.JSONDecodeError, OSError) as e:
            print(f"{tag:<26} UNREADABLE ({type(e).__name__})")
            continue
        r = analyse(payload)
        out[tag] = r
        et = r["enable_thinking"]
        think = "?" if et is None else ("ON" if et else "off")
        print(f"{tag:<26} {r['mean_chars']:>6.0f} {r['mid_word_frac']:>6.1%} "
              f"{r['multi_mention_frac']:>7.1%} {r['single_mention_frac']:>7.1%} "
              f"{r['acc_overall'] or 0:>6.3f} {r['acc_single_mention'] or 0:>7.3f} "
              f"{r['gap'] or 0:>+7.3f}  {think:>5}  {r['verdict']}")

    bad = [t for t, r in out.items() if r["verdict"] == "INVALID"]
    unknown = [t for t, r in out.items() if r["enable_thinking"] is None]
    print()
    print(f"{len(out) - len(bad)}/{len(out)} arms usable"
          + (f"; INVALID: {', '.join(sorted(bad))}" if bad else ""))
    if unknown:
        print(f"no enable_thinking recorded (pre-AA0 file, rendering unconfirmable): "
              f"{', '.join(sorted(unknown))}")

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
