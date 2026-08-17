#!/usr/bin/env python
"""The no-image control: what does each elicitation score when the face carries no signal?

Both elicitations are re-run with every photograph replaced by a uniform grey image of the
same size (`faces_eval.py --ablate-image grey`), so the prompt, the token count and the read
position are unchanged and only the pixels lose information. Without this, "verification
scores higher than generation" is compatible with verification exploiting label statistics
rather than reading the image.

WHAT THE NUMBERS ACTUALLY SAY, and it is not quite "both arms fall to chance". Every arm
answers `neutral` on every image -- one prediction, 2052 times. FACES is exactly balanced at
342 images per class, so a constant answer scores exactly 1/6 by construction. The reportable
claim is therefore the stronger and simpler one:

    with no facial signal, neither elicitation produces a varying prediction at all

and the 0.167 is an arithmetic consequence of that, not an independent measurement. Anyone
quoting "at chance" without the constant-response fact is quoting a coincidence of the
balanced design. Macro-F1 makes the degeneracy visible where accuracy hides it: a constant
predictor scores 2/(6+1)/6 = 0.048 on six balanced classes, far below the 1/6 that accuracy
reports.

The verification arm additionally exposes its six P(yes) values. Since every image is now the
same grey rectangle, a deterministic forward pass must return the same six numbers every
time, and it nearly does: each model produces exactly TWO distinct score vectors over 2052
identical inputs, one covering 8 images and one covering the other 2044. The images are the
same size, so this is not a resolution effect -- it is the bf16 batch-composition churn of
section F, visible here in its purest form because the inputs are identical and the only
thing left that can vary is which rows share a batch. It moves no label. A count far above
two would mean something in the run varied that should not have.

Usage:
    python analysis/noimg_control.py --results-dir results_faces_noimg
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarise(path: Path) -> dict:
    payload = json.load(open(path))
    info, rows = payload["batch_info"], payload["results"]
    preds = Counter(r["pred"] for r in rows)
    golds = Counter(r["gold"] for r in rows)
    top, top_n = preds.most_common(1)[0]
    tp = Counter(r["gold"] for r in rows if r["pred"] == r["gold"])

    # macro-F1 of the observed prediction vector, over the classes present in gold or pred.
    f1 = []
    for c in set(golds) | set(preds):
        pr = tp[c] / preds[c] if preds[c] else 0.0
        rc = tp[c] / golds[c] if golds[c] else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    out = {"model": info["model_id"].split("/")[-1], "arm": info["arm"],
           "ablate_image": info.get("ablate_image"), "n": len(rows),
           "accuracy": sum(r["correct"] for r in rows) / len(rows),
           "chance": 1.0 / len(golds),
           "macro_f1": sum(f1) / len(f1),
           "distinct_predictions": len(preds),
           "modal_prediction": top, "modal_share": top_n / len(rows),
           "gold_balanced": len(set(golds.values())) == 1,
           "unparsed": info.get("unparsed")}
    if any("scores" in r for r in rows):
        vecs = {tuple(sorted(r["scores"].items())) for r in rows if "scores" in r}
        out["distinct_score_vectors"] = len(vecs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results_faces_noimg"))
    ap.add_argument("--out", type=Path, default=Path("analysis/noimg_control.json"))
    args = ap.parse_args()

    files = sorted(p for p in args.results_dir.glob("*__faces_*.json")
                   if not p.name.startswith("INVALID_"))
    if not files:
        print(f"no ablated FACES results in {args.results_dir}")
        return 1

    rows = [summarise(p) for p in files]
    print(f"{'model':<24}{'arm':<12}{'acc':>7}{'chance':>8}{'macroF1':>9}"
          f"{'distinct':>10}{'modal':>8}{'share':>7}{'score vecs':>12}")
    for r in rows:
        print(f"{r['model']:<24}{r['arm'].replace('faces_',''):<12}{r['accuracy']:>7.3f}"
              f"{r['chance']:>8.3f}{r['macro_f1']:>9.3f}{r['distinct_predictions']:>10d}"
              f"{str(r['modal_prediction']):>8}{r['modal_share']:>7.3f}"
              f"{r.get('distinct_score_vectors', '-'):>12}")

    degenerate = [r for r in rows if r["distinct_predictions"] == 1]
    print(f"\n{len(degenerate)}/{len(rows)} arms emit a single prediction for every image; "
          f"gold balanced: {all(r['gold_balanced'] for r in rows)}")
    print("Report the constant response, not the 1/6 -- the 1/6 follows from the balance.")

    args.out.write_text(json.dumps({"n_arms": len(rows),
                                    "n_degenerate": len(degenerate),
                                    "rows": rows}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
