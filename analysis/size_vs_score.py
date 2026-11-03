#!/usr/bin/env python
"""Does parameter count predict the score in this cohort?

The models were chosen as the newest per family that fits one consumer GPU, which spreads
them over an order of magnitude (1.30B to 13.95B) without that spread being an experimental
variable. This script asks whether the spread explains anything, so the paper can say what it
found rather than leaving the reader to wonder.

For P(yes) scoring it does not: Spearman rho is 0.27 on the five reliable categories and 0.12
on all 40, p well above any threshold at n=11. The honest reading is "no relationship
detectable at this cohort size", NOT "size does not matter": eleven points have very little
power. The parsed-text score is reported beside it (rho 0.79 on the five), so the contrast
between the two conditions is visible rather than left out.

Both category sets are reported, because the five and the forty rank the models differently
and a correlation on one is not a correlation on the other. Scores are the per-model held-out
kappa_w of results/fleet.json, whose provenance is checked first.

Parameter counts are the HuggingFace `safetensors.total` for each repo, the same source as
`tab:models` and `docs/model_selection.md`; they are not recoverable from any results JSON,
so they are listed here and checked against the scored models.

Usage:
    python analysis/size_vs_score.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoring as s2  # noqa: E402

# HF safetensors.total, in billions. Mirrors tab:models and docs/model_selection.md.
PARAMS_B = {
    "GLM-4.6V-Flash": 10.29,
    "InternVL3_5-8B-HF": 8.53,
    "MiMo-VL-7B-RL-2508": 8.31,
    "MiniCPM-V-4.6": 1.30,
    "Ministral-3-14B-Instruct-2512": 13.95,
    "Qwen2.5-VL-3B-Instruct": 3.75,
    "Qwen3-VL-8B-Instruct": 8.77,
    "Qwen3.5-9B": 9.65,
    "gemma-3-4b-it": 4.30,
    "gemma-3-12b-it": 12.19,
    "gemma-4-12B-it": 11.96,
}
# The paper's printed figures (sec. A_eif); a drift in the scoring path fails here.
EXPECT = {("pyes", "reliable_5"): (0.27, 0.42), ("pyes", "all_40"): (0.12, 0.73)}


def report(label: str, xs, ys) -> dict:
    rho, p_rho = spearmanr(xs, ys)
    r, p_r = pearsonr(xs, ys)
    print(f"{label:34s} rho={rho:+.3f} (p={p_rho:.3f})   r={r:+.3f} (p={p_r:.3f})   n={len(xs)}")
    return {"label": label, "n": len(xs), "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "pearson_r": float(r), "pearson_p": float(p_r)}


def main() -> int:
    fleet = json.loads((s2.RESULTS_DIR / "fleet.json").read_text())
    s2.check_provenance(fleet["provenance"], B=s2.B_CI, label="fleet")
    models = sorted(PARAMS_B, key=lambda m: PARAMS_B[m])
    if sorted(fleet["models"]) != sorted(models):
        print(f"scored models do not match the parameter list: {fleet['models']}", file=sys.stderr)
        return 1
    xs = [PARAMS_B[m] for m in models]
    print(f"{len(models)} models, {min(xs):.2f}B to {max(xs):.2f}B "
          f"({max(xs) / min(xs):.1f}x)\n")

    rows, kappa = [], {}
    for cset in ("reliable_5", "all_40"):
        cond = fleet["sets"][cset]["estimators"]["heldout"]["conditions"]
        for c in ("pyes", "text"):
            k = {r["model"]: r["kappa"] for r in cond[c]["rows"]}
            kappa[f"{c}_{cset}"] = k
            rows.append({"condition": c, "category_set": cset,
                         **report(f"{c}, {cset}", xs, [k[m] for m in models])})

    bad = [(key, want) for r in rows if (key := (r["condition"], r["category_set"])) in EXPECT
           and ((want := EXPECT[key])[0] != round(r["spearman_rho"], 2)
                or want[1] != round(r["spearman_p"], 2))]
    if bad:
        raise SystemExit(f"STOP: does not reproduce the printed rho/p: {bad}")

    out = s2.write_json("size_vs_score.json", {
        "params_b": PARAMS_B, "correlations": rows, "kappa": kappa,
        "provenance": {"script": "size_vs_score.py", "fleet_code_sha": fleet["provenance"]["code_sha"]}})
    print(f"\nreproduces the printed figures; wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
