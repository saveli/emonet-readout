"""Shared loading, reproduction guards and bootstrap plumbing for the robustness control scripts.

The six robustness control scripts (anchor_three_rater,
matched_resolution, pairwise_holm, detection_intensity, calibration_both_arms, controls_table)
all score the same two arms on the same 2500 images, so they share one loader and one guard.
The guard is the point: every script recomputes the paper's printed headline figures through
the scorer it is about to use and aborts before writing anything if one drifts.

  anchor, reliable five         0.468   (human_human_kappa, paper Table 1)
  anchor, all 40                0.204
  generative mean kappa_w       0.410   (best of paper_once/ours per model, 11 models)
  verification mean kappa_w     0.547   (results_verify_prefill, 11 models)
  binarised verification mAP    0.554   (P(yes) > 0.5, 4_results.tex)
  generative mAP                0.561   (same arms, each over its own parsed images)

Per-model values are also checked against `generative_bootstrap_ci.json` and
`verify_prefill_bootstrap_ci.json`, which are what the paper's tables were built from.

Not a script; import it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e0_report import load_arm_results, macro_ap  # noqa: E402
from e1_report import load_verify_results  # noqa: E402
from reliability_vs_performance import (  # noqa: E402
    RELIABLE_ALPHA,
    human_human_kappa,
    krippendorff_alpha,
    parse_into_arrays,
    scan_index,
)
from verify_bootstrap_ci import kappa_cal  # noqa: E402

PAPER = {
    "anchor_reliable_5": 0.468,
    "anchor_all_40": 0.204,
    "gen_mean": 0.410,
    "ver_mean": 0.547,
    "ver_map_binarised": 0.554,
    "gen_map": 0.561,
}
TOL = 0.0005          # figures are printed to three decimals
N_RATERS = 4          # every HQ cell carries exactly four expert ratings (checked on load)
GEN_ARMS = ("paper_once", "ours")


class Drift(SystemExit):
    """Raised when a recomputed figure does not reproduce the printed one."""


def add_common_args(ap):
    ap.add_argument("--hq-csv", type=Path, default=Path("data/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--gen-dir", type=Path, default=Path("results_e0"))
    ap.add_argument("--ver-dir", type=Path, default=Path("results_verify_prefill"))
    ap.add_argument("-B", "--replicates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--S", type=int, default=20, help="held-out splits")
    ap.add_argument("--tie-rule", choices=("midpoint", "fractional"), default="fractional",
                    help="tie rule (scoring.TIE_RULES)")
    ap.add_argument("--target", choices=("mean_halfup", "mean_halfeven", "rater_pooled"),
                    default="rater_pooled", help="target marginal (scoring.TARGETS)")


def load_hq(args) -> dict:
    # The filename -> row-index map decides the join between hq.csv and every local result;
    # without it every local score is scrambled, so its absence is fatal, not a warning.
    if not Path(args.index_map).exists():
        raise Drift(f"STOP: index map {args.index_map} is missing (analysis/build_image_index_map.py)")
    index_map = json.load(open(args.index_map))
    emotions, images, annotators = scan_index(args.hq_csv, index_map)
    human, n_raters, preds = parse_into_arrays(args.hq_csv, emotions, images, annotators)
    if not (n_raters == N_RATERS).all():
        raise Drift(f"expected exactly {N_RATERS} ratings in every cell; "
                    f"got {np.unique(n_raters)}")
    # Rating slots are filled in row order, so slots 0..3 hold all four ratings and 4..7 are
    # empty padding. A slot is a position, not a person: 'rater k' below means 'the k-th
    # rating of the cell', which is what leave-one-rater-out can honestly resample.
    human = human[:, :, :N_RATERS].astype(np.float64)
    short = [e.split("|")[-1] for e in emotions]
    gold = human.mean(axis=2)
    alpha = np.array([krippendorff_alpha(human[:, j, :], n_raters[:, j])
                      for j in range(len(emotions))])
    rel = [j for j in range(len(emotions)) if alpha[j] >= RELIABLE_ALPHA]
    return {"human": human, "gold": gold, "alpha": alpha, "short": short, "rel": rel,
            "all40": list(range(len(emotions))), "n": len(images), "preds": preds,
            "annotators": annotators}


def load_arms(args, D, arms=GEN_ARMS, ver_dir=None):
    """gen: {tag: (arm, matrix)} with the better arm per model on the reliable five, exactly
    as `verify_bootstrap_ci.py --protocol generative` selects it; ver: {tag: matrix}."""
    ver = {t: m.astype(np.float64) for t, (m, _) in
           load_verify_results(ver_dir or args.ver_dir, D["short"], D["n"]).items()}
    full = np.arange(D["n"])
    best = {}
    for (tag, arm), (mat, info) in load_arm_results(args.gen_dir, D["short"], D["n"]).items():
        if arm not in arms or info.get("total_images", 0) != D["n"]:
            continue
        k = kappa_cal(mat, D["human"], D["gold"], D["rel"], full)
        if tag not in best or k > best[tag][0]:
            best[tag] = (k, arm, mat.astype(np.float64))
    gen = {t: (a, m) for t, (_, a, m) in best.items() if t in ver}
    return gen, ver


def check(label, got, want, tol=TOL):
    ok = abs(got - want) <= tol
    print(f"  {label:44}{got:9.4f}{want:9.3f}{got - want:+9.4f}   {'YES' if ok else '*** NO ***'}")
    return ok


def reproduce(D, gen, ver, extra=()):
    """Recompute the printed headline figures; abort before any output if one drifts.

    `extra` is an iterable of (label, got, want) for script-specific figures.
    """
    human, gold, rel, full = D["human"], D["gold"], D["rel"], np.arange(D["n"])
    print(f"\nreproducing printed figures ({len(gen)} generative, {len(ver)} verification)")
    ok = len(gen) == 11 and len(ver) == 11 and set(gen) == set(ver)
    if not ok:
        print(f"  *** expected the same 11 models in both arms: {sorted(gen)} / {sorted(ver)}")
    for name, cats in (("reliable_5", rel), ("all_40", D["all40"])):
        a = float(np.nanmean([human_human_kappa(human[:, j, :]) for j in cats]))
        ok &= check(f"anchor, {name}", a, PAPER[f"anchor_{name}"])
    gk = {t: kappa_cal(m, human, gold, rel, full) for t, (_, m) in gen.items()}
    vk = {t: kappa_cal(m, human, gold, rel, full) for t, m in ver.items()}
    ok &= check("generative mean kappa_w", np.mean(list(gk.values())), PAPER["gen_mean"])
    ok &= check("verification mean kappa_w", np.mean(list(vk.values())), PAPER["ver_mean"])
    relmask = np.isin(np.arange(gold.shape[1]), rel)
    ok &= check("generative mAP", np.mean([macro_ap(m, gold, relmask) for _, m in gen.values()]),
                PAPER["gen_map"])
    ok &= check("binarised verification mAP",
                np.mean([macro_ap((m > 0.5).astype(float), gold, relmask)
                         for m in ver.values()]), PAPER["ver_map_binarised"])
    # Per model, against the JSONs the paper's tables were built from.
    here = Path(__file__).resolve().parent
    gpub = {r["arm"]: r["kappa"] for r in
            json.load(open(here / "generative_bootstrap_ci.json"))["rows"]}
    vpub = {r["arm"]: r["kappa"] for r in
            json.load(open(here / "verify_prefill_bootstrap_ci.json"))["rows"]}
    bad = [t for t, (a, _) in gen.items() if abs(gk[t] - gpub.get(f"{t} [{a}]", -9)) > 1e-9]
    bad += [t for t in ver if abs(vk[t] - vpub.get(t, -9)) > 1e-9]
    if bad:
        print(f"  *** per-model kappa differs from the published JSONs: {bad}")
        ok = False
    else:
        print(f"  {'per-model kappa vs *_bootstrap_ci.json':44}{'22/22 identical':>18}")
    for label, got, want in extra:
        ok &= check(label, got, want)
    if not ok:
        raise Drift("\nSTOP: a printed figure does not reproduce through this scorer, so nothing "
                    "it would write is interpretable. Investigate before rerunning.")
    print("all printed figures reproduced\n")
    return gk, vk


def draws(n, B, seed):
    """The same resample list `verify_bootstrap_ci.py` and `anchor_significance.py` draw."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, size=n) for _ in range(B)]


def ci(v):
    v = np.asarray(v, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# --- fork-based parallel map ------------------------------------------------------------------
# The bootstrap is embarrassingly parallel over replicates. Workers inherit the arrays by fork,
# so nothing large is pickled; only the replicate index goes over the pipe.
_FN = None


def _call(i):
    return _FN(i)


def pmap(fn, items, workers):
    global _FN
    _FN = fn
    if workers <= 1:
        return [fn(i) for i in items]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(_call, items, chunksize=max(1, len(items) // (workers * 4)))


# --- output -----------------------------------------------------------------------------------
def tex_name(tag: str) -> str:
    names = {
        "GLM-4.6V-Flash": "GLM-4.6V-Flash", "InternVL3_5-8B-HF": r"InternVL3\_5-8B",
        "MiMo-VL-7B-RL-2508": "MiMo-VL-7B-RL", "MiniCPM-V-4.6": "MiniCPM-V-4.6",
        "Ministral-3-14B-Instruct-2512": "Ministral-3-14B",
        "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B", "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
        "Qwen3.5-9B": "Qwen3.5-9B", "gemma-3-12b-it": "Gemma-3-12b-it",
        "gemma-3-4b-it": "Gemma-3-4b-it", "gemma-4-12B-it": "Gemma-4-12B-it",
    }
    return names.get(tag, tag.replace("_", r"\_"))


def f3(x, sign=False):
    return f"${x:+.3f}$" if sign else f"${x:.3f}$"


def fci(lo, hi):
    return f"$[{lo:+.3f}, {hi:+.3f}]$"


def write_outputs(json_path: Path, payload: dict, tex_path: Path | None, tex: str | None):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {json_path}")
    if tex_path is not None and tex is not None:
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(tex)
        print(f"wrote {tex_path}")
