#!/usr/bin/env python
"""The scoring path: the one estimator every number in the paper goes through.

Import it; `python analysis/scoring.py` reproduces the headline and asserts the printed
constants below.

------------------------------------------------------------------------------------------------
Primary
------------------------------------------------------------------------------------------------
  tie rule fractional, target rater_pooled, held-out two-fold cross-fit with one kappa over all
  cross-fitted outputs per split, mean over S = 20 splits; every kappa is between one column of
  levels and one rater column, averaged over columns, then over categories, then over models.
  midpoint / mean_halfup is kept as a sensitivity, frozen in EXPECT and
  reproduced by `--tie-rule midpoint --target mean_halfup`.

------------------------------------------------------------------------------------------------
Definitions
------------------------------------------------------------------------------------------------

Target marginal (`--target`). Per category, on the (prediction, target) rows where both exist:
  rater_pooled  (PRIMARY) the pooled histogram of the individual 0-7 ratings divided by the
                number of slots (fractional counts, no rounding): the marginal of what a model
                is scored against. For the panel, the pooled ratings of the three non-held-out
                slots.
  mean_halfup   round-half-up of the four-rater mean (J2)
  mean_halfeven numpy half-even rounding of the same mean

Mapping (`fit`, `apply`, `--tie-rule`). The distinct prediction values, sorted, form tie blocks;
block i occupies ranks [s_i, s_i + n_i) of the sorted predictions.
  fractional (PRIMARY) block i is spread over the levels its rank interval overlaps in the target
             histogram, in proportion: a row of level weights, used as fractional counts in the
             confusion matrix (oracle and held-out alike). This is the expected confusion matrix
             of a uniformly random tie-break; the reported number is the kappa OF that expected
             confusion matrix, not the expected kappa of randomised predictions (kappa is a
             nonlinear ratio; the two do not commute). Deterministic, order-free, and it
             preserves the FITTED histogram exactly; the realised histogram on held-out rows is
             not guaranteed to match it. Detection uses the expected 2x2 table; intensity on
             co-present cells is not defined (NaN).
  midpoint   block i gets the ONE level whose slice of the target histogram contains its
             midpoint s_i + n_i/2 (upper on a boundary). A block cannot be split, so a heavily
             tied condition is not marginal-matched: in-sample total-variation distance to the
             target (mean_halfup) is 0.19 for text (99.98% tied) vs 0.011 for P(yes); that lifts
             text by +0.029 over the fractional rule (J1).
  `apply` sends a value seen in the fit to its block (exact lookup) and an unseen value to the
  nearer fitted value (cut at the midpoint between neighbours, upper on an exact midpoint;
  outside the fit range -> the end block) (J4).

Estimators.
  heldout (PRIMARY)  two-fold cross-fit: images split in half, the map fit on one half's rows is
                     applied to the other half and vice versa; the cross-fitted outputs of all
                     images are scored with ONE kappa per split (same support and same kappa as
                     oracle); mean over S = 20 seeded splits. A split is over distinct images, so
                     bootstrap copies of an image never straddle halves.
  oracle (secondary) the same map fit in-sample on the scored images. oracle - heldout measures
                     the effect of in-sample vs cross-fitted calibration (label access, but also
                     half the training size and split-dependent estimation error); it does not
                     isolate label access alone.

Kappa conventions (J5). Every kappa is between one column of model levels (or level weights) and
one column of ratings; kappas are averaged over columns, then categories, then models.
  anchor (single expert)  mean over the six unordered rater-slot pairs, then categories
                          Pooled-pairs (0.468314 vs 0.468161) kept beside it.
  model vs experts        mean over the four slot kappas (`how="mean"`). One kappa pooled over the
                          four slots (`how="pooled"`) is kept in the JSON.
  three-expert panel      fold k: mean of the other three slots, mapped (same rule, estimator and
                          target construction) onto their marginal, scored against slot k; the
                          model analogue is the model mapped onto that same marginal, scored
                          against slot k. Mean over the four folds on both sides. "held out"
                          means a rating slot, which is not a fixed person across rater panels.
  single expert, mapped   slot o's ratings pushed through the models' map, scored against each of
                          the other three slots, mean over those slots and over sources o.
  detection               Cohen's kappa on present (> 0) vs absent: human mean over six pairs;
                          model mean over the four slots (J9).
  Fleet = equal-weight mean over models; Delta = P(yes) - text per model, then mean.
  Undefined kappa (a degenerate table): never dropped. Sample.model/anchor/model_panel raise on
  a NaN category, and p_percentile/holm/bh/verdict raise on non-finite input (p_percentile also
  on fewer replicates than expected), so a hole cannot shrink a support or become a finding.

Bootstrap. Image resampling; draws are sorted; the map and, for heldout, the S splits are redrawn
inside every replicate from a seed that depends only on (split seed, replicate), so every model,
condition and reference in one replicate shares draws and splits (paired). CIs: percentile,
B = 500. Pairwise tests: B = 5000, two-sided percentile p-value
p = min(1, 2 min(#{d* <= 0}+1, #{d* >= 0}+1) / (B+1)), Holm (FWER .05) and BH (FDR .05).

------------------------------------------------------------------------------------------------
Judgment calls
------------------------------------------------------------------------------------------------
J1 Tie rule = fractional (primary). The midpoint rule is also value-only, but it does not
   marginal-match the tied text condition; under it text rises by +0.029 over fractional
   (in-sample, mean_halfup: 0.430 vs 0.401) and Delta falls from +0.150 to +0.124. Ties in P(yes) (bf16 quantisation, 69% of values shared) get the same rule.
J2 Target = rater_pooled (primary). Rounding the four-rater mean manufactures an ordinal
   distribution and moves the forced present share (half-even 0.250, half-up 0.314, individual
   experts 0.198); every mapped number rises monotonically with that share while the unmapped
   anchor stays put. The pooled individual ratings are the population the model is scored
   against (present share 0.198), and need no rounding rule.
J3 Held-out kappa is one kappa over the cross-fitted outputs of all images (not the mean of two
   half kappas): same support and same kappa definition as the oracle.
J4 Unseen prediction values in apply go to the nearer fitted value (no directional bias, unlike a
   floor lookup); it depends on the prediction spacing.
J5 Mean over slots on the model side, same as the anchor's mean over pairs. Pooling the four
   slots instead moves text by about -0.005 and P(yes) by -0.001 (pooled minus mean -0.0046 /
   -0.0013).
   The panel (mean over folds) and the single-expert-mapped reference (mean over the other
   slots) follow the same convention. Pooled versions stay in the JSON.
J6 The reported text condition is each model's better of its two prompts (paper_once, ours),
   chosen on the evaluation images under the primary estimator. That is a text-favourable upper
   envelope (winner's bias toward text; conservative for Delta), held fixed inside the bootstrap,
   so text intervals omit the selection. The controls table has a row with the benchmark's own
   prompt (paper_once) for every model, and fleet.json records both arms per model.
J7 Category set = Krippendorff alpha >= 0.30 on all 2500 images (the reliable five), fixed; all
   40 is a secondary column where a script reports it.
J8 Splits for the point estimate use seeds (3407, s), s < S; inside replicate b they use
   (3407, b + 1, s). One split set is shared by every model and reference in a sample.
J9 Detection uses mean-over-slots Cohen kappa (binary kappa is prevalence-sensitive, and the
   four slots differ in severity 0.13-0.28).
J10 Ties are resolved by value only, never by image index or resample order. Image index
   correlates with gold, so an order-dependent tie-break lends the heavily tied text condition
   about +0.005 and puts every CI off-centre; the fractional rule is order-free by construction.
J11 Each text model is scored on the images where its answer parsed (unparsed share: MiMo 0.228,
   Qwen3-VL 0.072, MiniCPM 0.046, InternVL 0.008, others <= 0.001; printed by `main`) against
   the anchor on all 2500 images; parse failure is not random, so model and anchor then differ
   in image support as well as in rater (the coverage-matched control row removes this by
   recomputing the anchor on each model's own images).

Provenance: CODE_SHA hashes (at import) the docstring-free AST of scoring.py and the modules
in the number path (controls_common, e0_report, e1_report, verify_bootstrap_ci,
reliability_vs_performance); hq.csv, data/hq_image_index_map.json and the prediction directories
are required sources; check_provenance resolves sources against the repository root and fails on
a missing or changed source, a different code hash, primary, estimators, B, seed, S, split seed,
category set, tie rule or target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controls_common import ci, load_arms, load_hq, pmap  # noqa: E402,F401
from e0_report import load_arm_results  # noqa: E402

ESTIMATORS = ("heldout", "oracle")
PRIMARY = "heldout"
N_LEVELS = 8
S_SPLITS = 20
SPLIT_SEED = 3407
B_CI = 500
SEED = 0
ALPHA = 0.05
_IDX = np.arange(N_LEVELS)
W = (_IDX[:, None] - _IDX[None, :]) ** 2 / (N_LEVELS - 1) ** 2
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
TABLES_DIR = HERE.parent / "tables"

# Headline constants, computed by this module at commit time and frozen here, so a drift in the
# scoring path fails loudly.
ANCHORS = {"anchor_mean": 0.468161, "anchor_pooled": 0.468314, "anchor_mean_all40": 0.201548,
              "anchor_pooled_all40": 0.204212}
# One frozen set per (tie rule / target) setting; model figures in both kappa conventions.
EXPECT = {
    "fractional/rater_pooled": {**ANCHORS, "gen_heldout": 0.425443, "ver_heldout": 0.553066,
                                "gen_oracle": 0.426183, "ver_oracle": 0.554099,
                                "gen_heldout_pooled": 0.424563, "ver_heldout_pooled": 0.553147,
                                "gen_oracle_pooled": 0.425276, "ver_oracle_pooled": 0.554133},
    "midpoint/mean_halfup": {**ANCHORS, "gen_heldout_pooled": 0.428871,
                             "ver_heldout_pooled": 0.552021, "gen_oracle_pooled": 0.429767,
                             "ver_oracle_pooled": 0.553356, "gen_heldout": 0.433423,
                             "ver_heldout": 0.553324, "gen_oracle": 0.434355,
                             "ver_oracle": 0.554667},
}
TOL = 0.0005


# ---------------------------------------------------------------------------------------------
# configuration: tie rule and target marginal (switchable; see "Switches" in the docstring)
# ---------------------------------------------------------------------------------------------
TIE_RULES = ("midpoint", "fractional")
TARGETS = ("mean_halfup", "mean_halfeven", "rater_pooled")
DEFAULT_CONFIG = {"tie_rule": "fractional", "target": "rater_pooled"}   # the primary
CONFIG = dict(DEFAULT_CONFIG)


def configure(tie_rule=None, target=None):
    """Set the tie rule and target marginal for every subsequent fit (process-wide)."""
    if tie_rule is not None:
        assert tie_rule in TIE_RULES, tie_rule
        CONFIG["tie_rule"] = tie_rule
    if target is not None:
        assert target in TARGETS, target
        CONFIG["target"] = target


# ---------------------------------------------------------------------------------------------
# mapping
# ---------------------------------------------------------------------------------------------
def target_levels(gold: np.ndarray) -> np.ndarray:
    """Rounded gold mean as int: half-up (J2, default) or numpy half-even (`mean_halfeven`).
    Gold means of four integers are exact in binary, so the .5 cases are exact."""
    g = np.asarray(gold, dtype=np.float64)
    r = np.round(g) if CONFIG["target"] == "mean_halfeven" else np.floor(g + 0.5)
    return np.clip(r, 0, N_LEVELS - 1).astype(np.int64)


def _valid(pred, target):
    bad = np.isnan(target) if target.ndim == 1 else np.isnan(target).any(axis=1)
    return ~(np.isnan(pred) | bad)


def target_cumulative(target: np.ndarray) -> np.ndarray:
    """Cumulative target counts over levels 0..7, scaled to the number of rows. 1-D target = a
    gold mean (rounded per CONFIG); 2-D target (n, k) = individual ratings, whose pooled
    histogram divided by k is the `rater_pooled` marginal (fractional counts, no rounding)."""
    if target.ndim == 1:
        return np.cumsum(np.bincount(target_levels(target), minlength=N_LEVELS))
    k = target.shape[1]
    return np.cumsum(np.bincount(target.ravel().astype(np.int64), minlength=N_LEVELS)) / k


def fit(pred: np.ndarray, target: np.ndarray):
    """Tie-aware rank-to-histogram map. Returns (sorted distinct values, levels, weights) or None.

    midpoint   levels = one level per distinct value (block midpoint), weights = None
    fractional weights = (m, 8): the share of the block's rank interval [s, s+n) that falls in
               each level's slice of the target histogram = the expected level distribution of a
               uniformly random tie-break; levels = None. Marginal-preserving, deterministic.
    """
    ok = _valid(pred, target)
    if ok.sum() < 2:
        return None
    vals, nv = np.unique(pred[ok], return_counts=True)
    C = target_cumulative(target[ok])
    s = np.cumsum(nv) - nv
    if CONFIG["tie_rule"] == "midpoint":
        mid = s + nv / 2.0
        lev = np.minimum(np.searchsorted(C, mid, side="right"), N_LEVELS - 1)
        return vals, lev.astype(np.float64), None
    C0 = np.concatenate([[0.0], C.astype(np.float64)])
    C0[-1] = max(C0[-1], float(nv.sum()))      # guard float slack on the top boundary
    F = np.zeros((len(vals), N_LEVELS))
    for L in range(N_LEVELS):
        F[:, L] = np.clip(np.minimum(s + nv, C0[L + 1]) - np.maximum(s, C0[L]), 0, None)
    F /= F.sum(axis=1, keepdims=True)
    return vals, None, F


def _lookup(vals, pred):
    """Index of the fitted value for each prediction: exact for seen values, nearest fitted value
    for unseen ones (cut at the midpoint between neighbours, upper on an exact midpoint) (J4)."""
    i = np.searchsorted(vals, pred)
    seen = (i < len(vals)) & (vals[np.minimum(i, len(vals) - 1)] == pred)
    if len(vals) == 1:
        return np.zeros(len(pred), dtype=np.int64)
    mids = (vals[:-1] + vals[1:]) / 2.0
    near = np.searchsorted(mids, pred, side="right")
    return np.where(seen, i, near)


def _empty(pred):
    if CONFIG["tie_rule"] == "midpoint":
        return np.full(pred.shape, np.nan)
    return np.full((len(pred), N_LEVELS), np.nan)


def apply(mapping, pred: np.ndarray) -> np.ndarray:
    """Levels (midpoint: (n,)) or level weights (fractional: (n, 8)) for `pred`; NaN where pred
    is NaN or there is no map."""
    out = _empty(pred)
    if mapping is None:
        return out
    vals, lev, F = mapping
    ok = ~np.isnan(pred)
    idx = _lookup(vals, pred[ok])
    out[ok] = lev[idx] if F is None else F[idx]
    return out


def map_oracle(pred, target):
    """In-sample map."""
    return apply(fit(pred, target), pred)


def map_heldout(pred, target, in_a):
    """Cross-fitted levels: A mapped by the fit on B, B by the fit on A (J3)."""
    out = _empty(pred)
    a, b = in_a, ~in_a
    out[a] = apply(fit(pred[b], target[b]), pred[a])
    out[b] = apply(fit(pred[a], target[a]), pred[b])
    return out


def scored(lv):
    """Boolean mask of rows that carry a mapped prediction (1-D levels or 2-D weights)."""
    return ~np.isnan(lv) if lv.ndim == 1 else ~np.isnan(lv[:, 0])


def present_weight(lv):
    """P(level > 0) per row: 0/1 for midpoint levels, 1 - w_0 for fractional weights; NaN unscored."""
    if lv.ndim == 1:
        return np.where(np.isnan(lv), np.nan, lv > 0)
    return 1.0 - lv[:, 0]


# ---------------------------------------------------------------------------------------------
# kappa
# ---------------------------------------------------------------------------------------------
def kappa_cm(cm: np.ndarray) -> float:
    """Quadratic-weighted kappa from a (possibly fractional) 8x8 count matrix."""
    tot = cm.sum()
    if tot <= 0:
        return float("nan")
    cm = cm / tot
    den = (W * np.outer(cm.sum(1), cm.sum(0))).sum()
    return float(1 - (W * cm).sum() / den) if den > 0 else float("nan")


def kappa_w(x: np.ndarray, y: np.ndarray) -> float:
    """Quadratic-weighted kappa of two int arrays on 0..7."""
    if x.size < 2:
        return float("nan")
    cm = np.bincount(x * N_LEVELS + y, minlength=N_LEVELS * N_LEVELS).reshape(N_LEVELS, N_LEVELS)
    cm = cm / cm.sum()
    den = (W * np.outer(cm.sum(1), cm.sum(0))).sum()
    return float(1 - (W * cm).sum() / den) if den > 0 else float("nan")


def cm_soft(Pw: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Expected count matrix of level weights Pw (n, 8) against int ratings y (n,)."""
    return Pw.T @ np.eye(N_LEVELS)[y]


def cohen(a: np.ndarray, b: np.ndarray) -> float:
    """Unweighted Cohen's kappa of two boolean arrays."""
    if a.size < 2:
        return float("nan")
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def cohen_soft(w: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa of a present-probability vector w against boolean b (expected 2x2 table)."""
    if w.size < 2:
        return float("nan")
    bf = b.astype(np.float64)
    po = np.mean(w * bf + (1 - w) * (1 - bf))
    pa, pb = w.mean(), bf.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def kappa_slots(lv: np.ndarray, H: np.ndarray, how: str = "mean") -> float:
    """Model levels (n,) or weights (n, 8) vs rating slots H (n, k) int. pooled: one kappa over
    all (level, slot) pairs; mean: mean of the k per-slot kappas."""
    ok = scored(lv)
    if ok.sum() < 2:
        return float("nan")
    Hk = H[ok]
    if lv.ndim == 2:
        P = lv[ok]
        cms = [cm_soft(P, Hk[:, k]) for k in range(Hk.shape[1])]
        if how == "pooled":
            return kappa_cm(sum(cms))
        return float(np.mean([kappa_cm(c) for c in cms]))
    x = lv[ok].astype(np.int64)
    if how == "pooled":
        return kappa_w(np.tile(x, Hk.shape[1]), Hk.T.ravel())
    return float(np.mean([kappa_w(x, Hk[:, k]) for k in range(Hk.shape[1])]))


def anchor_cat(H: np.ndarray, how: str = "mean") -> float:
    """Expert-expert kappa_w for one category. mean: mean over the six unordered slot pairs
    (the definition); pooled: one kappa over the six pairs' concatenated tables."""
    K = H.shape[1]
    if how == "pooled":
        xs = np.concatenate([H[:, a] for a in range(K) for b in range(a + 1, K)])
        ys = np.concatenate([H[:, b] for a in range(K) for b in range(a + 1, K)])
        return kappa_w(xs, ys)
    return float(np.mean([kappa_w(H[:, a], H[:, b]) for a in range(K) for b in range(a + 1, K)]))


def detection_hh(Hb: np.ndarray) -> float:
    K = Hb.shape[1]
    return float(np.mean([cohen(Hb[:, a], Hb[:, b]) for a in range(K) for b in range(a + 1, K)]))


def detection_model(present: np.ndarray, Hb: np.ndarray) -> float:
    """present: 0/1 (or a present probability, fractional rule) with NaN for unscored images;
    Hb: (n, k) bool. Mean over slots of Cohen's kappa (J9)."""
    ok = ~np.isnan(present)
    if ok.sum() < 2:
        return float("nan")
    w = present[ok].astype(np.float64)
    if np.all((w == 0) | (w == 1)):
        p = w > 0.5
        return float(np.mean([cohen(p, Hb[ok, k]) for k in range(Hb.shape[1])]))
    return float(np.mean([cohen_soft(w, Hb[ok, k]) for k in range(Hb.shape[1])]))


# ---------------------------------------------------------------------------------------------
# samples: an image set (possibly with bootstrap duplicates) plus its splits
# ---------------------------------------------------------------------------------------------
def split_masks(rows: np.ndarray, n: int, S: int, key: tuple) -> list[np.ndarray]:
    """S two-fold splits of the DISTINCT images in `rows`, as boolean masks over `rows` (J8)."""
    uniq = np.unique(rows)
    out = []
    for s in range(S):
        rng = np.random.default_rng([*key, s])
        in_a = np.zeros(n, bool)
        in_a[uniq[rng.permutation(len(uniq))[: len(uniq) // 2]]] = True
        out.append(in_a[rows])
    return out


class Sample:
    """One image sample. `rep` = None for the point estimate, else the replicate index."""

    def __init__(self, D, rows, rep=None, S=S_SPLITS, split_seed=SPLIT_SEED):
        self.D, self.rows = D, np.asarray(rows)
        self.H = D["Hint"][self.rows]           # (m, 40, 4) int
        self.gold = D["gold"][self.rows]        # (m, 40)
        key = (split_seed,) if rep is None else (split_seed, rep + 1)
        self.masks = split_masks(self.rows, D["n"], S, key)

    def restrict(self, keep_img: np.ndarray, rep, S=S_SPLITS, split_seed=SPLIT_SEED):
        """Sub-sample on images where keep_img (bool over all n) holds; fresh splits of that set.
        `rep` is required (None only for the point estimate) so a replicate never reuses the
        point-estimate split seeds."""
        return Sample(self.D, self.rows[keep_img[self.rows]], rep, S if self.masks else 0,
                      split_seed)

    # -- targets -------------------------------------------------------------------------------
    def tgt(self, j):
        """Mapping target for category j: the four-rater mean (rounded per CONFIG at fit time),
        or the four individual ratings (`rater_pooled`)."""
        if CONFIG["target"] == "rater_pooled":
            return self.H[:, j, :].astype(np.float64)
        return self.gold[:, j]

    def _panel_tgt(self, H, k):
        """Target of fold k: the mean of the other three slots, or their individual ratings."""
        if CONFIG["target"] == "rater_pooled":
            return np.delete(H, k, axis=1).astype(np.float64)
        return (H.sum(axis=1) - H[:, k]) / (H.shape[1] - 1)

    # -- models --------------------------------------------------------------------------------
    def levels(self, p, g, est, m=None):
        if est == "oracle":
            return map_oracle(p, g)
        return map_heldout(p, g, m)

    def cat(self, mat, j, est, how="mean", target=None):
        """kappa of a prediction matrix (n_all, 40) for category j under estimator `est`."""
        p = mat[self.rows, j]
        g = self.tgt(j) if target is None else target
        H = self.H[:, j, :]
        if est == "oracle":
            return kappa_slots(map_oracle(p, g), H, how)
        return float(np.mean([kappa_slots(map_heldout(p, g, m), H, how) for m in self.masks]))

    def model(self, mat, cats, est, how="mean"):
        """Mean over categories. Plain mean: a NaN category is an error, never dropped silently."""
        v = [self.cat(mat, j, est, how) for j in cats]
        if np.isnan(v).any():
            raise ValueError(f"NaN category kappa in {[c for c, x in zip(cats, v) if x != x]}")
        return float(np.mean(v))

    def model_cats(self, mat, cats, est, how="mean"):
        return np.array([self.cat(mat, j, est, how) for j in cats])

    # -- references ----------------------------------------------------------------------------
    def anchor(self, cats, how="mean"):
        """Mean over categories; an undefined category kappa raises, as in model()."""
        v = [anchor_cat(self.H[:, j, :], how) for j in cats]
        if np.isnan(v).any():
            raise ValueError(f"undefined anchor kappa in categories {[c for c, x in zip(cats, v) if x != x]}")
        return float(np.mean(v))

    def anchor_cats(self, cats, how="mean"):
        return np.array([anchor_cat(self.H[:, j, :], how) for j in cats])

    def single_mapped(self, cats, est):
        """One expert pushed through the models' map, vs the other three, mean over sources."""
        out = []
        for j in cats:
            H, g = self.H[:, j, :], self.tgt(j)
            ks = []
            for o in range(H.shape[1]):
                others = np.delete(H, o, axis=1)
                p = H[:, o].astype(np.float64)
                if est == "oracle":
                    ks.append(kappa_slots(map_oracle(p, g), others))
                else:
                    ks.append(np.mean([kappa_slots(map_heldout(p, g, m), others)
                                       for m in self.masks]))
            out.append(np.mean(ks))
        return float(np.mean(out))

    def _panel_cat(self, j, est, pred=None, how="mean"):
        """Pooled over folds k: (panel or model) mapped onto the 3-rater marginal vs slot k."""
        H = self.H[:, j, :]
        K = H.shape[1]
        tot = H.sum(axis=1)

        def one(m):
            """J5: one kappa per fold (levels vs the held-out slot), mean over the folds; with
            how == "pooled" (kept for continuity) one kappa over the folds' summed tables."""
            ks, xs, ys, cm = [], [], [], np.zeros((N_LEVELS, N_LEVELS))
            for k in range(K):
                cons = (tot - H[:, k]) / (K - 1)
                tg = self._panel_tgt(H, k)
                p = cons if pred is None else pred
                lv = map_oracle(p, tg) if m is None else map_heldout(p, tg, m)
                ok = scored(lv)
                if lv.ndim == 2:
                    c = cm_soft(lv[ok], H[ok, k])
                    cm += c
                    ks.append(kappa_cm(c))
                else:
                    xs.append(lv[ok].astype(np.int64))
                    ys.append(H[ok, k])
                    ks.append(kappa_w(xs[-1], ys[-1]))
            if how == "mean":
                return float(np.mean(ks))
            if xs:
                return kappa_w(np.concatenate(xs), np.concatenate(ys))
            return kappa_cm(cm)

        if est == "oracle":
            return one(None)
        return float(np.mean([one(m) for m in self.masks]))

    def panel(self, cats, est):
        return float(np.mean([self._panel_cat(j, est) for j in cats]))

    def model_panel(self, mat, cats, est):
        v = [self._panel_cat(j, est, mat[self.rows, j]) for j in cats]
        if np.isnan(v).any():
            raise ValueError("NaN category kappa in model_panel")
        return float(np.mean(v))


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def add_args(ap):
    ap.add_argument("--hq-csv", type=Path, default=Path("data/hq.csv"))
    ap.add_argument("--index-map", type=Path, default=Path("data/hq_image_index_map.json"))
    ap.add_argument("--gen-dir", type=Path, default=Path("results_e0"))
    ap.add_argument("--ver-dir", type=Path, default=Path("results_verify_prefill"))
    ap.add_argument("-B", "--replicates", type=int, default=B_CI)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--S", type=int, default=S_SPLITS, help="held-out splits")
    ap.add_argument("--workers", type=int, default=30)
    add_switch_args(ap)


def add_switch_args(ap):
    ap.add_argument("--tie-rule", choices=TIE_RULES, default=DEFAULT_CONFIG["tie_rule"],
                    help="tie rule (primary: fractional; midpoint = sensitivity)")
    ap.add_argument("--target", choices=TARGETS, default=DEFAULT_CONFIG["target"],
                    help="target marginal (primary: rater_pooled)")


def load(args):
    configure(getattr(args, "tie_rule", None), getattr(args, "target", None))
    D = load_hq(args)
    D["Hint"] = np.clip(np.round(D["human"]), 0, N_LEVELS - 1).astype(np.int64)
    return D


def select_gen(args, D, arms=("paper_once", "ours")):
    """Reported text condition per model: better of paper_once / ours under the primary
    estimator (J6). Returns gen {tag: (arm, mat)}, ver {tag: mat}, and the full candidate table."""
    _, ver = load_arms(args, D)
    full = Sample(D, np.arange(D["n"]), S=args.S)
    cands, best = {}, {}
    for (tag, arm), (mat, info) in load_arm_results(args.gen_dir, D["short"], D["n"]).items():
        if arm not in arms or info.get("total_images", 0) != D["n"] or tag not in ver:
            continue
        m = mat.astype(np.float64)
        k = full.model(m, D["rel"], PRIMARY)
        cands.setdefault(tag, {})[arm] = k
        if tag not in best or k > best[tag][0]:
            best[tag] = (k, arm, m)
    gen = {t: (a, m) for t, (_, a, m) in best.items()}
    return gen, ver, cands


def draws(n, B, seed):
    """Image resamples, each sorted."""
    rng = np.random.default_rng(seed)
    return [np.sort(rng.integers(0, n, size=n)) for _ in range(B)]


def _finite(x, what, expected=None):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"{what}: {int((~np.isfinite(x)).sum())} non-finite values")
    if expected is not None and x.size != expected:
        raise ValueError(f"{what}: {x.size} values, expected {expected}")
    return x


def p_percentile(d: np.ndarray, expected=None) -> float:
    """Two-sided bootstrap percentile p-value for H0: difference = 0. Non-finite replicates, or
    fewer than `expected`, raise."""
    d = _finite(d, "p_percentile", expected)
    B = len(d)
    lo = (np.sum(d <= 0) + 1) / (B + 1)
    hi = (np.sum(d >= 0) + 1) / (B + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(p, alpha=ALPHA):
    p = _finite(p, "holm")
    o = np.argsort(p, kind="stable")
    rej = np.zeros(len(p), bool)
    for k, i in enumerate(o):
        if p[i] > alpha / (len(p) - k):
            break
        rej[i] = True
    return rej


def bh(p, alpha=ALPHA):
    p = _finite(p, "bh")
    o = np.argsort(p, kind="stable")
    m = len(p)
    passed = [k for k in range(m) if p[o[k]] <= alpha * (k + 1) / m]
    rej = np.zeros(m, bool)
    if passed:
        rej[o[: max(passed) + 1]] = True
    return rej


def verdict(lo, hi):
    _finite([lo, hi], "verdict")
    return "above" if lo > 0 else "below" if hi < 0 else "indistinguishable"


# ---------------------------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------------------------
def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def dir_sha(path: Path, pattern="*.json") -> str:
    """Hash of (name, size, content hash) of every file matching `pattern` in `path`. It covers
    every JSON in the directory, including runs the scorer ignores (e.g. paper_retry n250), so
    an added or edited file anywhere in an input directory invalidates the provenance."""
    h = hashlib.sha256()
    for f in sorted(Path(path).glob(pattern)):
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        h.update(file_sha(f).encode())
    return h.hexdigest()[:16]


# Code in the number path: this module plus everything it calls to load, parse and select.
# Hashed ONCE, at import, so a file edited during a long bootstrap cannot stamp its new hash on
# numbers the old code produced.
CODE_FILES = ("scoring.py", "controls_common.py", "e0_report.py", "e1_report.py",
              "verify_bootstrap_ci.py",
              "reliability_vs_performance.py")


def _code_sha(path: Path) -> str:
    """AST with docstrings removed, so documentation edits do not invalidate provenance but any
    change to code, constants or defaults does. ast.dump's format depends on the Python version
    (e.g. 3.13 added show_empty), so an interpreter upgrade changes every hash: that fails loudly
    (STOP on check), never silently."""
    import ast
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


def _all_code_sha() -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode())
        h.update(_code_sha(HERE / name).encode())
    return h.hexdigest()[:16]


CODE_SHA = _all_code_sha()
ROOT = HERE.parent


def module_sha() -> str:
    """The code hash taken at import (see CODE_FILES)."""
    return CODE_SHA


def _resolve(s) -> Path:
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def _src_key(s) -> str:
    p = _resolve(s).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def provenance(script, args, category_set="reliable_5", sources=(), B=None, **extra):
    """Provenance record. hq.csv and the filename->row index map are always sources."""
    src = {}
    for s in list(sources) + [args.hq_csv, args.index_map]:
        p = _resolve(s)
        if not p.exists():
            raise SystemExit(f"STOP: provenance source {s} does not exist")
        src[_src_key(s)] = dir_sha(p) if p.is_dir() else file_sha(p)
    return {"primary": PRIMARY, "estimators": list(ESTIMATORS),
            "B": B if B is not None else args.replicates, "seed": args.seed, "S": args.S,
            "split_seed": SPLIT_SEED, "category_set": category_set,
            "tie_rule": CONFIG["tie_rule"], "target": CONFIG["target"],
            "code_sha": module_sha(), "code_files": list(CODE_FILES),
            "script": script, "sources": src, **extra}


def check_provenance(prov: dict, *, B=None, seed=SEED, S=S_SPLITS, category_set="reliable_5",
                     label="", need=("results_e0", "results_verify_prefill")):
    """Raise unless an imported result JSON was produced by this code (CODE_SHA), with these settings
    and the current tie rule / target, from inputs that still exist and are unchanged. Paths are
    resolved against the repository root, never the working directory; a missing source fails."""
    want = {"primary": PRIMARY, "estimators": list(ESTIMATORS),
            "seed": seed, "S": S, "split_seed": SPLIT_SEED,
            "category_set": category_set, "code_sha": module_sha(),
            "tie_rule": CONFIG["tie_rule"], "target": CONFIG["target"]}
    if B is not None:
        want["B"] = B
    bad = {k: (prov.get(k), v) for k, v in want.items() if prov.get(k) != v}
    srcs = prov.get("sources", {})
    for nd in ("data/hq.csv", "data/hq_image_index_map.json", *need):
        if nd not in srcs:
            bad[f"source {nd}"] = ("absent from provenance", None)
    for s, h in srcs.items():
        p = _resolve(s)
        if not p.exists():
            bad[f"source {s}"] = (h, "MISSING")
            continue
        now = dir_sha(p) if p.is_dir() else file_sha(p)
        if now != h:
            bad[f"source {s}"] = (h, now)
    if bad:
        raise SystemExit(f"STOP: provenance mismatch in {label}: {bad}")


def write_json(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / name
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {out.relative_to(HERE.parent)}")
    return out


def write_tex(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")


def check(label, got, want, tol=TOL):
    if want is None or got is None:
        print(f"  {label:44}   *** NO *** (missing value or constant)")
        return False
    ok = abs(got - want) <= tol
    print(f"  {label:44}{got:10.6f}{want:10.6f}{got - want:+10.6f}   {'YES' if ok else '*** NO ***'}")
    return ok


def headline(args):
    """The figures every script asserts first, under the current switches."""
    D = load(args)
    gen, ver, _ = select_gen(args, D)
    full = Sample(D, np.arange(D["n"]), S=args.S)
    rel, all40 = D["rel"], D["all40"]
    out = {"anchor_pooled": full.anchor(rel, "pooled"),
           "anchor_pooled_all40": full.anchor(all40, "pooled"),
           "anchor_mean": full.anchor(rel, "mean"),
           "anchor_mean_all40": full.anchor(all40, "mean")}
    for est in ESTIMATORS:
        for how, sfx in (("mean", ""), ("pooled", "_pooled")):
            out[f"gen_{est}{sfx}"] = float(np.mean([full.model(m, rel, est, how)
                                                    for _, m in gen.values()]))
            out[f"ver_{est}{sfx}"] = float(np.mean([full.model(m, rel, est, how)
                                                    for m in ver.values()]))
    return D, gen, ver, out


def config_key():
    return f"{CONFIG['tie_rule']}/{CONFIG['target']}"


def assert_headline(out, tol=TOL):
    """Every frozen constant of the active setting must be present and reproduced. There is one
    frozen set per (tie rule / target) setting in EXPECT; for a setting without one only the
    anchors (no mapping involved) are asserted and the model figures are printed."""
    want = EXPECT.get(config_key(), ANCHORS)
    ok = True
    for k, w in want.items():
        ok &= check(f"{config_key()} {k}", out.get(k), w, tol)
    if config_key() not in EXPECT:
        for k in sorted(set(out) - set(want)):
            print(f"  {config_key()} {k:30}{out[k]:10.6f}   (no frozen set for this setting)")
    if not ok:
        raise SystemExit(f"STOP: the scoring path ({CONFIG}) does not reproduce its frozen headline.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    args = ap.parse_args()
    D, gen, _, out = headline(args)
    print(f"switches {CONFIG}; text condition per model "
          "(arm, share of images without a parsed text answer):")
    for t, (a, m) in sorted(gen.items()):
        print(f"  {t:32} {a:11} unparsed {np.isnan(m[:, D['rel'][0]]).mean():.3f}")
    assert_headline(out)
    print("headline reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
