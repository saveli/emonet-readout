#!/usr/bin/env python
"""Pick and print the Figure 1 example of the paper (sec/2_benchmark.tex, fig:qualitative).

Figure 1 is an inline LaTeX box, not a generated PDF; this script supplies its numbers.
Rule (stated in the figure's source comment):
  model  = the strongest model under text parsing on the reliable five (primary),
  images = those where its text reply parsed AND that have at least one of the five main
           categories (alpha >= 0.30) that >= 3 of the 4 experts rate present with a mean rating
           >= 2, that the reply leaves at 0, and to which P(yes) assigns an expected held-out
           level within 1 of the expert mean,
  pick   = among those, the image whose per-image gain (expected squared error of text minus
           that of P(yes), summed over all 40 categories, mean over the four experts, held-out
           map averaged over the S = 20 splits) is closest to the median gain over ALL parsed
           images of that model.
Run: python analysis/fig1_example.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring as s2  # noqa: E402

L = np.arange(s2.N_LEVELS)


def weights(S, mat):
    """(n, 40, 8) expected level weights under the held-out map, mean over the S splits."""
    W = np.zeros((mat.shape[0], 40, s2.N_LEVELS))
    for j in range(40):
        p = mat[:, j].astype(np.float64)
        W[:, j] = np.nanmean([s2.map_heldout(p, S.tgt(j), m) for m in S.masks], axis=0)
    return W


def main():
    ap = argparse.ArgumentParser()
    s2.add_args(ap)
    a = ap.parse_args()
    D = s2.load(a)
    gen, ver, _ = s2.select_gen(a, D)
    fleet = json.load(open(Path(__file__).resolve().parent.parent / "results" / "fleet.json"))
    rows = fleet["sets"]["reliable_5"]["estimators"]["heldout"]["conditions"]["text"]["rows"]
    tag = max(rows, key=lambda r: r["kappa"])["model"]
    arm, G = gen[tag]
    V = ver[tag]
    S = s2.Sample(D, np.arange(D["n"]))
    H = D["Hint"]                                     # (n, 40, 4)
    Wt, Wp = weights(S, G), weights(S, V)
    se = lambda W: (W * ((L[None, None, :, None] - H[:, :, None, :]) ** 2).mean(-1)).sum(-1)
    parsed = ~np.isnan(G).any(axis=1)
    gain = (se(Wt) - se(Wp)).sum(axis=1)              # (n,)
    med = float(np.median(gain[parsed]))
    El_t, El_p = Wt @ L, Wp @ L
    mean = H.mean(-1)
    present = ((H > 0).sum(-1) >= 3) & (mean >= 2)
    rel = np.zeros(40, bool)
    rel[D["rel"]] = True
    ok = present & rel[None, :] & (np.nan_to_num(G) == 0) & (np.abs(El_p - mean) <= 1.0)
    clean = ok.any(axis=1)
    cand = np.where(parsed & clean)[0]
    i = int(cand[np.argmin(np.abs(gain[cand] - med))])
    print(f"model {tag} ({arm}); parsed {parsed.sum()}; candidates {len(cand)}")
    print(f"median gain {med:.2f}; image {i}: gain {gain[i]:.2f}, "
          f"SE text {se(Wt)[i].sum():.1f}, SE P(yes) {se(Wp)[i].sum():.1f}")
    show = np.where(((H[i] > 0).sum(-1) >= 2) | (np.nan_to_num(G[i]) > 0))[0]
    for j in sorted(show, key=lambda j: -mean[i, j]):
        print(f"{D['short'][j]:<45} text {G[i, j]:.0f} -> {El_t[i, j]:.1f}   "
              f"P(yes) {V[i, j]:.4f} -> {El_p[i, j]:.1f}   experts {list(H[i, j])}")


if __name__ == "__main__":
    main()
