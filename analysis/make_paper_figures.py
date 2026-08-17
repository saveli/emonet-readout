#!/usr/bin/env python3
"""Generate the WACV 2027 paper figures from the analysis JSONs.

Produces (vector PDF, no rasterisation) into ``paper/figs/``:

  fig_forest.pdf   paired forest of kappa_w under generative vs. verification
                   elicitation, with the human-human anchor, plus a companion
                   panel contrasting between-model spread with measurement
                   precision.
  fig_readout.pdf  the binarisation ablation: continuous P(yes) vs. the same
                   answers read as text, against the generative range.
  fig_faces.pdf    FACES replication deltas with paired person-bootstrap CIs.

Every number is read from JSON; nothing about the results is hardcoded here.
Models are ordered ALPHABETICALLY in every figure -- the instability of the
ranking is one of the paper's findings, so sorting by score would contradict it.

Usage:
    conda activate vllm-emotion-eval
    python analysis/make_paper_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

GEN_CI = ANALYSIS / "generative_bootstrap_ci.json"
VER_CI = ANALYSIS / "verify_prefill_bootstrap_ci_matched10.json"
READOUT = ANALYSIS / "readout_and_taxonomy_ablations.json"
FACES = ANALYSIS / "faces_e3_report.json"

# Model excluded from the FACES headline by the generative-validity gate.
# It is drawn, greyed and set apart, rather than dropped: its delta is
# negative, so hiding it would flatter the hypothesis.
FACES_GATED_OUT = "MiMo-VL-7B-RL-2508"

# --------------------------------------------------------------------------
# print style -- Times-like serif (STIX) to match the WACV body font,
# small type that survives a single column, thin recessive rules.
# --------------------------------------------------------------------------
plt.rcParams.update(
    {
        "pdf.fonttype": 42,          # embed TrueType, keep text as text
        "ps.fonttype": 42,
        "pdf.compression": 6,
        "svg.fonttype": "none",
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "standard",   # exact canvas: see W_COL / W_FULL below
        "savefig.pad_inches": 0.0,
        "image.composite_image": False,
    }
)

# WACV geometry: \textwidth 6.875in, \columnsep 0.3125in.  Figures are drawn at
# exactly the width they will occupy, so \includegraphics[width=\linewidth]
# scales them by 1.0 and the point sizes above are the point sizes on paper.
W_FULL = 6.875
W_COL = (6.875 - 0.3125) / 2

# Okabe-Ito, chosen for colour-vision deficiency AND for separating in
# greyscale (blue ~ 0.34 luminance, vermillion ~ 0.47, grey ~ 0.55).
# Every series is ALSO distinguished by marker shape and line style, so no
# figure depends on colour alone.
C_GEN = "#D55E00"    # generative elicitation (vermillion)
C_VER = "#0072B2"    # verification (blue)
C_NULL = "#7F7F7F"   # not distinguishable from zero
C_DEAD = "#AFAFAF"   # excluded / background
INK = "#000000"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def load(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


DISPLAY = {
    "InternVL3_5-8B-HF": "InternVL3_5-8B",
    "MiMo-VL-7B-RL-2508": "MiMo-VL-7B-RL",
    "Ministral-3-14B-Instruct-2512": "Ministral-3-14B",
    "Qwen2.5-VL-3B-Instruct": "Qwen2.5-VL-3B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
}


def clean_arm(arm: str) -> str:
    """Strip the provenance suffix, e.g. 'GLM-4.6V-Flash [paper_once]'."""
    return re.sub(r"\s*\[[^\]]*\]\s*$", "", arm).strip()


def display_name(arm: str) -> str:
    return DISPLAY.get(clean_arm(arm), clean_arm(arm))


def natkey(name: str):
    """Alphabetical with numeric runs compared numerically, case-sensitive
    (upper-case first) -- reproduces the ordering used in the paper's tables."""
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


def by_model(rows, key="arm"):
    """{display name: row}, keyed on the cleaned model name."""
    return {display_name(r[key]): r for r in rows}


def thin_grid(ax, axis="x"):
    ax.grid(axis=axis, color="0.85", linestyle="-", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def row_bands(ax, n, color="0.94"):
    """Faint alternating bands so long label-to-mark runs stay trackable."""
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=color, lw=0, zorder=-5)


# --------------------------------------------------------------------------
# Figure 1 -- the paired forest plot (main figure, spans both columns)
# --------------------------------------------------------------------------
def fig_forest() -> None:
    gen = load(GEN_CI)
    ver = load(VER_CI)
    anchor = load(READOUT)["anchors"]["reliable_5"]

    g = by_model(gen["rows"])
    v = by_model(ver["rows"])
    models = sorted(set(g) & set(v), key=natkey)
    assert len(models) == len(g) == len(v), "generative/verification arms do not match"

    n_above_g = sum(g[m]["ci_lo"] > anchor for m in models)
    n_above_v = sum(v[m]["ci_lo"] > anchor for m in models)

    fig = plt.figure(figsize=(W_FULL, 3.05), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.04, hspace=0.0)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.15, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # ---- panel (a): the forest -------------------------------------------
    row_bands(ax, len(models))
    off = 0.19
    for i, m in enumerate(models):
        for row, colour, marker, ls, fill in (
            (g[m], C_GEN, "o", (0, (2.4, 1.2)), "white"),
            (v[m], C_VER, "s", "solid", C_VER),
        ):
            y = i - off if row is g[m] else i + off
            ax.plot(
                [row["ci_lo"], row["ci_hi"]], [y, y],
                color=colour, lw=1.1, ls=ls, solid_capstyle="butt", zorder=3,
            )
            for xcap in (row["ci_lo"], row["ci_hi"]):
                ax.plot([xcap, xcap], [y - 0.11, y + 0.11],
                        color=colour, lw=1.1, zorder=3)
            ax.plot(row["kappa"], y, marker=marker, ms=4.2, color=colour,
                    mfc=fill, mew=1.0, zorder=4, clip_on=False)

    ax.axvline(anchor, color=INK, lw=0.9, ls=(0, (5, 2)), zorder=2)
    ax.text(
        anchor, len(models) - 0.13,
        f"human–human anchor  $\\kappa_w = {anchor:.3f}$",
        ha="center", va="center", fontsize=6.8, color=INK, zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
    )

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_ylim(len(models) + 0.25, -0.5)         # alphabetical, top to bottom
    ax.set_xlabel("quadratic-weighted $\\kappa$  (5 reliably measurable categories)")
    thin_grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        Line2D([], [], color=C_GEN, ls=(0, (2.4, 1.2)), lw=1.1, marker="o",
               ms=4.2, mfc="white", mew=1.0,
               label=f"generative elicitation ({n_above_g}/{len(models)} CIs above anchor)"),
        Line2D([], [], color=C_VER, ls="solid", lw=1.1, marker="s",
               ms=4.2, mfc=C_VER,
               label=f"verification, $P(\\mathrm{{yes}})$ ({n_above_v}/{len(models)} CIs above anchor)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.145),
              ncol=2, columnspacing=1.8)
    ax.set_title("(a)", loc="left", x=0.0, y=1.01, fontsize=8)

    # ---- panel (b): spread against its own precision ---------------------
    labels = ["generative", "verification"]
    sd = [gen["between_model_sd"], ver["between_model_sd"]]
    wd = [gen["median_ci_width"], ver["median_ci_width"]]
    colours = [C_GEN, C_VER]
    xs = [0, 1]
    w = 0.34
    for k, x in enumerate(xs):
        axb.bar(x - w / 2, sd[k], width=w, color=colours[k], lw=0.6,
                edgecolor=colours[k], zorder=3)
        axb.bar(x + w / 2, wd[k], width=w, facecolor="white", lw=0.6,
                edgecolor=colours[k], hatch="////", zorder=3)
        axb.text(x - w / 2, sd[k] + 0.0015, f"{sd[k]:.3f}", ha="center",
                 va="bottom", fontsize=6.5)
        axb.text(x + w / 2, wd[k] + 0.0015, f"{wd[k]:.3f}", ha="center",
                 va="bottom", fontsize=6.5)

    axb.set_xticks(xs)
    axb.set_xticklabels(labels)
    axb.set_ylabel("$\\kappa$ units")
    axb.set_ylim(0, max(sd + wd) * 1.34)
    thin_grid(axb, "y")
    axb.legend(
        handles=[
            Patch(facecolor="0.35", edgecolor="0.35", label="between-model sd"),
            Patch(facecolor="white", edgecolor="0.35", hatch="////",
                  label="median 95% CI width"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=1,
        handlelength=1.4, handleheight=0.9, labelspacing=0.25,
    )
    axb.set_title("(b)", loc="left", x=0.0, y=1.01, fontsize=8)

    out = FIGS / "fig_forest.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  ({n_above_g}/{len(models)} gen, {n_above_v}/{len(models)} ver above {anchor:.3f})")


# --------------------------------------------------------------------------
# Figure 2 -- the readout ablation
# --------------------------------------------------------------------------
def fig_readout() -> None:
    ab = load(READOUT)
    anchor = ab["anchors"]["reliable_5"]
    rows = [r for r in ab["rows"] if not r["arm"].startswith("baseline")]
    gen_rows = load(GEN_CI)["rows"]

    cont = {display_name(r["arm"]): r["kappa_continuous"] for r in rows}
    binz = {display_name(r["arm"]): r["kappa_binarised"] for r in rows}
    gene = {display_name(r["arm"]): r["kappa"] for r in gen_rows}
    mean_delta = sum(r["readout_delta"] for r in rows) / len(rows)
    n_lose = sum(r["readout_delta"] < 0 for r in rows)

    y_cont, y_bin, y_gen = 2.0, 1.0, 0.0
    allv = list(cont.values()) + list(binz.values()) + list(gene.values())
    x_lo, x_hi = min(allv) - 0.030, max(allv) + 0.045   # room for the range text
    fig, ax = plt.subplots(figsize=(W_COL, 2.15), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)

    # generative range as a band behind everything: the reference the
    # binarised answers fall back into.
    glo, ghi = min(gene.values()), max(gene.values())
    ax.axvspan(glo, ghi, color="0.90", lw=0, zorder=0)

    # connectors: one per model, continuous -> binarised
    for m in cont:
        ax.plot([cont[m], binz[m]], [y_cont, y_bin], color="0.62", lw=0.5,
                zorder=2)

    def draw_row(vals, y, colour, marker, fill, label, y_text):
        vals = list(vals)
        lo, hi = min(vals), max(vals)
        ax.plot([lo, hi], [y, y], color=colour, lw=1.2, zorder=3)
        for xcap in (lo, hi):
            ax.plot([xcap, xcap], [y - 0.075, y + 0.075], color=colour,
                    lw=1.2, zorder=3)
        ax.plot(vals, [y] * len(vals), linestyle="none", marker=marker,
                ms=4.0, mec=colour, mfc=fill, mew=0.9, zorder=4)
        ax.text(x_lo + 0.004, y_text, label, ha="left", va="center",
                fontsize=7.2)
        ax.text(x_hi - 0.004, y_text, f"{lo:.3f}–{hi:.3f}", ha="right",
                va="center", fontsize=6.5, color="0.30")

    draw_row(cont.values(), y_cont, C_VER, "s", C_VER,
             "continuous $P(\\mathrm{yes})$", y_cont + 0.26)
    draw_row(binz.values(), y_bin, C_VER, "s", "white",
             "the same answers, binarised", y_bin - 0.28)
    draw_row(gene.values(), y_gen, C_GEN, "o", "white",
             "generative elicitation", y_gen - 0.28)

    ax.text(x_lo + 0.004, y_bin - 0.52,
            f"mean $\\Delta = {mean_delta:+.3f}$; all {n_lose}/{len(cont)} models lose",
            ha="left", va="center", fontsize=6.5, color="0.30")

    ax.axvline(anchor, color=INK, lw=0.9, ls=(0, (5, 2)), zorder=5)
    ax.text(anchor, y_gen - 0.68, f"human anchor {anchor:.3f}", ha="center",
            va="center", fontsize=6.5, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    ax.text((glo + ghi) / 2, y_cont + 0.72, "generative range", ha="center",
            va="center", fontsize=6.5, color="0.40")

    ax.set_yticks([])
    ax.set_ylim(y_gen - 0.95, y_cont + 0.90)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("quadratic-weighted $\\kappa$")
    thin_grid(ax, "x")
    ax.spines["left"].set_visible(False)

    out = FIGS / "fig_readout.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  (mean delta {mean_delta:+.4f}, {n_lose}/{len(cont)} lose)")


# --------------------------------------------------------------------------
# Figure 3 -- FACES replication deltas
# --------------------------------------------------------------------------
def fig_faces() -> None:
    rep = load(FACES)
    kept = [r for r in rep["rows"] if r["model"] != FACES_GATED_OUT]
    gated = [r for r in rep["rows"] if r["model"] == FACES_GATED_OUT]
    kept.sort(key=lambda r: natkey(display_name(r["model"])))

    ordered = kept + gated          # gated model sits below a separator
    n_keep = len(kept)
    mean_delta = sum(r["delta"] for r in kept) / n_keep
    n_gain = sum(r["delta"] > 0 and r["paired_ci"]["excludes_zero"] for r in kept)
    n_null = sum(not r["paired_ci"]["excludes_zero"] for r in kept)
    n_rev = sum(r["delta"] < 0 and r["paired_ci"]["excludes_zero"] for r in kept)

    fig, ax = plt.subplots(figsize=(W_COL, 2.85), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
    row_bands(ax, len(ordered))

    for i, r in enumerate(ordered):
        ci = r["paired_ci"]
        excluded = r["model"] == FACES_GATED_OUT
        if excluded:
            colour, marker, fill = C_DEAD, "D", C_DEAD
        elif not ci["excludes_zero"]:
            colour, marker, fill = C_NULL, "o", "white"
        elif r["delta"] < 0:
            colour, marker, fill = C_GEN, "v", C_GEN
        else:
            colour, marker, fill = C_VER, "s", C_VER
        ax.plot([ci["ci_lo"], ci["ci_hi"]], [i, i], color=colour, lw=1.1,
                zorder=3)
        for xcap in (ci["ci_lo"], ci["ci_hi"]):
            ax.plot([xcap, xcap], [i - 0.16, i + 0.16], color=colour, lw=1.1,
                    zorder=3)
        ax.plot(r["delta"], i, marker=marker, ms=4.2, color=colour, mfc=fill,
                mew=1.0, zorder=4)

    ax.axvline(0.0, color=INK, lw=0.9, zorder=2)
    if gated:
        ax.axhline(n_keep - 0.5, color="0.55", lw=0.5, ls=(0, (2, 2)), zorder=2)

    labels = [display_name(r["model"]) for r in ordered]
    if gated:
        labels[-1] = labels[-1] + "$^{\\dagger}$"
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels)
    for tick, r in zip(ax.get_yticklabels(), ordered):
        if r["model"] == FACES_GATED_OUT:
            tick.set_color("0.45")
    ax.set_ylim(len(ordered) - 0.5, -0.5)
    ax.set_xlabel("$\\Delta$ six-way accuracy  (verification $-$ generative)")
    thin_grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [
        Line2D([], [], color=C_VER, lw=1.1, marker="s", ms=4.2, mfc=C_VER,
               label=f"gain, CI excludes 0 ({n_gain})"),
        Line2D([], [], color=C_NULL, lw=1.1, marker="o", ms=4.2, mfc="white",
               label=f"CI spans 0 ({n_null})"),
        Line2D([], [], color=C_GEN, lw=1.1, marker="v", ms=4.2, mfc=C_GEN,
               label=f"reversal, CI excludes 0 ({n_rev})"),
    ]
    if gated:
        handles.append(
            Line2D([], [], color=C_DEAD, lw=1.1, marker="D", ms=4.0,
                   mfc=C_DEAD, label="$^{\\dagger}$excluded by validity gate")
        )
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.46, 1.0),
              ncol=2, columnspacing=0.9, handlelength=1.6, labelspacing=0.25)

    out = FIGS / "fig_faces.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  ({n_gain} gains, {n_null} nulls, {n_rev} reversal; "
          f"mean delta over {n_keep} models {mean_delta:+.4f})")


if __name__ == "__main__":
    fig_forest()
    fig_readout()
    fig_faces()
