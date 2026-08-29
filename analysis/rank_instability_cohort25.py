#!/usr/bin/env python3
"""Rank instability over the cohort the paper actually reports.

The published figure in sec:app-rank is computed over the benchmark's fourteen
published baselines plus the eleven VLMs of tab:models, each at the generative arm
we report (the better of the benchmark's prompt and ours, per contrast_as_reported).

Two steps, because the first is slow and needs hq.csv:

  python analysis/reliability_vs_performance.py \
      --hq-csv data/hq.csv --results-dir results_e0 \
      --out analysis/reliability_vs_performance_cohort25
  python analysis/rank_instability_cohort25.py

results_e0 holds three arms per model, so the first step yields 45 rows. This
script keeps the 14 published rows plus the one reported arm per model, giving 25.
"""
import json, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = json.load(open(ROOT / "results/reliability_vs_performance_cohort25.json"))["models"]
reported = {r["model"]: r["gen_arm"]
            for r in json.load(open(ROOT / "results/contrast_as_reported.json"))["rows"]}

keep = []
for r in rows:
    name = r["model"]
    if name.startswith("[paper]"):
        keep.append(r)
        continue
    m = re.match(r"\[ours\] (.+) \[(.+)\]$", name)
    if m and reported.get(m.group(1)) == m.group(2):
        keep.append(r)

by_rel = [r["model"] for r in sorted(keep, key=lambda r: -r["perf_reliable"])]
by_all = [r["model"] for r in sorted(keep, key=lambda r: -r["perf_all"])]
moved = sum(1 for i, m in enumerate(by_rel) if by_all.index(m) != i)
worst = max(abs(by_all.index(m) - i) for i, m in enumerate(by_rel))
rhos = [r["rho_alpha_vs_perf"] for r in keep]

out = {"n_systems": len(keep), "n_published": sum(1 for r in keep if r["model"].startswith("[paper]")),
       "n_ours": sum(1 for r in keep if r["model"].startswith("[ours]")),
       "rank_changes": moved, "max_places_moved": worst,
       "rho_median": statistics.median(rhos), "rho_min": min(rhos), "rho_max": max(rhos)}
json.dump(out, open(ROOT / "results/rank_instability_cohort25.json", "w"), indent=1)
print(f"{out['n_systems']} systems ({out['n_published']} published, {out['n_ours']} ours): "
      f"{moved} change rank, by up to {worst} places; "
      f"rho median {out['rho_median']:.2f}, range {out['rho_min']:.2f} to {out['rho_max']:.2f}")
