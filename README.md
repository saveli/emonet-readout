# Parsed Text Understates Vision–Language Models — code and predictions

Reproduction material for the paper *Parsed Text Understates Vision–Language Models: From Below
to Above Expert Agreement on Fine-Grained Facial Emotion by Reading Calibrated Probabilities*.
Every number in the paper regenerates from this repository on **CPU alone**, without GPU access
and without re-running any model.

The claim being reproduced: EmoNet-Face-HQ concludes that vision–language models (VLMs) offer
no dependable improvement over random guessing on its 40 facial emotion categories. Scored
against individual experts exactly as experts are scored against each other, eleven open-weight
VLMs average κw = 0.425 when their generated ratings are parsed (below the 0.468 two experts
reach) and 0.553 when the probability of "yes" to one question per category is read instead:
all eleven significantly above expert agreement.

## Quick start

```bash
pip install numpy scipy scikit-learn matplotlib      # the whole dependency list

# The benchmark's expert annotations are not redistributed here (91 MB, not ours to ship):
curl -sL -o data/hq.csv \
  https://raw.githubusercontent.com/LAION-AI/emonet-face/main/data/hq.csv

# The per-image predictions (every model reply, every P(yes)), ~60 MB packed:
for a in predictions/*.tar.xz; do tar -xJf "$a"; done

# The headline: anchor 0.468, parsed text 0.425, P(yes) 0.553 (asserted, not just printed)
python analysis/scoring.py

# Per-model scores, fleet contrast and the single-expert test (results/fleet.json)
python analysis/fleet.py
```

Run everything from the repository root. Each script asserts the paper's headline before it
writes anything and checks the provenance of every result JSON it reads (the code hash and the
hash of every input directory), so a drift in the scoring path or an edited input fails loudly
rather than producing a plausible different number.

## Scoring path

`analysis/scoring.py` is the one estimator every number goes through. Its docstring states the
definitions and every judgment call (J1–J11). In short: each model's per-category scores are
mapped onto the 0–7 scale by a tie-aware rank match onto the pooled marginal of the individual
expert ratings, fitted on one half of the images and applied to the other (two-fold cross-fit,
20 splits); κw is taken against each of the four rating slots and averaged; the expert
reference is the mean of the six pairwise expert κw. Intervals come from an image bootstrap
(B = 500; B = 5000 for the pairwise tests) with the mapping refit inside every replicate.

## What is here

| path | contents |
|---|---|
| `evaluation/verify_eval.py` | P(yes) scoring: one yes/no query per category, the probability read at the first answer token; `--question-set rate` is the graded-digit control |
| `evaluation/e0_prompt_sweep.py` | text parsing: the benchmark's own elicitation (its zero-shot prompt and ours) |
| `evaluation/score_multi_sample.py` | the K = 16 multi-sample control |
| `evaluation/faces_eval.py` | the FACES replication, both conditions |
| `analysis/scoring.py` | the scoring path and the headline assertion |
| `analysis/fleet.py` | per-model κw, both conditions, fleet Δ, tests against the expert reference |
| `analysis/anchor_three_rater.py` | the stricter references: one expert through the mapping, the three-expert panel |
| `analysis/controls_table.py` | the controls table, assembled from the control scripts below |
| `analysis/matched_resolution.py`, `calibration_both_arms.py`, `detection_intensity.py`, `multi_sample_control.py`, `rate_control.py` | the controls |
| `analysis/pairwise_holm.py` | all 55 model pairs, paired tests, Holm and BH |
| `analysis/eif_significance.py` | the comparison against the benchmark's trained model (Empathic-Insight-Face) |
| `analysis/published_baselines.py` | the benchmark's 14 published systems, rescored through the scoring path |
| `analysis/supplementary_tables.py` | phrasings, decoding, resolution, retry loop, training |
| `analysis/tie_rule_sensitivity.py` | every headline under five tie-rule / target settings |
| `analysis/size_vs_score.py` | parameter count against score |
| `analysis/faces_fleet.py` | FACES accuracy, all eleven models, person bootstrap |
| `analysis/make_paper_figures.py`, `permodel_table.py`, `rate_table.py`, `fig1_example.py` | figures and tables, from the JSONs |
| `analysis/build_image_index_map.py` | content-based join between annotation filenames and dataset row indices |
| `predictions/*.tar.xz` | raw per-image predictions, one archive per directory (below) |
| `results/*.json` | the computed results the paper reports, with provenance |
| `results/faces*/` | per-image FACES predictions (paths anonymised to `mountN/<file>`), shipped as `predictions/faces*.tar.xz` |

The archives extract to the directories the scripts read by default:

| directory | contents |
|---|---|
| `results_e0/` | text parsing, eleven models, the benchmark's prompt and ours, 512 px; includes the retry-loop rounds |
| `results_e0_px1024/`, `results_e0_decode/` | text parsing at 1024 px; under the other decoding policy |
| `results_e0_multi/`, `results_e0_multi_scored*/` | K = 16 sampled answers per image, raw and scored (500-image subset and full set) |
| `results_verify_prefill/` | P(yes) scoring, eleven models, with the answer prefix (the reported condition) |
| `results_verify/` | the same without the answer prefix |
| `results_verify_v1/`, `results_verify_v2/` | two alternative question phrasings (named v1 and v2 in the supplementary) |
| `results_verify_px512/` | P(yes) scoring at 512 px |
| `results_rate/` | the graded-digit control (500-image subset) |
| `results_e4/` | the LoRA training runs of the supplementary |

## What is deliberately not here

- **The benchmark's `hq.csv`**: 91 MB of expert annotations, not ours to redistribute. The
  `curl` above fetches it from the authors' repository. It is *not* in the HuggingFace
  dataset repo, which ships only 8 metadata columns and no ratings.
- **FACES images**: research-licensed. Only 72 of the 2,052 are public; the rest require
  registration and a signed Platform Release Agreement from the corpus authors. Predictions
  over them are included here; pixels are not.
- **`data/faces_siglip2_emb.npz`**: SigLIP2 features of those same restricted photographs,
  withheld on the same grounds. Once you have your own FACES copy, regenerate it and the
  transfer result reproduces:

  ```bash
  python evaluation/empathic_insight_faces.py --dump-embeddings data/faces_siglip2_emb.npz
  python analysis/faces_linear_probe.py
  ```
- **Cluster submission scripts**: site-specific Slurm wrappers encoding particular partitions
  and accounts. Useless elsewhere. `multi_sample_control.py` therefore runs with
  `--no-fetch --no-score` here (the shipped `results_e0_multi_scored*/` are its scored inputs).

## Reproducibility notes

- Annotation filenames and dataset row indices are an arbitrary permutation of one another
  (1 of 2,500 coincide by chance). `build_image_index_map.py` recovers the mapping by
  fingerprinting each image on its `(rater, 40-rating)` set, 2,500/2,500 exact;
  `data/hq_image_index_map.json` is its output.
- The mapping onto the 0–7 scale sees expert labels. The primary estimator therefore fits it on
  held-out images; the in-sample (oracle) fit is reported beside it and bounds what label
  access is worth. Both conditions go through the same mapping.
- Confidence intervals resample **images**, the independent unit, since categories share an
  image. On FACES they resample **persons** instead: each person contributes twelve photographs,
  so an image bootstrap there would understate the width.

## License

Two licences, because the code and the numbers need different ones: Creative Commons licences
are not intended for software, and MIT is a poor fit for a table of results.

| what | licence |
|---|---|
| `evaluation/`, `analysis/` (the code) | MIT (`LICENSE`) |
| `results/`, `predictions/`, `data/` (the predictions and computed results) | CC BY 4.0 (`LICENSE-DATA`) |

CC BY 4.0 matches EmoNet-Face-HQ upstream. Two things survive that grant and are spelled out
in `LICENSE-DATA`:

- **Predictions over FACES cannot be sublicensed by us.** That corpus ships under a research
  agreement; its predictions are here so the paper's results can be verified, and anyone
  reusing them needs their own FACES licence from the corpus authors.
- **EmoNet-Face-HQ's use restrictions are inherited.** Its dataset card forbids use in
  workplace or educational emotion recognition, surveillance, law enforcement, border and
  asylum decisions, credit, insurance or hiring. Those are the corpus authors' conditions,
  not conditions of CC BY, and they apply to what is shipped here.
