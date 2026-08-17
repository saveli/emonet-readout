# The Failure Is in the Readout — code and predictions

Reproduction material for the paper. Every number in the paper's tables regenerates from
this repository on **CPU alone**, without GPU access and without re-running any model.

The claim being reproduced: a fine-grained emotion benchmark reports that vision–language
models fail on a 40-category taxonomy. That result is substantially an artifact of how the
models are queried. Reading `P(yes)` from the logits instead of parsing generated text takes
the same checkpoints from 0/10 to 10/10 clearing the human–human agreement anchor, while the
between-model spread collapses inside the measurement precision.

## Quick start

```bash
pip install numpy scipy scikit-learn matplotlib      # that is the whole dependency list

# The benchmark's expert annotations are not redistributed here (91 MB, not ours to ship):
curl -sL -o data/hq.csv \
  https://raw.githubusercontent.com/LAION-AI/emonet-face/main/data/hq.csv

# Per-model kappa_w for one verification arm
python analysis/e1_report.py --results-dir <arm> --hq-csv data/hq.csv

# The confidence intervals behind the spread result
python analysis/verify_bootstrap_ci.py --hq-csv data/hq.csv -B 500

# The FACES replication. The report applies the validity gate itself.
python analysis/generative_validity_check.py --results-dir results/faces
python analysis/faces_e3_report.py --results-dir results/faces
```

## What is here

| path | contents |
|---|---|
| `evaluation/verify_eval.py` | the verification protocol: 40 binary queries, `P(yes)` read from the logits |
| `evaluation/e0_prompt_sweep.py` | the generative arm, i.e. the benchmark's own elicitation |
| `evaluation/faces_eval.py` | the FACES replication, both arms |
| `analysis/generative_validity_check.py` | **the validity gate** (see below) |
| `analysis/verify_bootstrap_ci.py` | image bootstrap; calibration is refit inside each replicate |
| `analysis/build_image_index_map.py` | content-based join between annotation filenames and dataset row indices |
| `analysis/faces_macro_f1.py` | the prior-shift control: macro-F1 with paired person bootstraps |
| `analysis/readout_bootstrap.py` | intervals on the binarisation drop and on the spread collapse |
| `analysis/published_baselines.py` | the benchmark's own 14 published baselines, rescored through this path |
| `analysis/make_paper_figures.py` | regenerates the paper's figures from the JSONs below |
| `data/verify_predictions.npz` | 46 matrices of `(2500 × 40)` `P(yes)`, every EmoNet arm |
| `analysis/noimg_control.py` | the perception floor: both arms with every face replaced by grey |
| `results/faces/` | per-image FACES predictions, 22 arms |
| `results/faces_noimg/` | the same, with the face removed — 6 arms, all constant-response |
| `results/*.json` | the computed tables the paper reports |

## The validity gate

A generative arm can fail without failing loudly. A model that recites the prompt's option
list before being truncated leaves *some* emotion word in its reply, so a parser reports a
high parse rate while the model never reached a conclusion — producing a plausible accuracy
that is really a coin flip resolved by wherever truncation landed.

`analysis/generative_validity_check.py` gates on the one diagnostic that is not a heuristic:

```
accuracy | reply names exactly one category    vs.    accuracy overall
```

A large gap means the parser is producing the score, not the model. On our runs ten of
eleven arms pass with a gap of exactly 0.000; the eleventh shows +0.174 with 98.1% of replies
ending mid-word, and is excluded from the paper's aggregate. **That excluded model's delta is
negative, so the exclusion moves the aggregate toward our hypothesis** — stated here for the
same reason it is stated in the paper.

`faces_e3_report.py` applies the gate itself and prints the gated aggregate, the ungated one,
and which way the exclusion moves the result, so the report and the gate cannot drift apart.
Pass `--gate off` for the ungated numbers.

## What is deliberately not here

- **The benchmark's `hq.csv`** — 91 MB of expert annotations, not ours to redistribute. The
  `curl` above fetches it from the authors' repository. Note it is *not* in the HuggingFace
  dataset repo, which ships only 8 metadata columns and no ratings.
- **FACES images** — research-licensed. Only 72 of the 2,052 are public; the rest require
  registration and a signed Platform Release Agreement from the corpus authors. Predictions
  over them are included here; pixels are not.
- **`data/faces_siglip2_emb.npz`** — SigLIP2 features of those same restricted photographs.
  A derived representation of the pixels, so it is withheld on the same grounds. Once you
  have your own FACES copy, regenerate it and the transfer result reproduces:

  ```bash
  python evaluation/empathic_insight_faces.py --dump-embeddings data/faces_siglip2_emb.npz
  python analysis/faces_linear_probe.py
  ```
- **Cluster submission scripts** — site-specific Slurm wrappers encoding particular
  partitions and accounts. Useless elsewhere.
- **Raw generative JSON** (265 MB) — the scores survive in `results/`.
- **The fine-tuning branch** — a preference-training experiment that produced a null result
  and that the paper does not use.

## Reproducibility notes

- `data/verify_predictions.npz` stores image **indices** alongside each matrix rather than
  assuming they are contiguous, so a partial arm round-trips honestly.
- Annotation filenames and dataset row indices are an arbitrary permutation of one another
  (1 of 2500 coincide by chance). `build_image_index_map.py` recovers the mapping by
  fingerprinting each image on its `(rater, 40-rating)` set — 2500/2500 exact.
- Quantile calibration uses the gold marginal and is therefore an **oracle**: reported
  figures are an upper bound, not a deployable procedure. The held-out control is in
  `results/verify_prefill_heldout_calibration.json` (maximum leakage +0.025 κw, two
  negative). This applies equally to both elicitations, so it does not affect the contrast.
- Confidence intervals resample **images**, the independent unit, since categories share an
  image and rater counts vary. On FACES they resample **persons** instead: each person
  contributes twelve photographs (six emotions, two sets), so an image bootstrap there would
  understate the width. `faces_macro_f1.py` shares one person draw across models, which is
  what makes its fleet interval a CI on the mean rather than a recombination of per-model
  intervals.

## License

Code is released under the MIT License (`LICENSE`). Predictions are derived works of the
respective corpora and inherit their terms; consult the original dataset licenses before
redistributing.
