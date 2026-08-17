#!/usr/bin/env python
"""E3 -- does the generative-vs-verification gap reproduce on REAL faces?

Everything the paper rests on was measured on EmoNet-Face-HQ, which is synthetic and whose
demographic labels are the generation prompt. If the elicitation effect is a property of
how models are asked, it must show up on photographs too; if it only appears on synthetic
faces, it is a fact about EmoNet and the paper is much smaller.

FACES (Ebner, Riediger & Lindenberger 2010) is the replication set: 2,052 photographs, 171
people, six posed expressions x two sets, exactly balanced at 342 images per expression,
with age and gender per image. `data/FACES`, filename
`<person>_<age>_<gender>_<emotion>_<set>.jpg`. That path is readable from the compute nodes
(it is the login node that has no `$DATASETS`); the submitter defaults to a pre-shrunk
copy only to avoid decoding a 10 MP JPEG once per image per arm.

**This is a CATEGORICAL replication and must be reported as one.** FACES ships one forced-
choice gold label per image, not the per-image multi-rater 0-7 ratings EmoNet has, so there
is no human-human anchor to clear and no quantile calibration to apply. The metric here is
accuracy over six classes (chance 1/6), and the question is only whether the *contrast*
between elicitations survives on real photographs. The graded claim stays with EmoNet.

Both arms answer over the same six-way space so the comparison is about elicitation alone:

  verify      six binary "Does this face express X?" queries, prediction = argmax P(yes).
              Identical machinery to verify_eval.py, six emotions instead of forty.
  generative  one open question, prediction parsed from the sampled text.

Per §S the interesting split is readout vs question format, so the generative arm is the
control that carries *both* differences, and `--arm verify --binarise` is available to
isolate the readout on this dataset too.

Usage:
    python evaluation/faces_eval.py --model MODEL_ID --arm verify --tag gemma-3-4b-it
    python evaluation/faces_eval.py --model MODEL_ID --arm generative --tag gemma-3-4b-it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_eval import (  # noqa: E402
    answer_token_ids,
    build_model,
    encode,
    scores_cached,
    uncached,
)
from e0_prompt_sweep import VramTracker  # noqa: E402

# FACES codes its expressions in the filename. `n` is neutral, which EmoNet has no category
# for -- that mismatch is a finding about the taxonomy (§D) and is kept here rather than
# dropped, because a six-way forced choice that silently omits the neutral option would
# inflate accuracy on the other five.
EMOTIONS = {"a": "anger", "d": "disgust", "f": "fear",
            "h": "happiness", "n": "a neutral expression", "s": "sadness"}
AGES = {"y": "young", "m": "middle", "o": "old"}
GENDERS = {"f": "female", "m": "male"}

VERIFY_PROMPT = "Does this face express {emotion}? Answer yes or no."
GENERATIVE_PROMPT = (
    "Which emotion does this face express? Answer with exactly one of: "
    "anger, disgust, fear, happiness, neutral, sadness."
)


def scan_faces(root: Path):
    """Parse the filename metadata. Returns a list of dicts, sorted for reproducibility."""
    out = []
    for p in sorted(root.glob("*.jpg")):
        parts = p.stem.split("_")
        if len(parts) != 5:
            continue                      # Thumbs.db and anything else malformed
        person, age, gender, emo, half = parts
        if emo not in EMOTIONS or age not in AGES or gender not in GENDERS:
            continue
        out.append({"path": str(p), "person": person, "age": AGES[age],
                    "gender": GENDERS[gender], "gold": emo, "set": half})
    return out


def parse_generative(text: str) -> str | None:
    """Map free text onto one of the six codes, or None if it names no class or several.

    Matching is whole-word and the first *distinct* class wins only if no other class is
    also named -- a reply like "not anger but sadness" must not score as anger just because
    the word appears first. Returning None keeps unparseable replies visible in the output
    instead of silently becoming errors attributed to the model's perception.
    """
    words = {"anger": "a", "angry": "a", "disgust": "d", "disgusted": "d",
             "fear": "f", "fearful": "f", "afraid": "f", "scared": "f",
             "happiness": "h", "happy": "h", "joy": "h",
             "neutral": "n", "neutrality": "n",
             "sadness": "s", "sad": "s"}
    low = text.lower()
    found = [(m.start(), code) for word, code in words.items()
             for m in re.finditer(rf"\b{word}\b", low)]
    if not found:
        return None
    distinct = {c for _, c in found}
    if len(distinct) == 1:
        return next(iter(distinct))
    # Several classes named. A reasoning model weighs options before committing ("could be
    # anger, but the brow suggests sadness"), so the LAST mention is the answer -- the same
    # reading a human would take. Returning None here instead would score every reasoning
    # model at zero and inflate the generative-vs-verification gap in exactly the direction
    # this study hopes to find, which is the one bias worth being paranoid about.
    #
    # Known weakness: "clearly disgust, not fear" resolves to fear, because last-mention
    # cannot see negation. Only reasoning models produce multi-mention replies at all (the
    # four gemma/Qwen-VL runs parse 2052/2052 with a single mention), so treat the
    # generative score of a reasoning model as carrying parser uncertainty the others do
    # not, and check `unparsed` plus a sample of raw_response before quoting it.
    return max(found)[1]


def resolve_thinking(args):
    """CLI string -> the bool|None `apply_template` expects.

    Passing the raw string through was a silent inversion: a non-empty string is truthy in
    Jinja, so every template that accepts `enable_thinking` rendered "off" byte-identically
    to True and reasoned anyway. Measured on four of eleven models -- Qwen3.5, gemma-4-12B,
    GLM-4.6V, MiniCPM; the other seven reject the kwarg and were unaffected.
    `verify_eval.py` and `e0_prompt_sweep.py` already did this, so the bug was FACES-only.

    A module-level helper rather than a local in `main`, because `_run` and `write` are
    module-level and take `args` -- a local would be invisible to both.
    """
    return None if args.thinking == "auto" else (args.thinking == "on")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --model rather than positional, matching verify_eval.py so run_faces_job.sh can be a
    # copy of run_verify_job.sh with one path changed.
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=["verify", "generative"], default="verify")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--faces-dir", type=Path, default=Path("data/FACES"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_faces"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--merge-adapter", action="store_true")
    ap.add_argument("--ablate-image", choices=["off", "grey", "shuffle"], default="off",
                    help="perception control. 'grey' replaces every face with a uniform grey "
                         "image of the same size, so the prompt, the token count and the read "
                         "position are unchanged and only the pixels carry no signal. "
                         "'shuffle' pairs each prompt with another image's pixels, which "
                         "keeps the image statistics and destroys only the correspondence. "
                         "A protocol that scores well here is scoring its own priors, and "
                         "no accuracy from either arm means anything without this floor.")
    ap.add_argument("--thinking", choices=["off", "on", "auto"], default="off")
    ap.add_argument("--answer-prefill", default="Answer:")
    # 12 was set assuming the reply is one word. It is not, for models that open with a
    # reasoning preamble -- Qwen3.5 produced "The user wants me to identify the emotion
    # shown in the face" on all 2052 images and got truncated before naming anything,
    # scoring 0.000 with 2052/2052 unparsed. That is a measurement artifact, and it inflates
    # the generative-vs-verification gap in the direction this study wants, so it is the
    # kind of bug that would have survived review. `--thinking off` does not prevent it
    # (section L: the flag closes a think block, it does not stop a preamble).
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="generative arm only. Must be long enough for a model that "
                         "reasons before answering; parse_generative takes the last named "
                         "class when several appear.")
    ap.add_argument("--min-yesno-mass", type=float, default=0.5,
                    help="verify arm only; abort after the first image if the model is not "
                         "answering yes/no at all (the §L failure)")
    # The uncached path re-encodes the image once per emotion -- six vision-tower passes per
    # image at --max-rows 1, where the cached path needs one. The saving is proportionally
    # LARGER here than in verify_eval, whose 40-row batch already amortises the image.
    #
    # Default is ON. The cache is not a new source of error: section F measured left-padded,
    # right-padded and one-at-a-time decoding disagreeing by 3.1e-2 in P(yes) *equally in
    # every pairing* -- bf16 logit quantisation interacting with batch composition, present
    # in the uncached path too. Every cached-vs-uncached delta measured here (1.3e-2 to
    # 2.8e-2) sits below that floor, so "uncached" is not a more faithful number, only a
    # slower one.
    #
    # What IS real, and belongs to the metric rather than the cache: this arm scores `argmax`
    # over six P(yes) values, and on the E3 runs 16-63% of labels are decided by a top-2 gap
    # smaller than 3.1e-2 (gemma-3-4b 63%, median gap 0.001, its section P saturation showing
    # through). Those labels can flip under ANY change of batch composition, cache or not.
    # The self-check still requires argmax agreement, because a cache that flips labels on
    # the check images is worth knowing about even when P(yes) agrees.
    ap.add_argument("--prefix-cache", choices=["auto", "off"], default="auto",
                    help="verify arm only. 'auto' enables the shared-prefix cache after a "
                         "self-check against the uncached path (both |dP(yes)| and argmax "
                         "agreement); 'off' forces the reference path.")
    ap.add_argument("--cache-tolerance", type=float, default=0.05,
                    help="FLOOR for the acceptance threshold. The actual threshold is "
                         "max(this, ambient * --cache-ambient-mult), where ambient is "
                         "measured per check image by re-running the uncached path at a "
                         "different batch width. An absolute constant does not transfer "
                         "between datasets -- 0.05 came from EmoNet and wrongly rejected "
                         "the cache on FACES.")
    ap.add_argument("--cache-ambient-mult", type=float, default=1.0,
                    help="how much of the measured ambient disagreement the cache is "
                         "allowed. 1.0 = the cache must be no worse than the uncached path "
                         "disagreeing with itself.")
    ap.add_argument("--cache-check-images", type=int, default=8,
                    help="images to A/B before trusting the cache; argmax must agree on all")
    # FACES photographs are 4:5 portrait (2835x3543) where EmoNet renders are square, and
    # they cost far more VRAM per row than the EmoNet sweep did. Measured on an A40 (44 GB),
    # verify arm, whole GPU, no contention:
    #
    #   gemma-3-4b    640^2, rows=2  -> fits, 2.30 img/s
    #   gemma-3-12b   640^2, rows=2  -> OOM on the FIRST image
    #   gemma-3-12b  1024^2, rows=1  -> fits, 1.37 img/s, peak 40.7 GB of 44
    #
    # so rows is the knob that moves with model size, not the pixel cap. I guessed gemma-3's
    # pan-and-scan cropping as the mechanism and never confirmed it -- the numbers above are
    # what is actually known.
    #
    # The cap is 1024^2 to match the EmoNet sweep's pixel budget. Dropping it to 640^2 would
    # be the cheaper fix for the OOM and is the wrong one: resolution could plausibly
    # interact with elicitation (a low-res image might hurt one arm more than the other),
    # which would confound the very contrast this script exists to measure.
    #
    # **max-image-pixels must be identical across every model in a sweep** -- it changes what
    # the model sees and therefore the score. max-rows does not: it only sets how many
    # independent queries share a forward pass, and `uncached` left-pads and reads the last
    # position, so batching is numerically inert. Vary rows per model, never the cap.
    ap.add_argument("--max-image-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--max-rows", type=int, default=1,
                    help="verify arm: emotion queries per forward pass. Each row carries "
                         "its own copy of the image, so this is the VRAM knob. Numerically "
                         "inert -- raise it for small models to go faster.")
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--resume", choices=["auto", "off"], default="auto",
                    help="auto: continue from <tag>__faces_<arm>.json.partial, skipping "
                         "images already scored. Matched on image path, not position.")
    ap.add_argument("--vram-interval", type=float, default=5.0)
    args = ap.parse_args()


    import torch
    from PIL import Image

    from e0_prompt_sweep import shrink

    tag = args.tag or args.model.split("/")[-1]
    items = scan_faces(args.faces_dir)
    if not items:
        raise SystemExit(f"no FACES images under {args.faces_dir}")
    if args.limit:
        items = items[:args.limit]
    print(f"[faces] {len(items)} images, arm={args.arm}, model={args.model}", flush=True)

    proc, model = build_model(args)
    tokenizer = getattr(proc, "tokenizer", proc)
    yes_ids, no_ids = answer_token_ids(tokenizer)
    codes = list(EMOTIONS)

    vram = VramTracker(interval=args.vram_interval)
    vram.start()
    started = time.time()
    out_path = args.out_dir / f"{tag}__faces_{args.arm}.json"
    # Checkpoints go to `.partial` and are renamed on completion, matching verify_eval.py.
    # Writing progress straight to the final name would make a killed job indistinguishable
    # from a finished one -- and faces_submit.sh skips any model whose final file exists, so
    # a truncated run would silently become the reported result.
    ckpt_path = out_path.with_suffix(".json.partial")
    results = []
    # Resume, matching verify_eval.py. Added after Qwen3.5's generative arm reached
    # 1800/2052 in three hours and was killed by the wall clock, discarding all of it --
    # the reasoning models run at ~0.17 img/s under the 128-token budget (§Z1), so a
    # restart-from-scratch policy makes them nearly unrunnable on a 6 h partition.
    #
    # Keyed on image BASENAME rather than position or full path. Position is stable today
    # because `scan_faces` sorts, but a checkpoint that silently mismatches its item list is
    # the kind of bug that produces a plausible file rather than an error.
    #
    # The full path was the key until 2026-08-17 and that was the worst variant of the three
    # path-join bugs found that day. The two clusters mount FACES at different absolute
    # paths, so a checkpoint that travels -- copied between mounts, or simply produced with a
    # --faces-dir pointing at the other mount of the same images -- matches nothing. And
    # because `results.extend(prev)` runs BEFORE the filter, a total mismatch does not lose
    # the checkpoint, it DUPLICATES it: every image is rescored, appended to the 2052 rows
    # already loaded, and the file ends with 4104 rows whose accuracy is a mean over
    # duplicates. Nothing downstream would flag that. The basename is the join key
    # everywhere else in this project for exactly this reason.
    if args.resume != "off" and ckpt_path.is_file():
        try:
            prev = json.load(open(ckpt_path))["results"]
            done = {Path(r["path"]).name for r in prev}
            results.extend(prev)
            items = [it for it in items if Path(it["path"]).name not in done]
            print(f"[faces] RESUMED from {ckpt_path.name}: {len(prev)} scored, "
                  f"{len(items)} to go", flush=True)
            if args.arm == "verify":
                print("[faces] NOTE: the yes/no mass average covers only images scored from "
                      "here on; the guard is weaker on a resumed run.", flush=True)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"[faces] checkpoint unreadable ({type(exc).__name__}), starting over",
                  flush=True)
            results = []

    # inference_mode, not bare forwards. Two reasons, one of which cost a smoke test:
    # without it the cache tensors are non-leaf and `expand_cache`'s deepcopy raises
    # "Only Tensors created explicitly by the user (graph leaves) support the deepcopy
    # protocol"; and every forward here was building an autograd graph nothing consumes,
    # which is part of why this arm's VRAM peak ran above verify_eval's on the same model
    # (gemma-3-12b 40.7 GB vs 30.4). verify_eval has always wrapped its loop this way.
    with torch.inference_mode():
        return _run(args, items, proc, model, tokenizer, yes_ids, no_ids, codes, results,
                    ckpt_path, out_path, started, vram)


def _run(args, items, proc, model, tokenizer, yes_ids, no_ids, codes, results,
         ckpt_path, out_path, started, vram):
    """The scoring loop. Split out only so the caller can wrap it in inference_mode."""
    think = resolve_thinking(args)
    import torch
    from PIL import Image

    from e0_prompt_sweep import shrink

    mass_total, mass_count = 0.0, 0
    tag = args.tag or args.model.split("/")[-1]
    use_cache = args.arm == "verify" and args.prefix_cache != "off"
    cache_checked = 0
    # Accumulators for the cache verdict, decided over the whole check window rather than
    # per image -- see the comment at the verdict itself.
    cache_deltas, amb_deltas, cache_agree, amb_agree = [], [], [], []

    for i, it in enumerate(items):
        img = shrink(Image.open(it["path"]).convert("RGB"), args.max_image_pixels)
        if args.ablate_image == "grey":
            img = Image.new("RGB", img.size, (128, 128, 128))
        elif args.ablate_image == "shuffle":
            # Deterministic derangement by index, so the control is reproducible and no
            # image is ever paired with itself.
            alt = items[(i + len(items) // 2) % len(items)]
            img = shrink(Image.open(alt["path"]).convert("RGB"), args.max_image_pixels)

        if args.arm == "verify":
            msgs = [[{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": VERIFY_PROMPT.format(emotion=EMOTIONS[c])}]}]
                for c in codes]
            p_yes = mass = None
            if use_cache:
                # Same construction as verify_eval: right-pad (passed to the call, since
                # some processors ignore the tokenizer attribute) to find where the six
                # rows diverge, then forward the shared image prefix once.
                tokenizer.padding_side = "right"
                enc = encode(proc, msgs, think, args.answer_prefill,
                             add_generation_prompt=True, tokenize=True, return_dict=True,
                             return_tensors="pt", padding=True, padding_side="right")
                tokenizer.padding_side = "left"
                enc = {k: (v.to(model.device)
                           if torch.is_tensor(v) and k != "pixel_values" else v)
                       for k, v in enc.items()}
                cids = enc["input_ids"]
                same = (cids == cids[0:1]).all(dim=0).tolist()
                plen = same.index(False) if False in same else cids.shape[1]
                try:
                    p_yes, mass = scores_cached(model, proc, msgs, cids, plen, yes_ids,
                                                no_ids, enc, think,
                                                args.answer_prefill)
                    if cache_checked < args.cache_check_images:
                        cache_checked += 1
                        ref, _ = uncached(proc, model, msgs, yes_ids, no_ids,
                                          think, args.answer_prefill,
                                          max_rows=args.max_rows)
                        # AMBIENT: the same uncached path again at a different batch width.
                        # Identical code, identical inputs, so any difference is bf16
                        # quantisation interacting with batch composition -- the noise this
                        # metric already carries, with no cache involved.
                        alt_rows = 2 if args.max_rows == 1 else 1
                        amb_scores, _ = uncached(proc, model, msgs, yes_ids, no_ids,
                                                 think, args.answer_prefill,
                                                 max_rows=alt_rows)
                        top = lambda v: max(range(len(codes)), key=lambda j: v[j])
                        cache_deltas.append(max(abs(a - b) for a, b in zip(ref, p_yes)))
                        amb_deltas.append(max(abs(a - b) for a, b in zip(ref, amb_scores)))
                        cache_agree.append(top(ref) == top(p_yes))
                        amb_agree.append(top(ref) == top(amb_scores))
                        print(f"[faces] cache check {cache_checked}/"
                              f"{args.cache_check_images}: cache |dP| "
                              f"{cache_deltas[-1]:.2e}  ambient |dP| {amb_deltas[-1]:.2e}",
                              flush=True)
                        # Use the reference scores while still deciding, so the check window
                        # is scored on the path the run would fall back to anyway.
                        p_yes = ref

                        if cache_checked == args.cache_check_images:
                            # Decide on DISTRIBUTIONS, not per-image. Both quantities are
                            # noisy maxima over six values; requiring every image to pass
                            # individually rejects on a single unlucky draw, which is what
                            # happened at check 4/8 (cache 7.3e-2 vs ambient 5.1e-2) for a
                            # cache whose 60-image aggregate is strictly better than ambient
                            # (mean 1.81e-2 vs 1.85e-2, labels 98.3% vs 96.7%).
                            cm = sum(cache_deltas) / len(cache_deltas)
                            am = sum(amb_deltas) / len(amb_deltas)
                            ca = sum(cache_agree) / len(cache_agree)
                            aa = sum(amb_agree) / len(amb_agree)
                            ok = (cm <= max(am * args.cache_ambient_mult,
                                            args.cache_tolerance)) and ca >= aa
                            # Spell the rule out. "cache 88% vs ambient 100%" and "cache
                            # 100% vs ambient 88%" look like the same finding and are
                            # opposite verdicts: the first is the cache disagreeing with a
                            # reference that is self-consistent, the second is a reference
                            # disagreeing with ITSELF more than the cache does. A reader
                            # already misread it once.
                            print(f"[faces] cache verdict over {cache_checked} images: "
                                  f"mean |dP| cache {cm:.2e} vs ambient {am:.2e}; "
                                  f"argmax: cache-vs-reference {ca:.0%}, "
                                  f"reference-vs-itself {aa:.0%} "
                                  f"-> {'ACCEPT' if ok else 'REJECT'} "
                                  f"(need cache-vs-reference >= reference-vs-itself, "
                                  f"{ca:.0%} {'>=' if ca >= aa else '<'} {aa:.0%})",
                                  flush=True)
                            if not ok:
                                print("[faces] using the uncached path for the rest of "
                                      "the run.", flush=True)
                                use_cache = False
                            p_yes, mass = ref, None
                except Exception as exc:
                    print(f"[faces] prefix cache unusable ({type(exc).__name__}: "
                          f"{str(exc)[:160]}); falling back to the uncached path",
                          flush=True)
                    use_cache = False
                    p_yes = None
            if p_yes is None:
                p_yes, mass = uncached(proc, model, msgs, yes_ids, no_ids,
                                       think, args.answer_prefill,
                                       max_rows=args.max_rows)
            if mass is None:
                # The fallback inside the self-check reused the reference scores and did not
                # re-measure mass; recompute rather than let the guard read a stale total.
                _, mass = uncached(proc, model, msgs[:1], yes_ids, no_ids, think,
                                   args.answer_prefill, max_rows=1)
                mass *= len(codes)
            mass_total += mass
            mass_count += len(codes)
            scores = dict(zip(codes, p_yes))
            pred = max(scores, key=scores.get)
            rec = {"scores": scores}
            # Same guard as verify_eval: a ratio between two tokens the model was never
            # going to emit looks like a score and is not one. Abort on the first image
            # rather than after hours (§N).
            if i == 0 and mass / len(codes) < args.min_yesno_mass:
                raise SystemExit(f"[faces] ABORT: mean yes/no mass {mass / len(codes):.2e} "
                                 f"< {args.min_yesno_mass} on the first image; this model "
                                 f"does not answer at the read position")
        else:
            msgs = [[{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": GENERATIVE_PROMPT}]}]]
            enc = encode(proc, msgs, think, None, add_generation_prompt=True,
                         tokenize=True, return_dict=True,
                         return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            text = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            pred = parse_generative(text)
            rec = {"raw_response": text}

        results.append({**it, "pred": pred, "correct": pred == it["gold"], **rec})

        if (i + 1) % args.checkpoint_every == 0 or i + 1 == len(items):
            done = sum(1 for r in results if r["correct"])
            rate = (i + 1) / max(time.time() - started, 1e-9)
            print(f"[faces] {i + 1}/{len(items)} acc {done / len(results):.3f} "
                  f"{rate:.2f} img/s", flush=True)
            # `use_cache` is local to this loop and can be switched off mid-run by the
            # self-check or by an exception; hand its CURRENT value to write() rather than
            # let batch_info report the request as though it were the outcome.
            args._cache_active = use_cache
            write(ckpt_path, args, tag, items, results, started, vram,
                  mass_total, mass_count)

    args._cache_active = use_cache
    write(ckpt_path, args, tag, items, results, started, vram, mass_total, mass_count)
    ckpt_path.replace(out_path)
    acc = sum(1 for r in results if r["correct"]) / len(results)
    unparsed = sum(1 for r in results if r["pred"] is None)
    print(f"\n[faces] {tag} {args.arm}: accuracy {acc:.3f} over {len(results)} images "
          f"(chance {1 / len(codes):.3f}), unparsed {unparsed}")
    print(f"[faces] wrote {out_path}")
    return 0


def code_sha1() -> str:
    """SHA1 of this file, recorded in every artifact it writes.

    Each cluster holds its own copy of this script, rsync'd by hand, and on 2026-08-17 one of
    them was three hours stale: it lacked --ablate-image and killed eight jobs on an argparse
    error. That failure was loud. The dangerous version is the silent one, where a copy is
    old enough to differ in behaviour but new enough to accept the flags, and the resulting
    JSON looks exactly like every other JSON. Stamping the file's own hash means a reader can
    tell two runs apart afterwards without trusting anyone's memory of which copy was current.
    """
    try:
        return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
    except OSError:
        return "unknown"


def write(path, args, tag, items, results, started, vram, mass_total, mass_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    acc = sum(1 for r in results if r["correct"]) / max(len(results), 1)
    json.dump({"batch_info": {
        "model": f"{tag} [faces_{args.arm}]", "model_id": args.model,
        "arm": f"faces_{args.arm}", "dataset": "FACES",
        "total_images": len(items), "scored": len(results),
        "accuracy": acc, "chance": 1 / len(EMOTIONS),
        "unparsed": sum(1 for r in results if r["pred"] is None),
        # Both, deliberately. Recording only the CLI string is what hid the inversion above:
        # every affected file says "thinking": "off" and was rendered with thinking on.
        "thinking": args.thinking, "enable_thinking": resolve_thinking(args),
        "ablate_image": args.ablate_image,
        # The numerics knobs. Absent until 2026-08-17, which meant the churn control's two
        # files were distinguishable from their baselines only by the Slurm log header --
        # a comparison whose whole claim is "these two runs differ in exactly these knobs"
        # cannot leave the knobs out of the artifact. `prefix_cache` is the requested mode;
        # `prefix_cache_used` is what the self-check actually granted, and they differ
        # whenever the cache is requested and rejected.
        "code_sha1": code_sha1(),
        "max_rows": args.max_rows,
        "max_image_pixels": args.max_image_pixels,
        "prefix_cache": args.prefix_cache,
        "prefix_cache_used": bool(getattr(args, "_cache_active", False)),
        "answer_prefill": args.answer_prefill,
        "mean_yesno_mass": (mass_total / mass_count) if mass_count else None,
        "prompt": VERIFY_PROMPT if args.arm == "verify" else GENERATIVE_PROMPT,
        "elapsed_s": time.time() - started, "vram": vram.summary(),
    }, "results": results}, open(path, "w"))


if __name__ == "__main__":
    raise SystemExit(main())
