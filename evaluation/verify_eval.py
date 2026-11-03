#!/usr/bin/env python
"""Verification-protocol evaluation for E1-E3 -- see docs/e1-e3-spec.md section 3.

A model trained to answer yes/no about ONE emotion cannot honestly be scored by prompting
it for a 40-dimensional intensity vector. That mismatch is exactly what voided the first
DPO run (findings.md section 2): trained single-label, evaluated 40-way, emitted 0.38
emotions/image against a ground truth of 12.1.

So this asks the trained question, 40 times per image:

    Does this face express {emotion}? Answer yes or no.

and scores P(yes) -- the probability mass on the yes tokens, renormalised against the no
tokens -- as a continuous per-emotion value. Continuous scores keep the existing analysis
stack usable unchanged: analysis/rescore.py consumes this output directly, and quantile
calibration (analysis/c4_calibration_control.py) maps it onto the 0-7 gold scale.

No generation loop: we need one token, so a single forward pass and a read of the logits
at the final position is both exact and ~10x cheaper than calling generate().

Cost: 2500 images x 40 emotions = 100k forward passes, and the vision tower re-encoding the
image for every emotion dominates.

`--prefix-cache` removes that. All 40 prompts for one image share a prefix -- the image
tokens plus "Does this face express " -- which measures 1389 of 1410 tokens (99%) on
Qwen2.5-VL. The prefix is forwarded once, its KV cache is expanded across the batch, and
only each emotion's ~14-token suffix is forwarded. Measured 18.6x end to end.

Emotions are grouped by suffix length and each group runs unpadded, rather than padding a
single batch. That is not a micro-optimisation: this model's output demonstrably depends on
padding. On one image, left-padded, right-padded and one-at-a-time decoding all disagree
with each other by 3.1e-2 in P(yes) -- equally, in every pairing, which makes it bf16 logit
quantisation interacting with batch composition rather than a bug in any one path. Feeding
unpadded batches removes the variable entirely.

That noise is real but immaterial here: perturbing the base run's scores by +-3.1e-2 moves
kappa_w by -0.005 (5 seeds, sd 0.001) against a protocol effect of 0.22. The self-check
tolerance below is set from that measurement, so it accepts ambient numerical disagreement
and still catches an actual indexing or position_id bug, which shows up an order of
magnitude larger.

Usage:
    python evaluation/verify_eval.py --model Qwen/Qwen2.5-VL-3B-Instruct --limit 20
    python evaluation/verify_eval.py --model Qwen/Qwen2.5-VL-3B-Instruct \
        --adapter results_e1/e1_3 --tag e1_3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e0_prompt_sweep import VramTracker, emotion_list, shrink  # noqa: E402

# Must match VERIFY_PROMPT in training/build_verification_pairs.py, minus the "<image>"
# placeholder -- LLaMA-Factory strips that and hands the image to the mm_plugin, and the
# chat template below does the same. If these two drift, the evaluation stops measuring
# the thing that was trained.
VERIFY_PROMPT = "Does this face express {emotion}? Answer yes or no."

# E2 (2026-08-11). The paper's claim is that how you ask determines the result, so a single
# phrasing of the verification query is the obvious hole in it. These are paraphrases, not
# a prompt search: same question, same yes/no answer space, deliberately varying sentence
# frame and register. `v0` must stay byte-identical to VERIFY_PROMPT -- it is the tie back
# to the main sweep, and re-running it is the check that a variant run reproduces.
PROMPT_VARIANTS = {
    "v0": VERIFY_PROMPT,
    "v1": "Is {emotion} visible in this person's facial expression? Answer yes or no.",
    "v2": "Look at the face. Would you say it shows {emotion}? Answer yes or no.",
}

# Graded control (2026-10-15): the same one-query-per-category design, but the answer is a
# digit 0-7 instead of yes/no, so "one question per category" can be separated from
# "probability readout". One forward pass gives both readouts: the argmax digit (what a greedy
# decode would print) and the expected digit under the distribution over the eight digit
# tokens. Same answer prefix, same prefix cache, same image subset mechanism.
RATE_PROMPT = ("Rate how strongly {emotion} is shown, from 0 (absent) to 7 (maximal). "
               "Answer with one digit.")
DIGITS = [str(d) for d in range(8)]

YES_VARIANTS = ["yes", "Yes", "YES", " yes", " Yes"]
NO_VARIANTS = ["no", "No", "NO", " no", " No"]


def answer_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """First-token ids for yes/no across casing and leading-space variants.

    Which variant a tokenizer actually emits depends on the chat template's trailing
    whitespace, so collecting all of them and summing their mass is more robust than
    guessing one. Returns (yes_ids, no_ids), de-duplicated.
    """
    def ids_for(variants):
        out = []
        for v in variants:
            enc = tokenizer.encode(v, add_special_tokens=False)
            if enc:
                out.append(enc[0])
        return sorted(set(out))

    yes_ids, no_ids = ids_for(YES_VARIANTS), ids_for(NO_VARIANTS)
    overlap = set(yes_ids) & set(no_ids)
    if overlap:
        # Would make P(yes) meaningless; refuse rather than silently score noise.
        raise SystemExit(f"[verify] yes/no token ids overlap: {overlap}")
    if not yes_ids or not no_ids:
        raise SystemExit("[verify] could not resolve yes/no token ids for this tokenizer")
    return yes_ids, no_ids


def digit_token_ids(tokenizer) -> list[list[int]]:
    """First-token ids for each digit 0..7, with and without a leading space.

    A variant is kept only if its FIRST token decodes to the digit itself: some tokenizers
    split " 3" into a bare space and "3", and a shared space token would put the same id under
    every digit. Returns one de-duplicated id list per digit; overlap between digits refuses.
    """
    out = []
    for d in DIGITS:
        ids = set()
        for v in (d, " " + d):
            enc = tokenizer.encode(v, add_special_tokens=False)
            if enc and tokenizer.decode([enc[0]]).strip() == d:
                ids.add(enc[0])
        if not ids:
            raise SystemExit(f"[rate] no first-token id decodes to digit {d!r}")
        out.append(sorted(ids))
    flat = [i for ids in out for i in ids]
    if len(flat) != len(set(flat)):
        raise SystemExit(f"[rate] digit token ids overlap: {out}")
    return out


def digit_space_id(tokenizer):
    """The separate space token a digit-splitting tokenizer puts before a digit (' 3' ->
    [' ', '3']: Qwen, GLM, MiniCPM, Gemma-3), or None if ' 3' starts with a digit token."""
    enc = tokenizer.encode(" 3", add_special_tokens=False)
    if len(enc) >= 2 and tokenizer.decode([enc[0]]) == " " and tokenizer.decode([enc[1]]) == "3":
        return enc[0]
    return None


def make_readout(question_set, token_ids, append_ids=None):
    """Map next-token probabilities (rows, vocab) to (values, mass_sum, extras).

    verify: value = P(yes) / (P(yes) + P(no)) -- the original computation, unchanged;
            mass = P(yes) + P(no); extras None.
    rate:   value = expected digit under the renormalised distribution over the eight digit
            tokens; mass = their total probability (the answer-mass analogue); extras per row
            = {probs (8, renormalised), mass, argmax, expected}.
    """
    import torch

    if question_set == "verify":
        yes_ids, no_ids = token_ids

        def readout(probs):
            p_yes, p_no = probs[:, yes_ids].sum(-1), probs[:, no_ids].sum(-1)
            vals = (p_yes / (p_yes + p_no).clamp_min(1e-9)).tolist()
            return vals, float((p_yes + p_no).sum()), [None] * len(vals)
        readout.append_ids = []
        return readout

    def readout(probs):
        P = torch.stack([probs[:, ids].sum(-1) for ids in token_ids], dim=-1)   # (rows, 8)
        mass = P.sum(-1)
        Pn = P / mass.clamp_min(1e-12)[:, None]
        expected = (Pn * torch.arange(8, device=Pn.device, dtype=Pn.dtype)).sum(-1)
        argmax = Pn.argmax(-1)
        extras = [{"probs": [round(float(x), 6) for x in Pn[r]], "mass": float(mass[r]),
                   "argmax": int(argmax[r]), "expected": float(expected[r])}
                  for r in range(P.shape[0])]
        return expected.tolist(), float(mass.sum()), extras
    readout.append_ids = list(append_ids or [])
    return readout


def _rope_owners(model, max_depth=8):
    """Yield `model` and its nested wrappers, outermost first, for a rope-helper lookup.

    Walks the `model` / `base_model` submodule chain that PEFT and transformers build up.
    Breadth-first and id-deduplicated so a wrapper that points back at itself, or exposes
    the same module under two names, cannot loop.

    The first level is `model.model` then `model`, which is the exact order the previous
    two-level probe used. Some transformers versions carry a legacy `get_rope_index` on the
    ForConditionalGeneration class as well as on the inner model, so reordering these two
    would change which helper an UNADAPTED model resolves to -- and every verification
    number already in the paper was produced under this preference. Deeper levels are
    additions, not a reshuffle.
    """
    seen, level = set(), [getattr(model, "model", None), model]
    for _ in range(max_depth):
        nxt = []
        for obj in level:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            yield obj
            for attr in ("model", "base_model"):
                # getattr on a delegating wrapper can resolve to a module several levels
                # down; that is harmless, the dedup drops it as a repeat.
                nxt.append(getattr(obj, attr, None))
        if not nxt:
            return
        level = nxt


def _suffix_position_ids(model, enc_full, plen, attn):
    """position_ids for the suffix, from the model's own rope helper when it has one.

    Qwen2.5-VL uses 3D mrope and derives position_ids from the full input when they are not
    supplied -- which, with a cache holding the prefix, produces positions for the whole
    sequence that no longer line up with the suffix being forwarded. Returns None for models
    that need no help, in which case transformers' own default is correct.

    The helper's signature moves between versions (transformers 5.14 takes
    `mm_token_type_ids` second positionally, earlier ones took `image_grid_thw`), so the call
    is driven by inspect rather than a hardcoded argument order.

    The owner is searched for by DESCENDING the wrapper chain, not by probing two fixed
    levels. On a bare Qwen2.5-VL the helper sits on `model.model` (`Qwen2_5_VLModel`), but
    PEFT inserts `PeftModel` -> `LoraModel` -> `Qwen2_5_VLForConditionalGeneration`, and
    that class does NOT define `get_rope_index` -- it holds the model that does. Attribute
    delegation does not paper over this: `LoraModel.__getattr__` forwards to the wrapped
    module, but `nn.Module.__getattr__` only searches submodules and parameters, so the
    lookup stops at `Qwen2_5_VLForConditionalGeneration` and returns nothing.

    A two-level probe therefore returns None for every adapted Qwen2.5-VL, transformers
    derives position_ids for the whole sequence, and the suffix forward dies with
    `The size of tensor a (14) must match the size of tensor b (1403) at non-singleton
    dimension 2` -- 1403 being the cached prefix and 14 the suffix. The caller catches that,
    prints "prefix cache unusable" and falls back, which costs ~2h12m per checkpoint against
    ~10min cached. That is what killed all four E4 Qwen eval jobs on the wall clock, and it
    only ever fired with an adapter applied, never on the same model's untrained base.
    """
    import inspect

    for owner in _rope_owners(model):
        fn = getattr(owner, "get_rope_index", None) if owner is not None else None
        if fn is None:
            continue
        params = inspect.signature(fn).parameters
        call = {k: v for k, v in enc_full.items() if k in params}
        call["input_ids"] = enc_full["input_ids"]
        if "attention_mask" in params:
            call["attention_mask"] = attn
        got = fn(**call)
        pos_full = got[0] if isinstance(got, tuple) else got
        return pos_full[..., plen:]
    return None


def apply_template(proc, msgs, enable_thinking: bool | None, **kwargs):
    """apply_chat_template, passing enable_thinking only where the template accepts it.

    Qwen3.5, GLM-4.6V and MiMo-VL expose it; gemma/InternVL/Ministral do not and raise on
    the unexpected kwarg, so this falls back rather than gating on a model allow-list that
    would rot on the next release. Mirrors `e0_prompt_sweep.apply_template`, but takes the
    encoding kwargs from the caller because the three call sites here differ on padding.

    Not cosmetic. A thinking template closes its generation prompt inside a reasoning
    block, so the next token is chain-of-thought and not an answer. The first verification
    sweep ran without this and measured `mean_yesno_mass` 2.7e-14 (GLM-4.6V), 1.5e-17
    (MiMo-VL) and 2.6e-07 (Qwen3.5) against ~0.99 for every non-thinking model: P(yes) was
    a ratio between two tokens the model was never going to emit. MiMo scored exactly the
    all-zero baseline; GLM scored a plausible-looking 0.556 off rank-preserving noise,
    which is the more dangerous failure of the two.

    The flag must be identical at every call site: the cached path locates the shared
    prefix by comparing token ids across the batch, so a prefix encoded under one template
    variant and suffixes under another would silently mis-split.
    """
    if enable_thinking is not None:
        try:
            return proc.apply_chat_template(msgs, enable_thinking=enable_thinking, **kwargs)
        except (TypeError, ValueError):
            pass
    return proc.apply_chat_template(msgs, **kwargs)


def encode(proc, msgs, think, prefill, **kwargs):
    """Render the verification prompt, optionally with an assistant-side answer prefill.

    `--thinking off` is not sufficient on its own. Probed on one image (Anger), first
    generated position, mass on {yes,no}:

        model            template default   thinking=off   + "Answer:" prefill
        Qwen3.5-9B            2.6e-07          0.9993            --
        GLM-4.6V-Flash        2.7e-14          4.4e-05          0.850
        MiMo-VL-7B-RL         1.5e-17          1.5e-17          0.990
        Qwen2.5-VL-3B         0.9988           0.9988           0.9997

    GLM closes its reasoning block under the flag and then opens a prose preamble
    ("To ..." at p=0.92); MiMo's chat template has no thinking switch at all, so all three
    settings render byte-identically and it always opens `<think`. Neither model is
    refusing the task -- the answer simply is not at the position this protocol reads. The
    prefill puts it there.

    `continue_final_message` is the mechanism rather than concatenating token ids, because
    ids would have to be inserted ahead of the batch padding, which is exactly where the
    prefix-split and left/right padding logic in this file would break.

    The prefill is applied to EVERY model or none. It is not free on the models that did
    not need it -- it moved Qwen2.5-VL's P(yes) from 0.294 to 0.377 on the probe cell,
    an order of magnitude above the 3.1e-2 padding noise -- so applying it only where the
    mass guard fires would put two prompt protocols inside one comparability sweep.
    `<think></think>` was rejected for the same reason from the other side: it rescues
    MiMo (0.869) and destroys Qwen2.5-VL (1.7e-04, which emits `<|im_end|>`).
    """
    if prefill:
        msgs = [m + [{"role": "assistant", "content": prefill}] for m in msgs]
        kwargs = dict(kwargs, add_generation_prompt=False, continue_final_message=True)
    return apply_template(proc, msgs, think, **kwargs)


def expand_cache(prefix_kv, b):
    """Copy a batch-1 prefix cache and widen it to `b` rows, whatever the layer type.

    The obvious version -- rebuild the cache from `layer.keys` / `layer.values` -- assumes
    every layer stores KV, and quietly does not hold for hybrids. Qwen3.5-9B (3:1 Gated
    DeltaNet to Gated Attention, so 24 linear layers to 8 full) and MiniCPM-V-4.6 (18 to 6)
    keep a convolution window and a recurrent matrix instead -- and the two models do not
    even agree on how to store them:

        MiniCPM  conv_states =    tensor [1, 6144, 4]      recurrent_states =    tensor
        Qwen3.5  conv_states = {0: tensor [1, 6144, 4]}    recurrent_states = {0: tensor}

    So this walks the instance `__dict__` and grows every leading-1 tensor dimension it can
    reach, bare or inside a dict. Naming the attributes instead would have covered one model
    and not the other; an earlier diagnostic that only looked at bare tensors reported
    Qwen3.5 as holding "no state at all", which was an artifact of the diagnostic.

    Dropping the linear-attention layers instead would be a correctness bug rather than a
    missed optimisation: their state is recurrent over the sequence, so a suffix forward
    starting from an empty state computes a different function, silently. A layer that
    genuinely exposes no state therefore raises here rather than proceeding, and the caller
    falls back to the uncached path, which is the reference implementation.
    """
    import copy

    import torch

    def widen(v):
        if v.ndim >= 1 and v.shape[0] == 1:
            return v.expand(b, *([-1] * (v.ndim - 1))).contiguous()
        return v

    cache = copy.deepcopy(prefix_kv)
    for i, layer in enumerate(cache.layers):
        carried = 0
        for name, v in list(vars(layer).items()):
            if torch.is_tensor(v):
                setattr(layer, name, widen(v))
                carried += 1
            elif isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
                setattr(layer, name, {k: widen(x) for k, x in v.items()})
                carried += 1
        if not carried:
            raise RuntimeError(
                f"layer {i} ({type(layer).__name__}) exposes no cached state, so the "
                f"prefix cannot be split from the suffix without changing the function "
                f"this layer computes")
        # Some layers assert against a recorded batch size on the next forward.
        if getattr(layer, "max_batch_size", None) == 1:
            layer.max_batch_size = b
    return cache


def append_tokens(d, n_rows, n_cols, app):
    """Append token ids `app` to every per-token tensor of an encoded batch (input_ids get the
    ids, attention_mask ones, any other per-token tensor -- token_type_ids, mm_token_type_ids --
    zeros, i.e. text). Valid for left-padded batches and for unpadded suffix groups, where the
    end of every row is its real end."""
    import torch

    out = {}
    for k, v in d.items():
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[0] == n_rows and v.shape[1] == n_cols:
            if k == "input_ids":
                col = torch.tensor(app, dtype=v.dtype, device=v.device).expand(n_rows, -1)
            elif k == "attention_mask":
                col = torch.ones((n_rows, len(app)) + tuple(v.shape[2:]), dtype=v.dtype,
                                 device=v.device)
            else:
                col = torch.zeros((n_rows, len(app)) + tuple(v.shape[2:]), dtype=v.dtype,
                                  device=v.device)
            out[k] = torch.cat([v, col], dim=1)
        else:
            out[k] = v
    return out


def scores_cached_r(model, proc, msgs, ids, plen, readout, enc_batch, think, prefill):
    """P(yes) per row, forwarding the shared prefix once and each suffix group unpadded.

    `enc_batch` is the already-encoded batch; `ids` its input_ids; `plen` the shared prefix
    length. Groups rows by true length so no padding is introduced -- see the module
    docstring for why padding is not safe to introduce here.
    """
    import torch
    from collections import defaultdict

    dev = model.device
    # Re-encode a single message for the prefix instead of slicing the batch: vision tensors
    # are packed (pixel_values is (total_patches, dim) over the whole batch, not (B, ...)),
    # so a batch-index slice silently desynchronises them from image_grid_thw.
    enc1 = encode(proc, msgs[:1], think, prefill, add_generation_prompt=True, tokenize=True,
                  return_dict=True, return_tensors="pt").to(dev)
    seq1 = enc1["input_ids"].shape[1]
    prefix_in = {}
    for k, v in enc1.items():
        per_token = (torch.is_tensor(v) and v.ndim >= 2 and v.shape[0] == 1
                     and v.shape[1] == seq1)
        prefix_in[k] = v[:, :plen] if per_token else v
    prefix_kv = model(**prefix_in, use_cache=True).past_key_values

    n_rows, n_cols = ids.shape
    groups = defaultdict(list)
    for r in range(n_rows):
        groups[int(enc_batch["attention_mask"][r].sum())].append(r)

    out = [0.0] * n_rows
    extras = [None] * n_rows
    mass = 0.0
    for real_len, rows in sorted(groups.items()):
        b = len(rows)
        idx = torch.tensor(rows, device=dev)
        full = ids[idx, :real_len]
        attn_b = torch.ones_like(full)

        # A forward consumes the cache in place, so each group needs its own copy of the
        # prefix -- sharing one object would append group 2's suffix after group 1's.
        cache = expand_cache(prefix_kv, b)

        sliced = {}
        for k, v in enc_batch.items():
            if torch.is_tensor(v) and v.ndim >= 2 and v.shape[0] == n_rows and v.shape[1] == n_cols:
                sliced[k] = v[idx, :real_len]
            elif k == "image_grid_thw" and torch.is_tensor(v):
                sliced[k] = v[:b]
            else:
                sliced[k] = v
        sliced["input_ids"] = full
        app = getattr(readout, "append_ids", [])
        end = real_len
        if app:
            # forced tokens after the prefix (rate: the space a digit-splitting tokenizer puts
            # before a digit), appended to the unpadded suffix group
            sliced = append_tokens(sliced, b, real_len, app)
            full = sliced["input_ids"]
            attn_b = torch.ones_like(full)
            end = real_len + len(app)
        pos = _suffix_position_ids(model, sliced, plen, attn_b)

        res = model(input_ids=full[:, plen:end], attention_mask=attn_b,
                    past_key_values=cache,
                    cache_position=torch.arange(plen, end, device=dev),
                    use_cache=False, **({"position_ids": pos} if pos is not None else {}))
        probs = torch.softmax(res.logits[:, -1, :].float(), dim=-1)
        vals, m, ex = readout(probs)
        mass += m
        for j, r in enumerate(rows):
            out[r] = float(vals[j])
            extras[r] = ex[j]
    return out, mass, extras


def uncached_r(proc, model, msgs, readout, think, prefill, max_rows=8):
    """The original path: left-padded batches of (image, emotion) pairs.

    Kept as both the fallback and the reference the prefix-cache self-check is measured
    against.

    Sub-batched at `max_rows` because every row carries its own copy of the image. Shards
    are sized for the cached path's footprint (14.7 GB on a 3B), and a fallback that put all
    40 rows in one forward needed ~44 GB -- so Qwen3.5-9B and InternVL3_5-8B fell back
    correctly and then died of OOM, turning a graceful degradation into a lost job.
    """
    import torch

    conditional, mass, extras = [], 0.0, []
    for s in range(0, len(msgs), max_rows):
        enc = encode(proc, msgs[s:s + max_rows], think, prefill, add_generation_prompt=True,
                     tokenize=True, return_dict=True,
                     return_tensors="pt", padding=True).to(model.device)
        app = getattr(readout, "append_ids", [])
        if app:   # left-padded, so every row's real end is the last column
            enc = append_tokens(dict(enc), *enc["input_ids"].shape[:2], app)
        probs = torch.softmax(model(**enc).logits[:, -1, :].float(), dim=-1)
        # Renormalise over the answer tokens (make_readout). The absolute mass on them is
        # tracked separately: if it is near zero the model is not answering the question at
        # all, and the ratio would be scoring noise.
        vals, m, ex = readout(probs)
        conditional += vals
        mass += m
        extras += ex
    return conditional, mass, extras


def scores_cached(model, proc, msgs, ids, plen, yes_ids, no_ids, enc_batch, think, prefill):
    """Original yes/no signature (faces_eval.py imports it): returns (P(yes) list, mass)."""
    out, mass, _ = scores_cached_r(model, proc, msgs, ids, plen,
                                   make_readout("verify", (yes_ids, no_ids)), enc_batch, think,
                                   prefill)
    return out, mass


def uncached(proc, model, msgs, yes_ids, no_ids, think, prefill, max_rows=8):
    """Original yes/no signature (faces_eval.py imports it): returns (P(yes) list, mass)."""
    out, mass, _ = uncached_r(proc, model, msgs, make_readout("verify", (yes_ids, no_ids)),
                              think, prefill, max_rows)
    return out, mass


def build_model(args):
    import torch
    import transformers
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = None
    errors = []
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                     "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(args.model, trust_remote_code=True,
                                        dtype=torch.bfloat16, device_map="auto")
            print(f"[verify] loaded via {cls_name}", flush=True)
            break
        except Exception as exc:
            errors.append(f"{cls_name}: {type(exc).__name__}: {str(exc)[:160]}")
    if model is None:
        raise RuntimeError("no AutoModel class accepted this checkpoint:\n  "
                           + "\n  ".join(errors))

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        # Name the adapter family from its own config rather than calling everything LoRA.
        # E4 scores OFT adapters through this same path, and a log line that says "LoRA"
        # over an OFT run is the kind of small lie that survives into a methods section.
        kind = "adapter"
        cfg = Path(args.adapter) / "adapter_config.json"
        if cfg.is_file():
            try:
                kind = json.load(open(cfg)).get("peft_type", "adapter")
            except (json.JSONDecodeError, OSError):
                pass
        print(f"[verify] applied {kind} adapter {args.adapter}", flush=True)
        if args.merge_adapter:
            model = model.merge_and_unload()
            print("[verify] merged adapter into base weights", flush=True)

    model.eval()
    if getattr(proc, "tokenizer", None) is not None and proc.tokenizer.padding_side != "left":
        proc.tokenizer.padding_side = "left"   # decoder-only batching needs left padding
    return proc, model


def write_out(path, args, tag, results, n, started, vram, mass_total, mass_count,
              resumed_from=None):
    """Serialise results. Used for both the `.partial` checkpoint and the final file.

    One writer for both so a checkpoint is a valid result file -- if a run is abandoned
    halfway, the partial can still be read by the analysis stack rather than being a
    bespoke format that only the resume path understands.
    """
    ordered = [results[i] for i in sorted(results)]
    above_half = [sum(1 for v in r["predicted_emotions"].values() if v > 0.5) for r in ordered]
    mean_density = sum(above_half) / max(len(above_half), 1)
    mean_mass = mass_total / max(mass_count, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"batch_info": {
            "model": f"{tag} [{args.question_set}]", "model_id": args.model,
            "arm": args.question_set,
            "adapter": args.adapter, "total_images": n,
            "successful_requests": len(ordered), "successful_parses": len(ordered),
            "paper_compliant": 0,
            # Provenance: results produced before 2026-08-10 carry no key here and ran under
            # the template's own default, which for GLM/MiMo/Qwen3.5 means thinking on.
            "thinking": args.thinking,
            "answer_prefill": args.answer_prefill,
            # Absent on files written before 2026-08-11, which all ran v0.
            "prompt_variant": args.prompt_variant,
            "prompt_text": (RATE_PROMPT if args.question_set == "rate"
                            else PROMPT_VARIANTS[args.prompt_variant]),
            "question_set": args.question_set,
            "forced_suffix_ids": getattr(args, "forced_suffix_ids", []),
            "image_indices": sorted(results),
            "max_image_pixels": args.max_image_pixels,
            # Which images. Absent on files written before 2026-08-17, which all ran the
            # full 2500 or a head slice. Two --limit runs are only comparable when these
            # three agree, so they travel with the scores rather than with the submitter.
            "limit": args.limit,
            "limit_mode": args.limit_mode if args.limit else "all",
            "subset_seed": args.subset_seed,
            "mean_nonzero_emotions": mean_density,
            "mean_yesno_mass": mean_mass,
            # Raw accumulators, so a resume can continue the average rather than restarting
            # it. Without these mean_yesno_mass would cover only the post-resume images --
            # and that number is the guard on whether P(yes) means anything at all.
            "yesno_mass_total": mass_total, "yesno_mass_count": mass_count,
            "mean_attempts": 1.0,
            "elapsed_s": time.time() - started,
            # None on a clean run; on a resumed one this records what was inherited, so
            # elapsed_s is visibly not the total compute the row cost.
            "resumed_from": resumed_from,
            "vram": vram.summary(),
            # For the rate arm the *_yesno_mass fields hold the digit mass (sum of the eight
            # digit tokens' probability), the same guard under the same name.
            "protocol": ("expected and argmax digit 0-7 over 40 per-emotion rating queries"
                         if args.question_set == "rate"
                         else "P(yes) over 40 per-emotion verification queries"),
        }, "results": ordered}, fh)
    return ordered, mean_density, mean_mass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir from an E1-E3 run")
    ap.add_argument("--merge-adapter", action="store_true",
                    help="merge LoRA into base weights before evaluating (faster forward)")
    ap.add_argument("--tag", default=None, help="output name; defaults to model/adapter base")
    ap.add_argument("--dataset", default="data/emonet-face-hq")
    ap.add_argument("--out-dir", type=Path, default=Path("results_e1"))
    ap.add_argument("--limit", type=int, default=0, help="score N images instead of all")
    # Mirrors e0_prompt_sweep.py's --limit-mode, deliberately including its default. The
    # head of EmoNet-Face-HQ is not a random sample: the first 250 rows under-sample
    # ethnicity category 5 (3.6% against 7.7%) and cover 117 of 207 prompt variants. A
    # --limit run whose scores get kappa-scored and demographically sliced must not inherit
    # that skew, and two sibling scripts whose --limit means different things is exactly the
    # trap this codebase keeps paying for. Full runs (--limit 0) are unaffected either way.
    ap.add_argument("--limit-mode", choices=["random", "head"], default="random",
                    help="how --limit selects rows. random (default) draws a deterministic "
                         "subset seeded by --subset-seed; head takes the first N.")
    ap.add_argument("--subset-seed", type=int, default=0,
                    help="seed for --limit-mode random. Recorded in batch_info, and a "
                         "checkpoint written under a different subset is refused, because "
                         "two runs on different images are not a comparison.")
    ap.add_argument("--batch-size", type=int, default=40,
                    help="(image, emotion) pairs per forward pass")
    ap.add_argument("--max-image-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--vram-interval", type=float, default=5.0)
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="write the .partial checkpoint every N images. 0 disables it, which "
                         "means a killed run loses everything -- these are ~14 h jobs.")
    ap.add_argument("--prefix-cache", choices=["auto", "off"], default="auto",
                    help="auto: forward the shared image prefix once per image and reuse its "
                         "KV across the 40 emotions (~18x). Self-checks against the uncached "
                         "path on the first image and falls back if it disagrees.")
    ap.add_argument("--fallback-rows", type=int, default=8,
                    help="(image, emotion) pairs per forward on the uncached path. Each row "
                         "carries its own copy of the image, and shards are sized for the "
                         "cached footprint, so a 40-row fallback OOMs.")
    ap.add_argument("--cache-check-emotions", type=int, default=8,
                    help="how many emotions the one-off self-check compares. Kept small "
                         "because the uncached reference is what sets the run's VRAM peak.")
    ap.add_argument("--cache-tolerance", type=float, default=0.05,
                    help="max |dP(yes)| the self-check accepts. Default is set above the "
                         "measured bf16 batch-composition noise (3.1e-2) and well below what "
                         "an indexing or position_id bug produces.")
    ap.add_argument("--thinking", choices=["off", "on", "auto"], default="off",
                    help="reasoning mode for hybrid-thinking checkpoints. off (default) "
                         "passes enable_thinking=False where the chat template supports it "
                         "and is silently ignored where it does not. This protocol reads "
                         "the logits at the first generated position, so a thinking "
                         "template puts essentially all the mass on reasoning tokens -- see "
                         "apply_template(). auto leaves the template's own default alone.")
    ap.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS), default="v0",
                    help="paraphrase of the verification query (E2). v0 is the sweep's "
                         "prompt and the tie back to it; v1/v2 vary the sentence frame. "
                         "Recorded in batch_info so a mixed directory cannot be scored as "
                         "one arm by accident.")
    ap.add_argument("--question-set", choices=["verify", "rate"], default="verify",
                    help="verify: the yes/no query (P(yes)). rate: the graded control -- "
                         "'Rate how strongly X is shown, from 0 to 7. Answer with one digit.' "
                         "-- read as the distribution over the digit tokens 0-7 (expected and "
                         "argmax digit). Writes <tag>__rate[_n<limit>]_eval.json.")
    ap.add_argument("--answer-prefill", default=None,
                    help="assistant-side text the reply is forced to start with, so the "
                         "position this protocol reads is inside the answer rather than at "
                         "the start of a preamble. Empty string disables it. See encode() "
                         "for the measured per-model effect and for why it is all-or-none. "
                         "Default 'Answer:' for both. For rate, digit-splitting tokenizers "
                         "(Qwen, GLM, MiniCPM, Gemma-3) put a separate space token before a "
                         "digit, so the space is forced as one input token after the prefix "
                         "and the digit is read at the following position (a trailing space "
                         "in the prefix text is stripped by continue_final_message).")
    ap.add_argument("--min-yesno-mass", type=float, default=0.5,
                    help="abort after the first scored image if less than this share of the "
                         "next-token mass sits on yes/no. The end-of-run warning this "
                         "replaces fired only after the full job: three models in the "
                         "2026-08-07 sweep burned ~12 GPU-hours producing scores that were "
                         "a ratio between two ~1e-14 probabilities. 0 disables the check.")
    ap.add_argument("--resume", choices=["auto", "off"], default="auto",
                    help="auto: continue from <tag>__verify_eval.json.partial, skipping "
                         "images already scored.")
    args = ap.parse_args()
    if args.answer_prefill is None:
        args.answer_prefill = "Answer:"

    import torch
    from datasets import load_dataset

    ds = load_dataset(args.dataset)["train"]
    # Indices stay TRUE dataset indices whichever mode is used, so every downstream join --
    # the demographic slices, the gold-rating lookup in reliability_vs_performance.py -- keys
    # on the same integer it always did.
    if args.limit and args.limit < len(ds):
        if args.limit_mode == "random":
            import random as _random

            indices = sorted(_random.Random(args.subset_seed).sample(range(len(ds)), args.limit))
        else:
            indices = list(range(args.limit))
    else:
        indices = list(range(len(ds)))
    index_set = set(indices)
    n = len(indices)
    emotions = emotion_list(ds)
    tag = args.tag or Path(args.adapter or args.model).name
    think = None if args.thinking == "auto" else (args.thinking == "on")
    print(f"[verify] model={args.model} adapter={args.adapter} images={n} "
          f"emotions={len(emotions)} thinking={args.thinking} "
          f"prefill={args.answer_prefill!r} -> {n * len(emotions)} forwards", flush=True)

    if args.question_set == "rate":
        if args.prompt_variant != "v0":
            raise SystemExit("[rate] --prompt-variant applies to the verify question only")
        label = f"rate_n{n}" if args.limit else "rate"
    else:
        label = "verify"
    out_path = args.out_dir / f"{tag}__{label}_eval.json"
    ckpt_path = args.out_dir / f"{tag}__{label}_eval.json.partial"

    results: dict[int, dict] = {}
    mass_total, mass_count = 0.0, 0
    resumed_from = None

    # Resume before loading the model: if the checkpoint already covers every image there is
    # nothing to do, and we should not spend minutes loading weights to discover that.
    if args.resume != "off" and ckpt_path.is_file():
        try:
            prev = json.load(open(ckpt_path))
            for r in prev.get("results", []):
                # Drop rows outside the current selection: resuming a full run under
                # --limit N would otherwise carry foreign indices into the output and
                # report more images than were asked for. Membership, not `< n`, because
                # under --limit-mode random the selected indices are not 0..n-1.
                if r["image_index"] in index_set:
                    results[r["image_index"]] = r
            bi = prev.get("batch_info", {})
            mass_total = float(bi.get("yesno_mass_total", 0.0))
            mass_count = int(bi.get("yesno_mass_count", 0))
        except (json.JSONDecodeError, KeyError, OSError, TypeError) as exc:
            print(f"[verify] checkpoint at {ckpt_path} unreadable ({type(exc).__name__}), "
                  f"starting clean", flush=True)
            results, mass_total, mass_count = {}, 0.0, 0
        if results:
            resumed_from = {"checkpoint": str(ckpt_path), "scored": len(results)}
            print(f"[verify] RESUMED from {ckpt_path.name}: {len(results)}/{n} images already "
                  f"scored, {n - len(results)} to go", flush=True)
            if mass_count == 0:
                # Pre-2026-08-06 checkpoints have no accumulators. Continuing would report a
                # mass averaged over only the remaining images, which is the guard on whether
                # P(yes) is meaningful -- say so rather than quietly reporting a partial mean.
                print("[verify] WARNING: checkpoint carries no yes/no mass accumulators; "
                      "mean_yesno_mass will cover only the images scored from here on.",
                      flush=True)

    proc, model = build_model(args)
    tokenizer = getattr(proc, "tokenizer", proc)
    if args.question_set == "rate":
        digit_ids = digit_token_ids(tokenizer)
        sp = digit_space_id(tokenizer)
        # The prefix is the verify arm's 'Answer:'; after it these tokenizers' next token is
        # the space before the digit (the HPC cluster probes 10117714/15: p(space) 0.9996-1.0000, and
        # transformers strips a trailing space from a continue_final_message prefix), so the
        # space is forced as one input token and the digit is read at the next position.
        append = [sp] if (sp is not None and args.answer_prefill) else []
        print(f"[rate] digit ids={digit_ids} forced suffix ids={append}", flush=True)
        readout = make_readout("rate", digit_ids, append)
        args.forced_suffix_ids = append
        prompt_tmpl = RATE_PROMPT
        check_scale = 7.0     # self-check compares expected digits on the P(yes) scale
    else:
        yes_ids, no_ids = answer_token_ids(tokenizer)
        print(f"[verify] yes ids={yes_ids} no ids={no_ids}", flush=True)
        readout = make_readout("verify", (yes_ids, no_ids))
        prompt_tmpl = PROMPT_VARIANTS[args.prompt_variant]
        check_scale = 1.0
    print(f"[verify] prompt {args.prompt_variant}: {prompt_tmpl}", flush=True)
    print(f"[verify] subset: limit={args.limit} mode="
          f"{args.limit_mode if args.limit else 'all'} seed={args.subset_seed} "
          f"first indices {indices[:5]}", flush=True)
    # A resumed checkpoint carries no record of which phrasing produced it before this flag
    # existed, so a variant run must not silently continue a v0 partial. Same class of bug
    # as the mixed-arm directory the --prompt-variant help text warns about.
    if resumed_from and args.prompt_variant != "v0":
        prev = json.load(open(ckpt_path))["batch_info"].get("prompt_variant", "v0")
        if prev != args.prompt_variant:
            raise SystemExit(f"[verify] checkpoint {ckpt_path.name} was written with prompt "
                             f"variant {prev!r}, refusing to resume it as "
                             f"{args.prompt_variant!r}")
    # Same class of guard for the image subset. The index filter above already drops foreign
    # rows, so a mismatched checkpoint would not corrupt the scores -- it would silently make
    # the run re-score everything and report a resume that saved nothing. Say so instead.
    if resumed_from and args.limit:
        bi = json.load(open(ckpt_path))["batch_info"]
        prev_mode = bi.get("limit_mode", "head")
        prev_seed = bi.get("subset_seed", 0)
        if (prev_mode, prev_seed) != (args.limit_mode, args.subset_seed):
            raise SystemExit(f"[verify] checkpoint {ckpt_path.name} covers subset "
                             f"{prev_mode}/seed {prev_seed}, refusing to resume it as "
                             f"{args.limit_mode}/seed {args.subset_seed} -- these are "
                             f"different images.")

    vram = VramTracker(interval=args.vram_interval)
    vram.start()

    started = time.time()
    use_cache = args.prefix_cache != "off"
    checked = False
    mass_checked = False

    with torch.inference_mode():
        for done, i in enumerate(indices):
            if i in results:
                continue
            img = shrink(ds[i]["path"], args.max_image_pixels)
            scores: dict[str, float] = {}
            rate_extras: dict[str, dict] = {}

            for start in range(0, len(emotions), args.batch_size):
                chunk = emotions[start:start + args.batch_size]
                msgs = [[{"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_tmpl.format(emotion=e)}]}]
                    for e in chunk]

                # Right-pad only to locate the shared prefix: under left padding every
                # sequence starts with pad tokens and the "common prefix" would be padding
                # rather than the image. The cached path never forwards this batch.
                #
                # `padding_side` is passed to the CALL as well as set on the tokenizer,
                # because some processors ignore the attribute. InternVL3_5-8B left-padded
                # regardless, which shifts each row by a different amount (row lengths here
                # are 280-282) and silently destroys the shared prefix: measured plen 0 with
                # the attribute alone, 266 with the explicit kwarg. That cost it the cache
                # and 10-20x throughput -- 0.04 img/s against 0.9 for gemma-3 -- and it read
                # as "InternVL is incompatible" rather than as a padding bug on our side.
                # No-op where the attribute was already honoured (Qwen3-VL and Qwen3.5 both
                # stay at plen 265), so it is safe to pass unconditionally.
                enc = None
                if use_cache:
                    tokenizer.padding_side = "right"
                    enc = encode(
                        proc, msgs, think, args.answer_prefill, add_generation_prompt=True,
                        tokenize=True, return_dict=True, return_tensors="pt", padding=True,
                        padding_side="right",
                    )
                    tokenizer.padding_side = "left"
                    # This batch holds one copy of the image per emotion, and on a 3B model
                    # that alone is ~33 GB of pixel_values on the GPU. The cached path only
                    # needs the token-level tensors and the tiny grid, so leave the rest on
                    # the host -- the prefix forward re-encodes the image once by itself.
                    enc = {k: (v.to(model.device)
                               if torch.is_tensor(v) and k != "pixel_values" else v)
                           for k, v in enc.items()}
                    ids = enc["input_ids"]
                    same = (ids == ids[0:1]).all(dim=0).tolist()
                    plen = same.index(False) if False in same else ids.shape[1]
                    try:
                        conditional, mass, extras = scores_cached_r(model, proc, msgs, ids,
                                                                  plen, readout, enc, think,
                                                                  args.answer_prefill)
                        if not checked:
                            checked = True
                            # Only the first few emotions: the uncached reference forwards
                            # one image copy per row, and at 40 rows that single call sets
                            # the run's VRAM peak (39 GB on a 3B model) and would drive
                            # shard sizing for a run whose steady state is far smaller.
                            k = min(args.cache_check_emotions, len(chunk))
                            ref, _, _ = uncached_r(proc, model, msgs[:k], readout,
                                                 think, args.answer_prefill,
                                                 args.fallback_rows)
                            delta = max(abs(a - b) for a, b in zip(ref, conditional[:k])) \
                                / check_scale
                            print(f"[verify] prefix-cache self-check on {k} emotions: prefix "
                                  f"{plen}/{ids.shape[1]} tokens, max |dP(yes)| {delta:.2e} "
                                  f"(tolerance {args.cache_tolerance:.2e})", flush=True)
                            if delta > args.cache_tolerance:
                                print("[verify] SELF-CHECK FAILED -- falling back to the "
                                      "uncached path for this chunk and the rest of the run.",
                                      flush=True)
                                use_cache = False
                                conditional, mass, extras = uncached_r(proc, model, msgs,
                                                                     readout, think,
                                                                     args.answer_prefill,
                                                                     args.fallback_rows)
                        mass_total += mass
                        mass_count += len(chunk)
                        for e, v, x in zip(chunk, conditional, extras):
                            scores[e] = round(float(v), 6)
                            if x is not None:
                                rate_extras[e] = x
                        continue
                    except Exception as exc:
                        # Fall through to the uncached path for this chunk. Scores already
                        # written for earlier chunks stay: both paths agree to within the
                        # ambient noise, so a mixed image is not a corrupted one.
                        print(f"[verify] prefix cache unusable ({type(exc).__name__}: "
                              f"{str(exc)[:160]}); falling back to the uncached path",
                              flush=True)
                        use_cache = False

                conditional, mass, extras = uncached_r(proc, model, msgs, readout, think,
                                                     args.answer_prefill, args.fallback_rows)
                mass_total += mass
                mass_count += len(chunk)
                for e, v, x in zip(chunk, conditional, extras):
                    scores[e] = round(float(v), 6)
                    if x is not None:
                        rate_extras[e] = x

            results[i] = {
                "image_index": i,
                "image_path": f"index_{i}",
                "model": tag,
                "success": True,
                "attempts": 1,
                "paper_compliant": False,
                "predicted_emotions": scores,
                "parse_success": True,
                "raw_response": "",
            }
            if args.question_set == "rate":
                # predicted_emotions = expected digit (the probability readout); the argmax
                # digit is the text readout a greedy decode would print.
                results[i]["argmax_emotions"] = {e: x["argmax"] for e, x in rate_extras.items()}
                results[i]["digit_probs"] = {e: x["probs"] for e, x in rate_extras.items()}
                results[i]["digit_mass"] = {e: round(x["mass"], 6) for e, x in rate_extras.items()}

            # Fail fast on a model that is not answering the question. The end-of-run
            # warning below only ever told us after the whole job; one image is enough to
            # know, and mean mass is ~0.97-1.00 on every model that works against ~1e-14 on
            # every model that does not, so the threshold is nowhere near a real value.
            if not mass_checked and mass_count:
                mass_checked = True
                m = mass_total / mass_count
                print(f"[verify] yes/no mass after image {i}: {m:.4f}", flush=True)
                if args.min_yesno_mass > 0 and m < args.min_yesno_mass:
                    vram.stop()
                    print(f"[verify] ABORT: only {m:.2%} of the next-token mass sits on "
                          f"yes/no (need {args.min_yesno_mass:.0%}). P(yes) would be a "
                          f"ratio between two tokens the model does not emit. Diagnose "
                          f"with evaluation/verify_probe_first_token.py, which prints the "
                          f"rendered prompt tail and the top-k tokens actually wanted "
                          f"there; to score anyway pass --min-yesno-mass 0.", flush=True)
                    return 2

            # Progress and checkpointing count POSITION IN THE SELECTION, not the dataset
            # index: under --limit-mode random the indices are sparse, so `(i+1) % 100`
            # would fire at arbitrary intervals and the checkpoint cadence would silently
            # stop being every N images.
            if (done + 1) % 100 == 0:
                # Rate is over images scored THIS run, not i+1 -- on a resume those differ,
                # and using i+1 would report a throughput the run never achieved and an ETA
                # that never arrives.
                done_here = len(results) - (resumed_from["scored"] if resumed_from else 0)
                rate = done_here / max(time.time() - started, 1e-9)
                print(f"[verify]   {len(results)}/{n} imgs  {rate:.2f} img/s  "
                      f"eta {(n - len(results)) / max(rate, 1e-9) / 60:.1f}m  "
                      f"vram_peak {vram.peak_mib / 1024:.1f} GB", flush=True)

            if args.checkpoint_every and (done + 1) % args.checkpoint_every == 0:
                write_out(ckpt_path, args, tag, results, n, started, vram,
                          mass_total, mass_count, resumed_from)

    vram.stop()

    # Density against the ground truth is a MANDATORY read-out (spec section 3): it is the
    # single number that would have caught the 0.38-vs-12.1 collapse immediately.
    _, mean_density, mean_mass = write_out(out_path, args, tag, results, n, started, vram,
                                           mass_total, mass_count, resumed_from)
    ckpt_path.unlink(missing_ok=True)

    print(f"[verify] DONE {out_path}\n"
          f"     images {n} | mean emotions>0.5 {mean_density:.2f} (human 8.14, GT 12.1) | "
          f"mean P(yes|no) mass {mean_mass:.3f} | {time.time() - started:.0f}s", flush=True)
    if mean_mass < 0.5:
        print(f"[verify] WARNING: only {mean_mass:.1%} of the next-token mass sits on "
              f"yes/no. The model is largely answering something else, so P(yes) is a "
              f"ratio between two unlikely tokens -- treat the scores as unreliable.",
          flush=True)
    v = vram.summary()
    if v["available"]:
        print(f"[verify] VRAM_PEAK {tag} peak={v['peak_gb']}GB "
              f"recommend_shard={v['recommended_shard_gb']}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
