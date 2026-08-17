#!/usr/bin/env python
"""Why does the prefix cache refuse three models? Diagnose before fixing.

`verify_eval.py` builds a shared-prefix KV cache and self-checks it against the uncached
path. Three of eleven models fail that construction and fall back, at a 10-20x throughput
cost (InternVL 0.04 img/s, Qwen3.5 0.05, against 0.9 for gemma-3):

    InternVL3_5-8B  ValueError: Image features and image tokens do not match, tokens: 0, features: 2560
    MiniCPM-V-4.6   ValueError: Multimodal features and tokens do not match, tokens: 0, features: 424
    Qwen3.5-9B      AttributeError: 'LinearAttentionLayer' object has no attribute 'keys'

That is not only a speed problem. Cache compatibility tracks architecture family, so
selecting "the fast models" for a sweep silently selects two families (gemma and Qwen) and
calls the result a survey of eleven.

Two hypotheses, and this script tests both without committing to a fix:

  A. `plen` lands before the image placeholders. It comes from `same.index(False)` -- the
     first token position where the 40 rows diverge -- and if the rendered prompt puts
     anything row-dependent before the image, the sliced prefix carries pixel features and
     zero image tokens, which is exactly what "tokens: 0" reports.

  B. The cache is not uniformly a KV cache. `scores_cached` copies it with `layer.keys` /
     `layer.values` over `prefix_kv.layers`; a hybrid model whose linear-attention layers
     hold a recurrent state instead has no such attributes. Note that skipping those layers
     would be WRONG rather than merely incomplete -- their state would silently reset -- so
     the fix has to carry them, not drop them.

Stage A needs no weights (processor + template only). Stage B loads the model, so it is
opt-in via --with-weights.

Usage:
    python evaluation/prefix_cache_probe.py --models OpenGVLab/InternVL3_5-8B-HF ...
    python evaluation/prefix_cache_probe.py --models Qwen/Qwen3.5-9B --with-weights
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_eval import PROMPT_VARIANTS, encode  # noqa: E402

EMOTIONS = ["Anger", "Amusement", "Astonishment", "Elation", "Gratitude", "Sadness",
            "Fear", "Disgust"]


def stage_a(model_id, prefill, think, n_rows):
    """Where does the shared prefix end, and where do the image placeholders sit?"""
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tok = getattr(proc, "tokenizer", proc)
    if getattr(tok, "padding_side", None) != "left":
        tok.padding_side = "left"
    img = Image.new("RGB", (512, 512), (128, 128, 128))
    tmpl = PROMPT_VARIANTS["v0"]
    msgs = [[{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": tmpl.format(emotion=e)}]}]
        for e in EMOTIONS[:n_rows]]

    # Right padding, matching verify_eval: under left padding every row starts with pad
    # tokens and the "shared prefix" would be padding rather than the image.
    tok.padding_side = "right"
    enc = encode(proc, msgs, think, prefill, add_generation_prompt=True, tokenize=True,
                 return_dict=True, return_tensors="pt", padding=True)
    ids = enc["input_ids"]
    same = (ids == ids[0:1]).all(dim=0).tolist()
    plen = same.index(False) if False in same else ids.shape[1]

    # Does the processor actually honour tokenizer.padding_side? Some do not, and left-pad
    # regardless -- which shifts each row by a different amount and destroys the shared
    # prefix without any error. Passing padding_side explicitly to the call is the fix if so.
    fix = {}
    try:
        enc2 = encode(proc, msgs, think, prefill, add_generation_prompt=True, tokenize=True,
                      return_dict=True, return_tensors="pt", padding=True,
                      padding_side="right")
        ids2 = enc2["input_ids"]
        same2 = (ids2 == ids2[0:1]).all(dim=0).tolist()
        plen2 = same2.index(False) if False in same2 else ids2.shape[1]
        fix = {"plen_with_explicit_padding_side": plen2,
               "improved": plen2 > plen}
    except Exception as exc:
        fix = {"explicit_padding_side_failed": f"{type(exc).__name__}: {str(exc)[:120]}"}

    # Locate image placeholders by every id the config calls an image token.
    cand = {}
    for attr in ("image_token_id", "image_token_index", "img_context_token_id",
                 "image_start_token_id"):
        v = getattr(getattr(proc, "config", None), attr, None) or getattr(proc, attr, None)
        if isinstance(v, int):
            cand[attr] = v
    for name in ("<image>", "<img>", "<IMG_CONTEXT>", "<|image_pad|>", "<|image|>",
                 "<image_soft_token>"):
        try:
            i = tok.convert_tokens_to_ids(name)
            if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", -1):
                cand[name] = i
        except Exception:
            pass

    row0 = ids[0].tolist()
    found = {}
    for label, tid in cand.items():
        pos = [i for i, t in enumerate(row0) if t == tid]
        if pos:
            found[label] = (tid, len(pos), pos[0], pos[-1])

    # WHERE do the rows differ, and with what? plen == 0 means row 0 and row k differ at the
    # very first token, which under right padding should be impossible for a shared template
    # -- and it matters, because clamping plen past the image would then cache a prefix that
    # is not actually shared and corrupt every row silently.
    diverge = [i for i, s in enumerate(same) if not s][:12]
    sample = {}
    for pos in diverge[:6]:
        col = ids[:, pos].tolist()
        sample[pos] = {"ids": col[:6],
                       "decoded": [tok.decode([t]) for t in col[:6]],
                       "n_distinct": len(set(col))}
    lens = [int(m.sum()) for m in enc["attention_mask"]] if "attention_mask" in enc else []

    out = {"model": model_id, "seq_len": ids.shape[1], "plen": plen, "padding_fix": fix,
           "pad_token_id": getattr(tok, "pad_token_id", None),
           "row_true_lengths": lens, "first_divergences": diverge, "divergence_detail": sample,
           "image_tokens": {k: {"id": v[0], "count": v[1], "first": v[2], "last": v[3]}
                            for k, v in found.items()},
           "pixel_values_shape": (list(enc["pixel_values"].shape)
                                  if "pixel_values" in enc else None)}
    last_img = max((v[3] for v in found.values()), default=None)
    if last_img is None:
        out["verdict"] = ("NO image placeholders in input_ids -- this processor does not "
                          "expand the image at encode time, so slicing input_ids can never "
                          "carry them. Hypothesis A confirmed, and the plen fix is not "
                          "enough on its own.")
    elif plen <= last_img:
        out["verdict"] = (f"plen {plen} cuts at or before the last image token {last_img} "
                          f"-- Hypothesis A confirmed. Clamp plen > {last_img}.")
    else:
        out["verdict"] = (f"plen {plen} already clears the last image token {last_img} "
                          f"-- Hypothesis A REFUTED for this model; look elsewhere.")
    return out


def stage_b(model_id, prefill, think):
    """What does the prefix forward actually return as a cache?"""
    import torch
    from PIL import Image

    from verify_eval import build_model

    class A:
        model = model_id
        adapter = None
        merge_adapter = False
    proc, model = build_model(A)
    img = Image.new("RGB", (512, 512), (128, 128, 128))
    msgs = [[{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": PROMPT_VARIANTS["v0"].format(emotion="Anger")}]}]]
    enc = encode(proc, msgs, think, prefill, add_generation_prompt=True, tokenize=True,
                 return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        kv = model(**enc, use_cache=True).past_key_values
    info = {"model": model_id, "cache_class": type(kv).__name__}
    layers = getattr(kv, "layers", None)
    if layers is None:
        info["verdict"] = "cache exposes no .layers -- verify_eval's copy loop assumes it"
        return info
    kinds = {}
    for i, layer in enumerate(layers):
        name = type(layer).__name__
        # vars() is the instance __dict__ -- the state actually stored on this layer, as
        # opposed to dir(), which mixes in methods and properties that may raise on access.
        # A batch expansion that walks vars() and grows every leading-1 tensor dimension
        # handles both layer types without hard-coding keys/values vs conv/recurrent state.
        state = {}
        for k, v in vars(layer).items():
            if torch.is_tensor(v):
                state[k] = {"shape": list(v.shape), "dtype": str(v.dtype),
                            "leading_is_1": bool(v.shape and v.shape[0] == 1)}
            elif isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]):
                state[k] = {"list_of_tensors": len(v), "elem_shape": list(v[0].shape)}
            elif isinstance(v, (int, float, bool, type(None), str)):
                state[k] = v
        d = kinds.setdefault(name, {"count": 0, "indices": [], "has_keys": hasattr(layer, "keys"),
                                    "has_batch_repeat": hasattr(layer, "batch_repeat_interleave"),
                                    "instance_state": state})
        d["count"] += 1
        if len(d["indices"]) < 4:
            d["indices"].append(i)
    info["layer_kinds"] = kinds
    bad = [k for k, v in kinds.items() if not v["has_keys"]]
    info["verdict"] = (f"layers WITHOUT .keys: {bad} -- these hold a recurrent state that "
                       f"must be carried, not skipped. Hypothesis B confirmed."
                       if bad else "every layer has .keys -- Hypothesis B refuted here.")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--with-weights", action="store_true",
                    help="also run stage B, which loads the model (needs a GPU)")
    ap.add_argument("--answer-prefill", default="Answer:")
    ap.add_argument("--thinking", choices=["off", "on", "auto"], default="off")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("analysis/prefix_cache_probe.json"))
    args = ap.parse_args()

    results = []
    for m in args.models:
        print(f"\n===== {m}", flush=True)
        for stage, fn in (("A", lambda: stage_a(m, args.answer_prefill, args.thinking,
                                                args.rows)),
                          ("B", lambda: stage_b(m, args.answer_prefill, args.thinking))):
            if stage == "B" and not args.with_weights:
                continue
            try:
                r = fn()
                r["stage"] = stage
                results.append(r)
                for k, v in r.items():
                    if k != "verdict":
                        print(f"  {k}: {v}", flush=True)
                print(f"  VERDICT: {r['verdict']}", flush=True)
            except Exception:
                print(f"  stage {stage} raised:\n{traceback.format_exc()}", flush=True)
                results.append({"model": m, "stage": stage,
                                "error": traceback.format_exc()[-900:]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
