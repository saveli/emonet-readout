#!/usr/bin/env python
"""What token does this checkpoint actually want at the first generated position?

`verify_eval.py` reads the logits at one position and renormalises P(yes) against P(no).
That is only meaningful if yes/no is where the mass is. Three models in the 2026-08-07
sweep put ~1e-14 there, and `--thinking off` moved GLM-4.6V only to 3e-4 -- so the template
flag is not reaching whatever those templates do.

This prints the rendered prompt tail and the top-k next tokens under each `--thinking`
setting, which is the difference between fixing the prompt and guessing at it.

Usage:
    python evaluation/verify_probe_first_token.py --model zai-org/GLM-4.6V-Flash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_eval import (  # noqa: E402
    VERIFY_PROMPT,
    answer_token_ids,
    apply_template,
    build_model,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--merge-adapter", action="store_true")
    ap.add_argument("--emotion", default="Anger")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--prefills", nargs="*", default=None,
                    help="assistant-side prefill candidates to try under --thinking off, "
                         "via continue_final_message. Empty string = no prefill.")
    ap.add_argument("--max-image-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--dataset", default="data/emonet-face-hq")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset

    from e0_prompt_sweep import shrink

    ds = load_dataset(args.dataset)["train"]
    img = shrink(ds[0]["path"], args.max_image_pixels)

    proc, model = build_model(args)
    tok = getattr(proc, "tokenizer", proc)
    yes_ids, no_ids = answer_token_ids(tok)

    msgs = [[{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": VERIFY_PROMPT.format(emotion=args.emotion)}]}]]

    for label, think in (("auto (template default)", None), ("off", False), ("on", True)):
        enc = apply_template(proc, msgs, think, add_generation_prompt=True, tokenize=True,
                             return_dict=True, return_tensors="pt")
        ids = enc["input_ids"]
        tail = tok.decode(ids[0, -24:], skip_special_tokens=False)
        with torch.inference_mode():
            out = model(**{k: (v.to(model.device) if torch.is_tensor(v) else v)
                           for k, v in enc.items()})
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        mass = float(probs[yes_ids].sum() + probs[no_ids].sum())
        top = torch.topk(probs, args.topk)
        print(f"\n--- thinking={label}  len={ids.shape[1]}  yes/no mass={mass:.6f}")
        print(f"    prompt tail: {tail!r}")
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"      {p:8.4f}  id={i:<8} {tok.decode([i])!r}")

    # Prefill candidates. `continue_final_message` is the documented way to hand a model a
    # partial assistant turn: the template renders the assistant header, appends the text,
    # and omits the closing EOS, so the next position is inside the reply rather than at
    # its start. Doing this by concatenating token ids instead would have to insert them
    # ahead of the batch padding, which is where the prefix-split and left/right padding
    # logic in verify_eval.py would break.
    for prefill in (args.prefills or []):
        pre_msgs = [m + [{"role": "assistant", "content": prefill}] for m in msgs]
        try:
            enc = apply_template(proc, pre_msgs, False, add_generation_prompt=False,
                                 continue_final_message=True, tokenize=True,
                                 return_dict=True, return_tensors="pt")
        except Exception as exc:
            print(f"\n--- prefill {prefill!r}: UNSUPPORTED "
                  f"({type(exc).__name__}: {str(exc)[:120]})")
            continue
        ids = enc["input_ids"]
        with torch.inference_mode():
            out = model(**{k: (v.to(model.device) if torch.is_tensor(v) else v)
                           for k, v in enc.items()})
        probs = torch.softmax(out.logits[0, -1, :].float(), dim=-1)
        mass = float(probs[yes_ids].sum() + probs[no_ids].sum())
        top = torch.topk(probs, 5)
        print(f"\n--- prefill {prefill!r}  len={ids.shape[1]}  yes/no mass={mass:.6f}")
        print(f"    prompt tail: {tok.decode(ids[0, -24:], skip_special_tokens=False)!r}")
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"      {p:8.4f}  id={i:<8} {tok.decode([i])!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
