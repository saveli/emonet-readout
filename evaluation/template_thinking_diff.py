#!/usr/bin/env python
"""Does `--thinking off` change the rendered prompt for the models already scored?

The 2026-08-07 verification sweep ran before `verify_eval.py` had a `--thinking` flag, so
those files were produced under each chat template's own default. The three thinking
checkpoints (GLM-4.6V, MiMo-VL, Qwen3.5) are being re-run under `--thinking off`, which
leaves the sweep mixed-provenance unless the flag is a no-op everywhere else.

It should be: `apply_template` only passes `enable_thinking` where the template accepts it,
and gemma/InternVL/Ministral raise on the unexpected kwarg. But "should be" is not a
control. This renders the actual verification prompt both ways and compares token ids. Any
model that differs was scored under a prompt that is no longer the one the code produces,
and its file has to be re-run too rather than sitting in the same table.

CPU only -- processors and tokenizers, no weights.

Usage:
    python evaluation/template_thinking_diff.py --models-file evaluation/e0_models.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_eval import VERIFY_PROMPT, apply_template, encode  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-file", type=Path, default=Path("evaluation/e0_models.txt"))
    ap.add_argument("--emotion", default="Anger",
                    help="any single emotion; the suffix is the same shape for all 40")
    ap.add_argument("--answer-prefill", default="Answer:",
                    help="also check that continue_final_message renders this prefill. It "
                         "is template-dependent -- GLM rejects a prefill containing tags it "
                         "emits itself with 'the final message does not appear in the chat "
                         "after applying the chat template' -- and a sweep cannot use a "
                         "prefill that silently fails to render on some models.")
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor

    # A real image, not a path: the template inserts a fixed number of placeholder tokens
    # per image and some processors need to see the pixels to know how many.
    img = Image.new("RGB", (336, 336))
    msgs = [[{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": VERIFY_PROMPT.format(emotion=args.emotion)}]}]]

    rows, differ = [], []
    for line in open(args.models_file):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        hf = line.split()[0]
        try:
            proc = AutoProcessor.from_pretrained(hf, trust_remote_code=True)
        except Exception as exc:
            rows.append((hf, f"LOAD FAILED: {type(exc).__name__}"))
            continue
        kwargs = dict(add_generation_prompt=True, tokenize=True,
                      return_dict=True, return_tensors="pt")
        default = apply_template(proc, msgs, None, **kwargs)["input_ids"]
        off = apply_template(proc, msgs, False, **kwargs)["input_ids"]
        same = default.shape == off.shape and bool((default == off).all())

        # The prefill has to actually land at the END of the rendered prompt. A template
        # that accepts continue_final_message but re-orders or wraps the assistant turn
        # would leave the read position somewhere else entirely, and the yes/no mass guard
        # would only catch that after a model was loaded.
        tok = getattr(proc, "tokenizer", proc)
        try:
            pre = encode(proc, msgs, False, args.answer_prefill, **kwargs)["input_ids"]
            tail = tok.decode(pre[0, -8:], skip_special_tokens=False)
            ok = tail.rstrip().endswith(args.answer_prefill.rstrip())
            pstat = f"prefill={'ok' if ok else 'MISPLACED'} tail={tail!r}"
        except Exception as exc:
            ok = False
            pstat = f"prefill=UNSUPPORTED ({type(exc).__name__}: {str(exc)[:80]})"

        rows.append((hf, f"think_same={same}  {pstat}"))
        if not ok:
            differ.append(f"{hf} (prefill)")

    for hf, note in rows:
        print(f"{hf:46}{note}")
    print()
    if differ:
        print(f"the prefill {args.answer_prefill!r} does not render at the end of the "
              f"prompt for these -- the sweep cannot use it uniformly:")
        for hf in differ:
            print(f"  {hf}")
        return 1
    print(f"prefill {args.answer_prefill!r} renders last on every model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
