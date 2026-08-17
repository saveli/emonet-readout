#!/usr/bin/env python
"""Why does Qwen3.5's prefix cache flip FACES labels when nobody else's does?

`prefix_cache_probe.py` answers "does the cache BUILD" -- that was section V, and it is
fixed. This answers the next question: the cache builds, self-checks fine on EmoNet
(section V: |dP| 1.81e-02 against a 5e-02 tolerance), and then loses the FACES argmax
check at 88% agreement against a reference that agrees with itself 100% of the time.

Qwen3.5 is ALONE in this. Measured over all ten arms of the 2026-08-14 rerun, MiniCPM,
GLM, Ministral and gemma-4-12B all pass at 100%, so whatever this is, it is not generic
to hybrid linear-attention caches -- MiniCPM is one too and it passes.

Three hypotheses, and the row-position axis separates them. FACES batches six rows, one
per emotion, so a batch-1 prefix state is widened to six:

  A. ALIASING. `expand_cache` hands out views rather than copies, so row 0's suffix mutates
     the recurrent state the later rows still need. Predicts: error RISES with row position
     and gets worse as batch width grows. This is the one that would also be silently wrong
     rather than merely imprecise.

  B. PARTIAL WIDENING. Qwen3.5 stores `conv_states = {0: tensor}` keyed by layer while
     MiniCPM stores a bare tensor (section V). If only the reachable entry is widened, most
     layers keep a batch-1 state. Predicts: error INDEPENDENT of row position, and present
     from row 0.

  C. NEAR-TIES. The cache is as accurate as it ever was and the FACES metric is simply
     harsher -- a six-way argmax has no tolerance, where EmoNet's continuous kappa_w
     absorbs the same perturbation. Predicts: no row-position pattern, and disagreements
     concentrated where the top-2 gap is small. This is the cheap outcome: the cache would
     be safe to keep for EmoNet and merely marginal for FACES.

C is the null and is tested by the `top2 gap` column, not assumed away. Section V refuted
two plausible fixes that would each have failed silently, so nothing here edits
`expand_cache` -- it only measures.

Usage:
    python evaluation/cache_argmax_probe.py --model Qwen/Qwen3.5-9B --images 40
    python evaluation/cache_argmax_probe.py --model openbmb/MiniCPM-V-4.6 --images 40  # control
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from faces_eval import EMOTIONS, VERIFY_PROMPT, resolve_thinking, scan_faces  # noqa: E402
from verify_eval import answer_token_ids, build_model, encode, scores_cached, uncached  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--faces-dir", type=Path,
                    default=Path("data/FACES"))
    ap.add_argument("--images", type=int, default=40)
    ap.add_argument("--thinking", choices=["off", "on", "auto"], default="off")
    ap.add_argument("--answer-prefill", default="Answer:")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--merge-adapter", action="store_true")
    ap.add_argument("--max-rows", type=int, default=1,
                    help="batch width of the REFERENCE pass, matching faces_eval's default")
    # MUST match faces_eval's default (1024*1024), not e0's 262144. Resolution sets the
    # image-token count, hence the prefix length and the cache being judged -- faces_eval
    # line 218 says so explicitly. A first run of this probe used 262144 and reported 100%
    # argmax agreement for a cache that the production job rejects at 88%: it was measuring
    # a different object.
    ap.add_argument("--max-image-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from e0_prompt_sweep import shrink

    think = resolve_thinking(args)
    tag = args.tag or args.model.split("/")[-1]
    items = scan_faces(args.faces_dir)[:args.images]
    if not items:
        raise SystemExit(f"no FACES images under {args.faces_dir}")

    codes = list(EMOTIONS)                      # six rows, fixed order
    proc, model = build_model(args)
    tokenizer = getattr(proc, "tokenizer", proc)
    # The TOKENIZER, not the processor -- Qwen3VLProcessor has no .encode, and faces_eval
    # unwraps it the same way at its own call site.
    yes_ids, no_ids = answer_token_ids(tokenizer)

    n_rows = len(codes)
    cache_err = [[] for _ in range(n_rows)]
    amb_err = [[] for _ in range(n_rows)]
    agree_cache, agree_amb, gaps_disagree, gaps_agree = [], [], [], []

    with torch.inference_mode():
        for k, it in enumerate(items):
            img = shrink(Image.open(it["path"]).convert("RGB"), args.max_image_pixels)
            msgs = [[{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": VERIFY_PROMPT.format(emotion=EMOTIONS[c])}]}]
                for c in codes]

            # Reference and ambient exactly as faces_eval builds them: the ambient floor is
            # the SAME uncached path at a DIFFERENT batch width, not a repeat run. Two
            # identical calls would be deterministic and report an ambient of ~0, which
            # would make every cache look bad by comparison.
            ref, _ = uncached(proc, model, msgs, yes_ids, no_ids, think,
                              args.answer_prefill, max_rows=args.max_rows)
            alt_rows = 2 if args.max_rows == 1 else 1
            amb, _ = uncached(proc, model, msgs, yes_ids, no_ids, think,
                              args.answer_prefill, max_rows=alt_rows)

            # Right-pad passed to the CALL, not just set on the tokenizer -- section V.
            tokenizer.padding_side = "right"
            enc = encode(proc, msgs, think, args.answer_prefill,
                         add_generation_prompt=True, tokenize=True, return_dict=True,
                         return_tensors="pt", padding=True, padding_side="right")
            tokenizer.padding_side = "left"
            enc = {kk: (v.to(model.device)
                        if torch.is_tensor(v) and kk != "pixel_values" else v)
                   for kk, v in enc.items()}
            cids = enc["input_ids"]
            same = (cids == cids[0:1]).all(dim=0).tolist()
            plen = same.index(False) if False in same else cids.shape[1]
            cached, _ = scores_cached(model, proc, msgs, cids, plen, yes_ids, no_ids,
                                      enc, think, args.answer_prefill)

            for r in range(n_rows):
                cache_err[r].append(abs(float(cached[r]) - float(ref[r])))
                amb_err[r].append(abs(float(amb[r]) - float(ref[r])))

            top = lambda v: max(range(n_rows), key=lambda j: float(v[j]))
            agree_cache.append(top(ref) == top(cached))
            agree_amb.append(top(ref) == top(amb))

            srt = sorted((float(v) for v in ref), reverse=True)
            gap = srt[0] - srt[1]
            (gaps_agree if top(ref) == top(cached) else gaps_disagree).append(gap)

            if (k + 1) % 10 == 0:
                print(f"[probe] {k+1}/{len(items)}", flush=True)

    print(f"\n=== {tag}: {len(items)} images, {n_rows} rows/batch ===\n")
    print("row  mean|dP| cache   mean|dP| ambient   ratio")
    for r in range(n_rows):
        cm = statistics.mean(cache_err[r])
        am = statistics.mean(amb_err[r])
        print(f" {r}      {cm:.3e}         {am:.3e}      {cm/am if am else float('inf'):6.2f}")

    ca = sum(agree_cache) / len(agree_cache)
    aa = sum(agree_amb) / len(agree_amb)
    print(f"\nargmax agreement   cache {ca:.1%}   ambient {aa:.1%}")
    if gaps_disagree:
        print(f"top-2 gap  where cache DISAGREES  median {statistics.median(gaps_disagree):.4f}"
              f"  (n={len(gaps_disagree)})")
    if gaps_agree:
        print(f"top-2 gap  where cache agrees     median {statistics.median(gaps_agree):.4f}"
              f"  (n={len(gaps_agree)})")

    # Row-position slope is the discriminator: A predicts a rise, B and C predict flat.
    per_row = [statistics.mean(cache_err[r]) for r in range(n_rows)]
    first, last = per_row[0], per_row[-1]
    print(f"\nrow 0 {first:.3e} -> row {n_rows-1} {last:.3e}   "
          f"ratio {last/first if first else float('inf'):.2f}")
    print("READ: rises with row -> A (aliasing, expand_cache hands out views).")
    print("      flat and nonzero from row 0 -> B (partial widening of the dict state).")
    print("      flat, and disagreements sit at a much smaller top-2 gap -> C (near-ties).")

    if args.out:
        args.out.write_text(json.dumps({
            "model": args.model, "tag": tag, "n_images": len(items),
            "cache_err_by_row": [statistics.mean(c) for c in cache_err],
            "ambient_err_by_row": [statistics.mean(a) for a in amb_err],
            "argmax_agree_cache": ca, "argmax_agree_ambient": aa,
            "top2_gap_disagree": gaps_disagree, "top2_gap_agree": gaps_agree,
        }, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
