#!/usr/bin/env python
"""Feasibility check: score Empathic-Insight-Face on FACES (real faces, real demographics).

Answers one question before any bias analysis is attempted: does an EmoNet-trained model
transfer to real faces at all? If accuracy under the taxonomy mapping is at chance, the
per-group accuracies are noise and any "bias" computed from them is uninterpretable.

Pipeline: frozen SigLIP2 embeddings -> 40 MLP regression heads -> taxonomy map -> 6-way
FACES label. Inference only; no training.

FACES is the right target because it is balanced on the two axes that matter (age
y/m/o = 696/672/684, gender f/m = 1020/1032) and because age is exactly where synthetic-face
fidelity is most suspect -- see [[emonet-demographics-are-prompt-parameters]].

Usage:
    python evaluation/empathic_insight_faces.py --limit 200      # quick check
    python evaluation/empathic_insight_faces.py                  # all 2052
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from taxonomy_map import EMONET_40, NEUTRAL_BY_THRESHOLD, mapping_for  # noqa: E402

# FACES filename: {id}_{age}_{gender}_{emotion}_{set}.jpg
AGE = {"y": "young", "m": "middle", "o": "old"}
GENDER = {"f": "female", "m": "male"}
EMO = {"a": "angry", "d": "disgust", "f": "fear", "h": "happy", "n": "neutral", "s": "sad"}
FNAME = re.compile(r"^(\d+)_([ymo])_([fm])_([adfhns])_([ab])$")


class Head(object):
    """The released head architecture: 1152 -> 1024 -> 512 -> 256 -> 1 with ReLU+dropout.

    Reconstructed from the state dicts (layers.0/3/6/9), since the repo ships weights but no
    module definition. Indices 1,2 / 4,5 / 7,8 are the activation and dropout slots, which
    hold no parameters -- hence the gaps.
    """

    @staticmethod
    def build():
        import torch.nn as nn

        # The weights are keyed `layers.N.*`, so the Sequential must sit on an attribute
        # named `layers` -- a bare nn.Sequential loads as `N.*` and fails to match.
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(1152, 1024), nn.ReLU(), nn.Dropout(0.0),
                    nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.0),
                    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.0),
                    nn.Linear(256, 1),
                )

            def forward(self, x):
                return self.layers(x)

        return MLP()


def load_faces(root: Path, limit: int = 0):
    rows = []
    for p in sorted(glob.glob(str(root / "*.jpg"))) + sorted(glob.glob(str(root / "*.png"))):
        m = FNAME.match(Path(p).stem)
        if not m:
            continue
        pid, age, gender, emo, _set = m.groups()
        rows.append({"path": p, "person": pid, "age": AGE[age],
                     "gender": GENDER[gender], "gold": EMO[emo]})
    if limit:
        # Stratify by gold emotion. The directory is sorted by person then emotion, so
        # taking the head gives a handful of people, and STRIDING aligns with the 6-emotion
        # cycle and silently returns only 3 of the 6 classes -- which is what a first pass
        # here did, producing an "accuracy" computed over half the label space.
        import random
        by_gold: dict[str, list] = {}
        for r in rows:
            by_gold.setdefault(r["gold"], []).append(r)
        rng = random.Random(3407)
        per = max(1, limit // len(by_gold))
        out = []
        for g in sorted(by_gold):
            pool = by_gold[g]
            rng.shuffle(pool)
            out.extend(pool[:per])
        rng.shuffle(out)
        rows = out
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces-dir", type=Path, default=Path("data/FACES"))
    ap.add_argument("--heads-dir", type=Path, default=None,
                    help="dir of model_<emotion>_best.pth; default = the HF cache snapshot")
    ap.add_argument("--siglip", default="google/siglip2-so400m-patch16-384")
    ap.add_argument("--variant", default="core", choices=["core", "inclusive", "minimal"])
    ap.add_argument("--agg", default="max", choices=["max", "mean"],
                    help="how member-category scores combine into a target-class score")
    ap.add_argument("--neutral-threshold", type=float, default=NEUTRAL_BY_THRESHOLD)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("analysis/empathic_faces.json"))
    ap.add_argument("--dump-embeddings", type=Path, default=None,
                    help="save the frozen SigLIP2 features to this .npz, so a probe can be "
                         "trained on the same representation without a second GPU pass")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    if args.heads_dir is None:
        hits = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--laion--Empathic-Insight-Face-Large/snapshots/*"))
        if not hits:
            print("heads not found; run snapshot_download first", file=sys.stderr)
            return 1
        args.heads_dir = Path(hits[0])

    rows = load_faces(args.faces_dir, args.limit)
    if not rows:
        print(f"no FACES images under {args.faces_dir}", file=sys.stderr)
        return 1
    print(f"[eif] {len(rows)} images | age {dict(Counter(r['age'] for r in rows))} | "
          f"gender {dict(Counter(r['gender'] for r in rows))}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(args.siglip)
    # float32 throughout: the heads were trained on float32 SigLIP2 embeddings and are only
    # 1.8 M params each, so there is nothing to gain from bf16 and a real risk of shifting
    # regression outputs near the neutral threshold. `dtype=` is the newer spelling and
    # `torch_dtype=` the older one; accept whichever this env's transformers offers.
    try:
        siglip = AutoModel.from_pretrained(args.siglip, dtype=torch.float32)
    except TypeError:
        siglip = AutoModel.from_pretrained(args.siglip, torch_dtype=torch.float32)
    siglip = siglip.to(dev).eval()

    heads, missing = {}, []
    for name in EMONET_40:
        f = args.heads_dir / f"model_{name}_best.pth"
        if not f.is_file():
            missing.append(name)
            continue
        sd = torch.load(f, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        h = Head.build()
        h.load_state_dict(sd)
        heads[name] = h.to(dev).eval()
    if missing:
        print(f"[eif] WARNING missing heads: {missing}", flush=True)
    print(f"[eif] loaded {len(heads)}/40 heads on {dev}", flush=True)

    mapping = mapping_for("faces", args.variant)
    print(f"[eif] mapping={args.variant} agg={args.agg} neutral_threshold="
          f"{args.neutral_threshold}", flush=True)

    scores = {}
    embeddings = []
    started = time.time()
    with torch.inference_mode():
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i:i + args.batch_size]
            imgs = [Image.open(r["path"]).convert("RGB") for r in chunk]
            enc = proc(images=imgs, return_tensors="pt").to(dev)
            emb = siglip.get_image_features(**enc).float()
            if args.dump_embeddings:
                embeddings.append(emb.cpu())
            for name, h in heads.items():
                scores.setdefault(name, []).append(h(emb).squeeze(-1).cpu())
            if i and (i // args.batch_size) % 10 == 0:
                done = i + len(chunk)
                rate = done / max(time.time() - started, 1e-9)
                print(f"[eif]   {done}/{len(rows)}  {rate:.1f} img/s", flush=True)
    scores = {k: torch.cat(v).numpy() for k, v in scores.items()}

    if args.dump_embeddings:
        import numpy as np
        args.dump_embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dump_embeddings,
            emb=torch.cat(embeddings).numpy(),
            person=np.array([r["person"] for r in rows]),
            age=np.array([r["age"] for r in rows]),
            gender=np.array([r["gender"] for r in rows]),
            gold=np.array([r["gold"] for r in rows]),
        )
        print(f"[eif] wrote {args.dump_embeddings} "
              f"({len(rows)} x {embeddings[0].shape[1]} frozen SigLIP2 features)", flush=True)

    labels = list(mapping)
    correct = 0
    per_group = defaultdict(lambda: [0, 0])
    confusion = Counter()
    # Per-image records, so the class x demographic breakdown can be recomputed offline.
    # An overall age gap is not a bias finding on its own: with two of six classes
    # collapsing entirely, it can equally be class difficulty correlated with age, and only
    # the joint table separates the two.
    records = []
    for idx, r in enumerate(rows):
        best_lab, best_val = None, -1e9
        # Keep the whole per-label score vector, not just the argmax. A class that never
        # wins the argmax may still rank second everywhere, which is a head-scale problem
        # fixable by per-head calibration -- and indistinguishable from a genuinely
        # unrecognised class if only the winner is recorded.
        agg_scores = {}
        for lab in labels:
            members = [m for m in mapping[lab] if m in scores]
            if not members:
                continue
            vals = [scores[m][idx] for m in members]
            v = max(vals) if args.agg == "max" else sum(vals) / len(vals)
            agg_scores[lab] = float(v)
            if v > best_val:
                best_lab, best_val = lab, float(v)
        # EmoNet has no neutral category: fall back to it only when nothing fires.
        pred = "neutral" if best_val < args.neutral_threshold else best_lab
        ok = pred == r["gold"]
        correct += ok
        confusion[(r["gold"], pred)] += 1
        records.append({"person": r["person"], "age": r["age"], "gender": r["gender"],
                        "gold": r["gold"], "pred": pred, "score": best_val,
                        "label_scores": agg_scores})
        for key in (("age", r["age"]), ("gender", r["gender"]), ("overall", "all")):
            per_group[key][0] += ok
            per_group[key][1] += 1

    n = len(rows)
    chance = 1.0 / len(labels)
    acc = correct / n
    print(f"\n[eif] overall accuracy {acc:.3f}  (chance {chance:.3f}, n={n})")

    import statistics
    out = {"n": n, "accuracy": acc, "chance": chance, "variant": args.variant,
           "agg": args.agg, "neutral_threshold": args.neutral_threshold, "groups": {}}
    for axis in ("age", "gender"):
        vals = {}
        for (ax, grp), (c, t) in sorted(per_group.items()):
            if ax == axis:
                vals[grp] = c / t
                print(f"      {axis:<7} {grp:<7} {c/t:.3f}  (n={t})")
        out["groups"][axis] = vals
        if len(vals) > 1:
            spread = statistics.pstdev(list(vals.values()))
            out["groups"][axis + "_std"] = spread
            print(f"      {axis} bias (std of group accuracies): {spread:.4f}")

    print("\n  gold -> predicted (top 12):")
    for (g, p), c in confusion.most_common(12):
        print(f"      {g:<8} -> {p:<8} {c:>5}")
    out["confusion"] = {f"{g}->{p}": c for (g, p), c in confusion.items()}
    out["records"] = records
    out["predicted_distribution"] = dict(Counter(p for (_g, p), c in confusion.items()
                                                 for _ in range(c)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n[eif] wrote {args.out}")
    if acc < chance * 1.5:
        print("[eif] VERDICT: at or near chance. Per-group accuracies are noise and a bias "
              "number computed from them is uninterpretable. Fix transfer before analysing "
              "bias.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
