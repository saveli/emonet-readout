#!/usr/bin/env python3
"""Reproduce `tab:transfer` from committed artifacts.

`A_transfer.tex` prints three accuracies on the five mapped FACES classes, and until now
none of them was reproducible from `analysis/`: the two EIF-head rows existed only as
stdout, and the probe row was silently the *six*-class figure. This emits all three,
five-class throughout, so the release claim in `6_limitations.tex` holds.

FACES is balanced by construction -- 171 persons x 12 images, two per emotion -- so each
of the six classes has exactly 342 images and dropping neutral leaves 1,710. Equal class
counts make the macro and micro averages identical, which is why a plain mean over the
surviving classes is the accuracy.

    python analysis/transfer_table.py
"""
import json
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
DROP = "neutral"
# Group sizes on the age axis, from faces_e3_report.json's per-group denominators.
GROUP_N = {"young": 696, "middle": 672, "old": 684}


def five_class_accuracy(path, axis=None):
    """Mean per-class accuracy over the non-neutral classes, groups weighted by size.

    `axis` names the age-axis key. The EIF artifacts store it as "age"; the probe stores
    both "age_all6" and "age_mapped5" and we want the five-class one.
    """
    d = json.load(open(path))
    axes = d["axes"]
    if axis is None:
        axis = "age_mapped5" if "age_mapped5" in axes else "age"
    per_class = axes[axis]["per_class"]
    total = sum(GROUP_N.values())
    accs = {}
    for cls, by_group in per_class.items():
        if cls == DROP:
            continue
        accs[cls] = sum(GROUP_N[g] * by_group[g] for g in GROUP_N) / total
    return sum(accs.values()) / len(accs), accs


def main():
    rows = {}
    for label, fname in (("released", "faces_class_by_group.json"),
                         ("zscored", "faces_class_by_group_zscored.json")):
        acc, per_class = five_class_accuracy(ANALYSIS / fname)
        rows[label] = {"accuracy": acc, "per_class": per_class, "source": fname}

    probe = json.load(open(ANALYSIS / "faces_linear_probe.json"))
    p_acc, p_per_class = five_class_accuracy(ANALYSIS / "faces_linear_probe.json")
    rows["probe"] = {"accuracy_six_class": probe["accuracy"],
                     "accuracy_five_class": p_acc,
                     "source": "faces_linear_probe.json"}

    released, zscored = rows["released"]["accuracy"], rows["zscored"]["accuracy"]
    probe_acc = p_acc
    out = {
        "classes": "five mapped (six minus %s)" % DROP,
        "n_images": 1710,
        "rows": rows,
        "gaps": {
            "head_scaling": zscored - released,
            "training_data": probe_acc - zscored,
            "raw": probe_acc - released,
        },
    }
    print(f"{'row':<34}{'five-class acc':>15}")
    print(f"{'benchmark-trained heads, released':<34}{released:>15.4f}")
    print(f"{'benchmark-trained heads, z-scored':<34}{zscored:>15.4f}")
    print(f"{'FACES-supervised linear probe':<34}{p_acc:>15.4f}"
          f"   (six-class figure currently in the table: {probe['accuracy']:.4f})")
    print(f"\nhead scaling  {out['gaps']['head_scaling']:+.4f}"
          f"   training data {out['gaps']['training_data']:+.4f}"
          f"   raw {out['gaps']['raw']:+.4f}")

    dest = ANALYSIS / "transfer_table.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    raise SystemExit(main())
