#!/usr/bin/env python
"""EmoNet 40-category taxonomy -> the 6/7/8-way label spaces of real-face FER datasets.

Needed to score an EmoNet-trained model (Empathic-Insight-Face, or our E1-E3 adapters) on
datasets whose demographics are REAL rather than prompt-assigned. See
[[emonet-demographics-are-prompt-parameters]] for why that substitution matters.

The mapping is the obvious objection to any transfer result, so read this first:

  * It is NOT defensible for absolute accuracy. "Model X scores 71% on FACES" depends
    entirely on how Elation / Amusement / Triumph / Pride were collapsed into "happy".
  * It IS defensible for a DIFFERENCE between models scored under the SAME mapping. The
    mapping is then a shared constant, so it cannot manufacture a change in the spread of
    per-group accuracies. That is why the headline should be delta-bias between an
    EmoNet-trained model and a comparison model, never a raw accuracy.

Three variants are provided so the result can be shown stable across plausible choices
rather than tuned to one. CORE is the pre-registered default; report it, and report the
others as robustness.

  CORE       only unambiguous members
  INCLUSIVE  adds defensible-but-arguable members (contempt->angry, sourness->disgust, ...)
  MINIMAL    one EmoNet category per target where a 1:1 name match exists

Two things the taxonomies genuinely do not share, which no mapping can fix:

  * **Neutral.** EmoNet has no neutral category -- every one of its 40 is a present
    emotion. FACES, RaFD and CAFE all have one. Assigning `emotional_numbness` to neutral
    is wrong: numbness is a felt state, not the absence of expression. The honest handling
    is NEUTRAL_BY_THRESHOLD -- predict neutral when no mapped category exceeds a cutoff --
    with the cutoff fixed on a held-out split, never tuned per model.
  * **Surprise.** FACES has no surprise class, so `astonishment_surprise` is unmapped
    there; RaFD and CAFE do have one.

Categories mapping to no target are excluded from the decision entirely rather than being
forced somewhere. Roughly a third of the taxonomy has no counterpart in a 6-way scheme
(awe, confusion, doubt, interest, longing, infatuation, sexual_lust, teasing, intoxication,
concentration, contemplation, ...) -- that is a real finding about the taxonomies, not a
defect in the mapping, and it should be reported as coverage.
"""

from __future__ import annotations

# The 40 head names as released by laion/Empathic-Insight-Face-* (from the .pth filenames).
EMONET_40 = [
    "affection", "amusement", "anger", "astonishment_surprise", "awe", "bitterness",
    "concentration", "confusion", "contemplation", "contempt", "contentment",
    "disappointment", "disgust", "distress", "doubt", "elation", "embarrassment",
    "emotional_numbness", "fatigue_exhaustion", "fear", "helplessness",
    "hope_enthusiasm_optimism", "impatience_and_irritability", "infatuation", "interest",
    "intoxication_altered_states_of_consciousness", "jealousy_&_envy", "longing",
    "malevolence_malice", "pain", "pleasure_ecstasy", "pride", "relief", "sadness",
    "sexual_lust", "shame", "sourness", "teasing", "thankfulness_gratitude", "triumph",
]

# --- target label spaces, verified on disk 2026-08-06 -------------------------------------
# FACES  data/FACES            2052 imgs, filename {id}_{age}_{gender}_{emo}_{set}
#        age y/m/o = 696/672/684, gender f/m = 1020/1032, 6 emotions x 342
# RaFD   $DATASETS/Radboud-Faces/images    8040 imgs, Rafd{angle}_{id}_{eth}_{gender}_{emo}_{gaze}
#        8 emotions x 1005; ethnicity Caucasian/Kid/Moroccan -- NOTE Moroccan subset is all male,
#        so ethnicity and gender are confounded there; Kid is an age group in the ethnicity field
# CAFE   $DATASETS/CAFE/...                1192 imgs, 154 child models aged 2-8, 7 emotions
TARGET_SPACES = {
    "faces": ["angry", "disgust", "fear", "happy", "neutral", "sad"],
    "rafd": ["angry", "contemptuous", "disgusted", "fearful", "happy", "neutral", "sad",
             "surprised"],
    "cafe": ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprise"],
}

# --- the mapping ---------------------------------------------------------------------------
# Keyed on the canonical six; the 7/8-way spaces reuse these and add their own entries below.
CORE = {
    "angry": ["anger", "impatience_and_irritability"],
    "disgust": ["disgust"],
    "fear": ["fear"],
    "happy": ["elation", "amusement", "contentment", "pleasure_ecstasy", "triumph", "pride",
              "affection", "thankfulness_gratitude", "hope_enthusiasm_optimism", "relief"],
    "sad": ["sadness", "disappointment", "helplessness", "distress"],
    "neutral": [],          # see NEUTRAL_BY_THRESHOLD; EmoNet has no neutral category
}

# Adds members that are defensible but contestable, so the result can be shown not to hinge
# on them. bitterness/malevolence/jealousy are anger-adjacent in valence but differ in
# arousal and social meaning; sourness is a taste-face that reads visually as disgust;
# pain/shame/embarrassment are distress-adjacent but have distinct facial signatures.
INCLUSIVE = {
    "angry": CORE["angry"] + ["bitterness", "malevolence_malice", "jealousy_&_envy",
                              "contempt"],
    "disgust": CORE["disgust"] + ["sourness", "contempt"],
    "fear": CORE["fear"] + ["distress"],
    "happy": CORE["happy"] + ["teasing", "awe"],
    "sad": CORE["sad"] + ["pain", "shame", "embarrassment", "fatigue_exhaustion"],
    "neutral": ["emotional_numbness"],
}

# Strict 1:1 name matches only. Weakest coverage, strongest defensibility -- useful as the
# floor case: if the bias result survives here, mapping choice is not driving it.
MINIMAL = {
    "angry": ["anger"],
    "disgust": ["disgust"],
    "fear": ["fear"],
    "happy": ["elation"],
    "sad": ["sadness"],
    "neutral": [],
}

# Extra entries for the 7/8-way spaces.
_EXTRA = {
    "surprise": ["astonishment_surprise"],
    "surprised": ["astonishment_surprise"],
    "contemptuous": ["contempt"],
    # RaFD/CAFE spell some classes as adjectives; alias them onto the canonical members.
    "disgusted": None, "fearful": None,
}
_ALIAS = {"disgusted": "disgust", "fearful": "fear"}

VARIANTS = {"core": CORE, "inclusive": INCLUSIVE, "minimal": MINIMAL}

# Predict neutral when no mapped category exceeds this score. EmoNet heads emit roughly 0-7.
# FIX THIS ON A HELD-OUT SPLIT AND NEVER TUNE IT PER MODEL -- a per-model cutoff would let
# every model pick the threshold that flatters its own bias profile.
NEUTRAL_BY_THRESHOLD = 1.0


def mapping_for(dataset: str, variant: str = "core") -> dict[str, list[str]]:
    """{target_label: [emonet categories]} for one dataset and mapping variant."""
    if dataset not in TARGET_SPACES:
        raise KeyError(f"unknown dataset {dataset!r}; have {sorted(TARGET_SPACES)}")
    base = VARIANTS[variant]
    out: dict[str, list[str]] = {}
    for label in TARGET_SPACES[dataset]:
        key = _ALIAS.get(label, label)
        if key in base:
            out[label] = list(base[key])
        elif label in _EXTRA and _EXTRA[label]:
            out[label] = list(_EXTRA[label])
        else:
            out[label] = []
    return out


def coverage(variant: str = "core") -> tuple[list[str], list[str]]:
    """(used, unmapped) EmoNet categories under a variant, across the 6-way space.

    The unmapped list is a reportable number, not an embarrassment: it quantifies how much
    of EmoNet's taxonomy has no counterpart in classical FER label spaces.
    """
    used = {e for members in VARIANTS[variant].values() for e in members}
    return sorted(used), sorted(set(EMONET_40) - used)


if __name__ == "__main__":
    for v in VARIANTS:
        used, unmapped = coverage(v)
        print(f"{v:<10} uses {len(used):2d}/40, unmapped {len(unmapped):2d}")
    print("\nunmapped under 'core':")
    print(" ", ", ".join(coverage("core")[1]))
    print("\nFACES mapping (core):")
    for k, vv in mapping_for("faces", "core").items():
        print(f"  {k:<9} <- {', '.join(vv) if vv else '(threshold rule)'}")
