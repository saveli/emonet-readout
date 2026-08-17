#!/usr/bin/env python
"""E0 -- prompt-protocol sweep over EmoNet-Face HQ.

The EmoNet-Face paper reports that most VLMs fail at fine-grained emotion recognition.
Their zero-shot harness does not merely *ask* for 25 emotions per image, it enforces the
count by rejection sampling: any response where `len(value) != 25 or any(v == 0)` is
discarded and regenerated, up to 20 times, with a scolding "aha moment" follow-up. So the
surviving output has exactly 25 emotions by construction. Measured on their published
predictions, five baselines emit 25.00 emotions with std 0.00 across all 2500 images,
against 8.14 (std 4.00) for human experts -- and all five score at their own Random
Baseline.

This sweep separates the prompt text from the enforcement mechanism:

  paper_retry  their exact prompt + the 25-pair rejection loop   (reproduces their number)
  paper_once   their exact prompt, single shot, no retry         (prompt text alone)
  ours         our free-cardinality prompt                       (no count constraint)

Comparing paper_retry against paper_once isolates how much of the reported failure is
caused by the enforcement loop rather than by model capability.

--arm picks the PROMPT; --accept picks the ACCEPTANCE TEST that drives the retry loop.
They used to be welded together, which conflated two unrelated jobs. The 25-pair test is
only defensible as a reproduction of LAION's harness -- as a measurement protocol it
rejects every response that agrees with the human ground truth (experts mark 8.14 of 40;
the test demands exactly 25). For measurement use `--accept parse`, which retries until
the response is parsable and imposes no cardinality at all:

    --arm paper_retry --accept paper25   reproduction; expect ~2.5% compliance at 5 rounds
    --arm paper_once  --accept parse     the defensible protocol, recovers unparsed images
    --arm ours        --accept none      single shot

Output matches the existing results/ schema so analysis/rescore.py and
analysis/reliability_vs_performance.py consume it unchanged.

Usage:
    python evaluation/e0_prompt_sweep.py --model Qwen/Qwen3.5-9B --arm paper_retry
    python evaluation/e0_prompt_sweep.py --model google/gemma-3-4b-it --arm ours --limit 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

# Their example output lists exactly 25 slots, and the retry loop rejects anything else.
PAPER_REQUIRED_PAIRS = 25
PAPER_MAX_ATTEMPTS = 20

# LAION posted to an OpenAI-compatible gateway (api.hyprlab.io/v1/chat/completions) with
# `max_tokens` as the ONLY generation parameter -- no temperature, top_p, top_k or seed in
# either published notebook. That means every baseline ran at the gateway default, which
# for the OpenAI chat-completions spec is temperature 1.0. Their rejection loop only
# converges *because* of that: at temperature 0 a retry re-sends an identical prompt to a
# deterministic decoder and gets back an identical answer, so `paper_retry` degenerates
# into round 1 repeated N times. Reproducing their protocol therefore requires sampling,
# and requires it uniformly across models -- per-model tuning would make the loop's
# convergence a function of each vendor's default entropy rather than of perception.
PAPER_GATEWAY_TEMPERATURE = 1.0

# Below this, a retry round is deterministic for practical purposes and the rejection loop
# cannot converge. Not a theoretical bound -- it exists because several vendor defaults sit
# just above zero (Qwen2.5-VL ships temperature 1e-6, Ministral's card says "below 0.1")
# and a naive `temperature <= 0` check would wave those through as if they were stochastic.
MIN_RETRY_TEMPERATURE = 0.05

# Vendor-recommended sampling, for the capability arms where each model should get its
# best shot rather than a protocol-matched uniform setting. Sources, checked 2026-08-05:
#   generation_config.json on the hub: gemma-4, GLM-4.6V, Qwen3-VL, Qwen2.5-VL
#   generation_config.json from the node-local HF cache (gated on the hub): gemma-3 4b/12b,
#     which ship do_sample/top_k/top_p but no temperature field, so HF's default 1.0 stands
#   model card prose:                  Qwen3.5 ("Instruct, general"), MiMo-VL, InternVL3.5,
#                                      Ministral ("temperature below 0.1")
# Note how heterogeneous these are: Ministral asks for <0.1, Qwen2.5-VL ships 1e-6 and GLM
# ships top_k=2 -- all effectively deterministic. That is exactly why the reproduction arm
# must NOT use these values; under them the rejection loop could never converge for those
# three models, and "compliance" would measure vendor defaults instead of perception.
# presence_penalty is pinned to 0 even where the card says otherwise (Qwen3.5 recommends
# 1.5): the output here is a 25-key JSON dict, and penalising repeated structural tokens
# ({, ", :) fights the format. Same call as ml/llm_features.py in the kodill project.
MODEL_SAMPLING = {
    "Qwen/Qwen3.5-9B":                          {"temperature": 0.7, "top_p": 0.8,  "top_k": 20},
    "Qwen/Qwen3-VL-8B-Instruct":                {"temperature": 0.7, "top_p": 0.8,  "top_k": 20},
    "Qwen/Qwen2.5-VL-3B-Instruct":              {"temperature": 1e-6, "repetition_penalty": 1.05},
    "google/gemma-4-12B-it":                    {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "google/gemma-3-4b-it":                     {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "google/gemma-3-12b-it":                    {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "zai-org/GLM-4.6V-Flash":                   {"temperature": 0.8, "top_p": 0.6,  "top_k": 2},
    "XiaomiMiMo/MiMo-VL-7B-RL-2508":            {"temperature": 0.3, "top_p": 0.95},
    "OpenGVLab/InternVL3_5-8B-HF":              {"temperature": 0.6, "top_p": 0.95, "top_k": 50},
    "mistralai/Ministral-3-14B-Instruct-2512":  {"temperature": 0.09},
}

PAPER_PROMPT = """
Please do the annotation of the image according to the instructions. The 0 shouldn't be in the result (the best way is to omit keys with emotions that are not present in the image). These are the instructions: {{ "instructions": "{instructions}", "options": {options} }}. Use this example output format:

Example:
{example_output}

Please provide the full output in this exact format as a JSON object without any code block formatting or additional text. Do not include any explanations or commentary—just the JSON object.

Remember to include all of the keys and values from the example, updating "file_name" with the actual image file name "{file_name}". Give the output now.
"""

PAPER_AHA = """
Wait, wait. Wait. That’s an aha moment. I can flag you here because you are not following my instructions.
Please ensure that the "value" dictionary in your output:

1. Contains exactly 25 key-value pairs, as per the example.
2. Does not include any keys with a value of 0 (omit keys for emotions that are not present).

Provide the corrected JSON output now.
"""

OUR_PROMPT = """You are an expert in facial emotion recognition. Analyze this facial image and rate the intensity of each of the following 40 emotions on a scale from 0 to 7, where:
- 0 = not present at all
- 1-2 = very low intensity
- 3-4 = moderate intensity
- 5-6 = high intensity
- 7 = extremely high intensity

Emotions to evaluate: {categories}

Please respond with ONLY a valid JSON object containing ONLY the emotions that have intensity greater than 0. Do not include emotions with intensity 0. For example:

{{
    "Anger": 5,
    "Sadness": 3,
    "Interest": 2
}}

If no emotions are present (all would be 0), respond with an empty JSON object: {{}}

Respond with ONLY the JSON, no other text."""


class VramTracker:
    """Sample this process's own GPU memory so the shard request can be sized from data.

    `--gres=shard:a40:N` is advisory -- nothing in the cgroup enforces it, so a job that
    asks for 34 GB and uses 38 GB simply steals the difference from whatever co-tenant
    the scheduler put on the same card. Measured on the live E0 sweep, 4 of 7 readable
    jobs were over their request (Ministral-3-14B asked 38, used 42.9 of a 46 GB card).

    NVML per-PID rather than torch.cuda: torch's allocator counters miss the ~0.5-1 GB
    CUDA context, and device-level free/total is useless here because the card is shared
    with another job. `used_memory` for our own PID is exactly what nvidia-smi reports and
    exactly what has to fit in the shard.

    A sampler thread is needed because peak VRAM lands mid-generate() -- the long-prompt
    paper_retry batches spike well above the post-load resting value, and a single reading
    after the run would report the trough.
    """

    def __init__(self, interval=5.0):
        self.interval = interval
        self.peak_mib = 0
        self.after_load_mib = 0
        self.marks: dict[str, int] = {}
        self.source = "none"
        self._nvml = None
        self._handle = None
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            # CUDA_VISIBLE_DEVICES may remap indices, so resolve by index 0 of what we see.
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.source = "pynvml"
        except Exception:
            # pynvml is not in emonet-infer and adding a dep to a conda env that is already
            # pinned against transformers>5.8 is not worth it -- nvidia-smi is always there.
            if self._probe_smi():
                self.source = "nvidia-smi"
            else:
                print("[vram] disabled: no pynvml and nvidia-smi unusable "
                      "(an A40 node has a driver/NVML version mismatch)", flush=True)

    @staticmethod
    def _probe_smi() -> bool:
        try:
            import subprocess

            subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, timeout=15, check=True)
            return True
        except Exception:
            return False

    def _sample_smi(self) -> int:
        import subprocess

        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15, check=True).stdout
        except Exception:
            return 0
        me = str(os.getpid())
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == me and parts[1].isdigit():
                return int(parts[1])
        return 0

    def _sample(self) -> int:
        """Our own process's VRAM in MiB, or 0 if unavailable."""
        if self.source == "nvidia-smi":
            return self._sample_smi()
        if self._handle is None:
            return 0
        try:
            procs = self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        except Exception:
            return 0
        me = os.getpid()
        for p in procs:
            if p.pid == me and p.usedGpuMemory:
                return int(p.usedGpuMemory) // (1024 * 1024)
        return 0

    def reset(self) -> None:
        """Zero the peak so a sweep can attribute one peak per configuration.

        The allocator does not hand memory back, so a later config inherits the high-water
        mark of an earlier one. Callers that sweep batch sizes must therefore go small to
        large and read the delta, or empty_cache() between configs -- see vram_profile.py.
        """
        self.peak_mib = 0

    def mark(self, label: str) -> int:
        """Record a named checkpoint (and fold it into the peak)."""
        mib = self._sample()
        self.marks[label] = mib
        self.peak_mib = max(self.peak_mib, mib)
        return mib

    def start(self):
        if self.source == "none" or self._thread is not None:
            return
        self.after_load_mib = self.mark("after_load")
        print(f"[vram] after model load: {self.after_load_mib / 1024:.1f} GB", flush=True)
        if self.interval <= 0:      # marks only, no sampler thread
            return

        def loop():
            while not self._stop.wait(self.interval):
                self.peak_mib = max(self.peak_mib, self._sample())

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)
            self._thread = None
        self.peak_mib = max(self.peak_mib, self._sample())

    def recommended_shard_gb(self) -> int:
        """Peak + 3 GB margin, rounded up. One shard == 1 GB on the a40 nodes."""
        if not self.peak_mib:
            return 0
        return int(-(-self.peak_mib // 1024)) + 3

    def summary(self) -> dict:
        return {
            "peak_mib": self.peak_mib,
            "peak_gb": round(self.peak_mib / 1024, 2),
            "after_load_gb": round(self.after_load_mib / 1024, 2),
            "marks_gb": {k: round(v / 1024, 2) for k, v in self.marks.items()},
            "recommended_shard_gb": self.recommended_shard_gb(),
            "requested_shard_gb": int(os.environ.get("E0_SHARD_GB", 0) or 0),
            "source": self.source,
            "available": self.peak_mib > 0,
        }


def write_results(args, ordered, attempts, emotions, n, started, suffix="", vram=None,
                  sampling=None, resumed_from=None):
    """Serialise results in the shared results/ schema. `suffix` marks partial dumps."""
    parsed = sum(r["parse_success"] for r in ordered)
    compliant = sum(r["paper_compliant"] for r in ordered)
    # Round-1 outcome, so a retry run still reports the single-shot condition it recovered
    # from. first_parsed is what a --accept none run of the same arm would have scored.
    first_parsed = sum(1 for r in ordered if r.get("first_parse_success"))
    have_first = sum(1 for r in ordered if r.get("first_parse_success") is not None)
    recovered = parsed - first_parsed
    nz = [sum(1 for v in r["predicted_emotions"].values() if v > 0)
          for r in ordered if r["parse_success"]]
    mean_nz = sum(nz) / len(nz) if nz else 0.0
    tag = args.model.strip("/").split("/")[-1]
    label = run_label(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{tag}__{file_label(args)}_eval.json{suffix}"
    with open(path, "w") as fh:
        json.dump({"batch_info": {
            "model": f"{tag} [{label}]", "model_id": args.model, "arm": label,
            "prompt_arm": args.arm, "accept": resolve_accept(args),
            "total_images": n, "limit_mode": args.limit_mode if args.limit else "all",
            "successful_requests": len(ordered),
            "successful_parses": parsed, "paper_compliant": compliant,
            "first_attempt_parses": first_parsed if have_first else None,
            "recovered_by_retry": recovered if have_first else None,
            "mean_nonzero_emotions": mean_nz,
            "mean_attempts": sum(attempts.values()) / max(len(attempts), 1),
            "elapsed_s": time.time() - started,
            # None on a clean run. On a resumed run this records what was inherited, so
            # elapsed_s is visibly NOT the total compute the row cost.
            "resumed_from": resumed_from,
            "vram": vram.summary() if vram is not None else None,
            "attn": args.attn, "thinking": args.thinking,
            "sampling": {"policy": args.sampling, "seed": args.seed,
                         "accept": resolve_accept(args),
                         "retries": args.retries, "backend": args.backend,
                         **(sampling or {})},
        }, "results": ordered}, fh)
    return path, parsed, compliant, mean_nz


def emotion_list(ds) -> list[str]:
    """The 40 bare emotion names, taken from the first row's expert annotations.

    The `label` column holds `[{rater: {'Family|Emotion': score, ...}}, ...]`; every rater
    scores all 40, so one rater's key set is the full taxonomy.
    """
    import ast

    raters = ast.literal_eval(ds[0]["label"])
    first = raters[0][next(iter(raters[0]))]
    return [k.split("|")[-1] for k in sorted(first)]


def load_paper_assets(assets_dir: Path):
    """Instructions / options / example output, extracted from the paper's notebook."""
    with open(assets_dir / "paper_prompt_assets.json") as fh:
        a = json.load(fh)
    return a["instructions"], a["options"], a["example_output"]


def build_prompt(arm: str, emotions: list[str], assets, file_name: str) -> str:
    if arm == "ours":
        return OUR_PROMPT.format(categories=", ".join(emotions))
    instructions, options, example_output = assets
    return PAPER_PROMPT.format(
        instructions=instructions,
        options=json.dumps(options),
        example_output=example_output,
        file_name=file_name,
    )


def strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
THINK_OPEN = re.compile(r"^\s*<think>.*", re.S | re.I)


def strip_think(text: str) -> str:
    """Drop reasoning blocks before JSON extraction.

    Belt and braces alongside `--thinking off`: some checkpoints emit <think> regardless of
    the template flag. Measured on the greedy paper_once runs, 2159/2159 of GLM-4.6V-Flash's
    unparsed responses contained a <think> tag, and Qwen3.5-9B emitted the same reasoning
    untagged -- both burned the whole max_new_tokens budget before reaching any JSON, which
    is why they parsed at 13.6% and 4.5% while gemma-4-12B parsed at 100%.

    An unterminated <think> (truncated mid-reasoning) leaves nothing recoverable, so that
    case returns empty rather than handing the JSON scanner a wall of prose to regex over.
    """
    t = THINK_BLOCK.sub("", text)
    if THINK_OPEN.match(t) and "</think>" not in t.lower():
        return ""
    return t


def apply_template(proc, msgs, enable_thinking: bool | None):
    """apply_chat_template, passing enable_thinking only where the template accepts it.

    Qwen3.5 and GLM-4.6V both expose it; gemma/InternVL/Ministral do not and raise on the
    unexpected kwarg, so this falls back rather than gating on a model allow-list that would
    rot on the next release.
    """
    kwargs = dict(add_generation_prompt=True, tokenize=True,
                  return_dict=True, return_tensors="pt", padding=True)
    if enable_thinking is not None:
        try:
            return proc.apply_chat_template(msgs, enable_thinking=enable_thinking, **kwargs)
        except (TypeError, ValueError):
            pass
    return proc.apply_chat_template(msgs, **kwargs)


def parse_response(text: str, arm: str, emotions: list[str]) -> tuple[dict | None, dict | None]:
    """Return (per-emotion scores keyed by bare emotion name, raw value dict).

    The paper's schema nests scores under `value` with `Family|Emotion` keys; ours is a
    flat `Emotion: score` object. Both normalise to bare emotion names.
    """
    t = strip_think(strip_code_fence(text))
    obj = None
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)          # salvage a trailing-prose response
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None, None
    if not isinstance(obj, dict):
        return None, None

    raw = obj.get("value") if arm != "ours" and isinstance(obj.get("value"), dict) else obj
    if not isinstance(raw, dict):
        return None, None

    valid = {e.lower(): e for e in emotions}
    scores = {e: 0 for e in emotions}
    for k, v in raw.items():
        if not isinstance(v, (int, float)):
            continue
        name = k.split("|")[-1].strip()
        canon = valid.get(name.lower())
        if canon:
            scores[canon] = v
    return scores, raw


def shrink(img, max_pixels: int):
    """Cap image area. EmoNet renders are 1024x1024, which is more tokens than the task needs."""
    if img.width * img.height <= max_pixels:
        return img
    scale = (max_pixels / (img.width * img.height)) ** 0.5
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))


def resolve_accept(args) -> str:
    """Which acceptance test drives the retry loop.

    Deliberately orthogonal to --arm. The retry loop and the 25-pair test used to be welded
    together, which conflated two unrelated jobs:

      paper25  reproduce LAION's harness, where "exactly 25 non-zero pairs" IS the object of
               study. Only meaningful for the reproduction arm. Measured on gemma-3-4b at
               temperature 1.0: 1/40 compliant after 5 rounds, ~0.5%/round, and the model
               emits 29.6 emotions -- it overshoots 25 rather than undershooting, and the
               test rejects both directions.
      parse    retry until the response is parsable. The defensible protocol: it recovers
               missing data instead of imposing a cardinality. Under greedy, paper_once
               parsed only 4.5% for Qwen3.5-9B and 13.6% for GLM-4.6V-Flash, so those two
               models were being ranked on a self-selected tenth of the benchmark.
      none     single shot.

    Caveat for `parse`: retrying only recovers *stochastic* parse failures. A model that
    systematically answers in prose will fail all N attempts, and the run just costs N times
    as much. Probe with --limit before committing a full arm.
    """
    if args.accept != "auto":
        return args.accept
    return "paper25" if args.arm == "paper_retry" else "none"


def run_label(args) -> str:
    """Output name for this (arm, accept) combination.

    The acceptance test changes what the run measures, so it has to be part of the
    identity. `--arm paper_once --accept parse` and plain `--arm paper_once` are different
    experiments; sharing `paper_once_eval.json` would make the second silently overwrite
    the first, and make the submit script's done-guard skip a run that has not happened.
    Legacy combinations keep their bare arm name so existing files stay addressable.

    This is the ARM identity and goes into batch_info; it deliberately excludes the row
    count, because analysis/e0_report.py filters on it against a fixed ARMS list. The row
    count belongs to the FILE identity -- see file_label().
    """
    accept = resolve_accept(args)
    legacy = "paper25" if args.arm == "paper_retry" else "none"
    return args.arm if accept == legacy else f"{args.arm}_{accept}"


def file_label(args) -> str:
    """Output filename stem for this run: the arm identity plus the row count.

    The count was missing until 2026-08-06, even though e0_submit.sh built an `_n250` label
    and used it for its done-guard. The guard therefore compared against a name that could
    never exist, and the n=250 paper_retry sweep silently OVERWROTE the full-2500
    paper_retry files it was meant to sit beside. `total_images` in batch_info still records
    the truth, so affected files are identifiable after the fact, but the names lied.
    """
    label = run_label(args)
    return f"{label}_n{args.limit}" if getattr(args, "limit", 0) else label


def acceptable(accept: str, ok: bool, compliant: bool) -> bool:
    """Has this response satisfied the active acceptance test?"""
    if accept == "paper25":
        return compliant
    if accept == "parse":
        return ok
    return True


def resolve_sampling(args) -> dict:
    """Decide the decoding parameters for this run and say why, in the log.

    Three policies, because the arms are asking different questions:

      paper    -- uniform PAPER_GATEWAY_TEMPERATURE, matching what LAION's gateway
                  defaulted to. The only setting under which their retry loop means
                  anything. Required for `paper_retry`; using per-model values there
                  would confound compliance with vendor entropy defaults.
      card     -- vendor-recommended values from MODEL_SAMPLING. The fair-shot condition
                  for capability arms; several cards warn that temperature 0 causes
                  degeneration (Qwen: repetition loops; Gemma: collapse).
      greedy   -- do_sample=False. Deterministic and reproducible, and what the original
                  sweep ran. Valid for single-shot arms, invalid for anything with retries.

    Explicit --temperature/--top-p/--top-k always win over the policy.
    """
    accept = resolve_accept(args)
    if args.sampling == "greedy":
        cfg = {"temperature": 0.0}
    elif args.sampling == "paper":
        cfg = {"temperature": PAPER_GATEWAY_TEMPERATURE}
    else:
        cfg = dict(MODEL_SAMPLING.get(args.model, {}))
        if not cfg:
            print(f"[e0] WARNING: no MODEL_SAMPLING entry for {args.model}, "
                  f"falling back to temperature {PAPER_GATEWAY_TEMPERATURE}", flush=True)
            cfg = {"temperature": PAPER_GATEWAY_TEMPERATURE}
    for key, val in (("temperature", args.temperature), ("top_p", args.top_p),
                     ("top_k", args.top_k)):
        if val is not None:
            cfg[key] = val

    # A retry arm needs a decoder that can actually produce a different answer. Identical
    # prompt + greedy = identical output, so the retry budget is silently spent re-running
    # round 1. Measured on InternVL3_5-8B: rounds 2-5 took 10190/10193/10187/10171s and
    # settled 0/2500 every time -- 8.5 of 14.6 hours reproducing round 2 exactly.
    rounds = args.retries if accept != "none" else 1
    # The uniformity requirement is specific to the *reproduction*: there we measure how
    # often the 25-pair test is satisfied, so per-model entropy defaults would confound the
    # convergence rate. Retrying merely until a response parses has no such property --
    # it recovers missing data, it does not measure anything -- so vendor values are fine.
    if rounds > 1 and accept == "paper25" and args.sampling == "card":
        raise SystemExit(
            f"[e0] refusing to run: --sampling card with --accept paper25 and --retries "
            f"{args.retries}. Vendor defaults span temperature 1e-6 (Qwen2.5-VL) to 1.0 "
            f"(gemma), so per-model values would make the rejection loop converge at a rate "
            f"set by each vendor's entropy default rather than by the model's perception. "
            f"LAION applied one uniform setting to every baseline -- use --sampling paper.")
    if rounds > 1 and cfg.get("temperature", 0.0) < MIN_RETRY_TEMPERATURE:
        raise SystemExit(
            f"[e0] refusing to run: arm={args.arm} with --retries {args.retries} but "
            f"temperature={cfg.get('temperature')} < {MIN_RETRY_TEMPERATURE}. At that "
            f"temperature a retry re-sends an identical prompt to an effectively "
            f"deterministic decoder and gets an identical answer back, so the rejection "
            f"loop cannot converge and the retry budget just replays round 1. Use "
            f"--sampling paper (LAION-matched, temperature {PAPER_GATEWAY_TEMPERATURE}) "
            f"or set --retries 1.")

    print(f"[e0] sampling policy={args.sampling} accept={accept} seed={args.seed} "
          f"rounds={rounds} -> {cfg}", flush=True)
    return cfg


def make_backend(args, sampling):
    """Return generate(list[(prompt_text, PIL image)]) -> list[str].

    transformers is the default: vLLM's current wheels are built against a CUDA 13 torch
    and the A40 nodes run driver 12.9, while the pinned cu128-compatible vLLM (0.11)
    predates the 2026 model families we most want to test.
    """
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(model=args.model, trust_remote_code=True,
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_mem_util,
                  tensor_parallel_size=args.tensor_parallel,
                  limit_mm_per_prompt={"image": 1})
        tok = llm.get_tokenizer()
        # vLLM wants top_k=-1 rather than None for "unrestricted"; a bare None raises.
        sp_kwargs = {"temperature": sampling.get("temperature", 0.0),
                     "max_tokens": args.max_tokens, "seed": args.seed}
        if "top_p" in sampling:
            sp_kwargs["top_p"] = sampling["top_p"]
        if "top_k" in sampling:
            sp_kwargs["top_k"] = sampling["top_k"]
        if "repetition_penalty" in sampling:
            sp_kwargs["repetition_penalty"] = sampling["repetition_penalty"]
        sampling_params = SamplingParams(**sp_kwargs)

        def generate(items, round_idx=0, on_chunk=None):
            # on_chunk is accepted and ignored: llm.generate() is one opaque call, so there
            # is no mid-round boundary to checkpoint at. A vLLM run that hits the wall clock
            # loses the round, unlike the transformers path.
            batch = [{
                "prompt": tok.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"},
                                                  {"type": "text", "text": text}]}],
                    tokenize=False, add_generation_prompt=True),
                "multi_modal_data": {"image": shrink(img, args.max_image_pixels)},
            } for text, img in items]
            # The seed MUST advance with the round. vLLM's per-request seed makes a given
            # (prompt, seed) pair deterministic, so a fixed seed would reintroduce exactly
            # the bug this flag exists to fix: round 3 re-deriving round 2 token for token.
            sp = sampling_params
            if args.seed is not None and round_idx:
                sp = sampling_params.clone()
                sp.seed = args.seed + round_idx
            return [o.outputs[0].text for o in llm.generate(batch, sp)]

        return generate

    import torch
    import transformers
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Not every VL model registers under AutoModelForImageTextToText -- some custom
    # remote-code configs only bind AutoModel or the older Vision2Seq alias. Try the
    # specific classes first so we get the right generate() behaviour where available.
    # AutoModel is last: it happily returns a bare backbone with no .generate (that is how
    # Intern-S1-mini loaded and then died on generate), so any candidate must be rejected
    # unless it can actually generate.
    # flash_attention_2 where it is installed and the architecture supports it. Not every
    # VL checkpoint accepts the kwarg (some custom remote-code configs reject it outright),
    # so each class is tried with FA2 first and retried without -- an allow-list of model
    # ids would rot on the next release.
    attn_impls = []
    if args.attn == "flash":
        attn_impls = ["flash_attention_2", None]
    elif args.attn == "sdpa":
        attn_impls = ["sdpa", None]
    else:
        attn_impls = [None]

    model = None
    errors = []
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                     "AutoModelForCausalLM", "AutoModel"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        cand = None
        for impl in attn_impls:
            kw = {"trust_remote_code": True, "dtype": torch.bfloat16, "device_map": "auto"}
            if impl:
                kw["attn_implementation"] = impl
            try:
                cand = cls.from_pretrained(args.model, **kw)
                print(f"[e0] attn_implementation={impl or 'default'}", flush=True)
                break
            except Exception as exc:
                errors.append(f"{cls_name}/{impl or 'default'}: "
                              f"{type(exc).__name__}: {str(exc)[:140]}")
        if cand is None:
            continue
        if not hasattr(cand, "generate"):
            errors.append(f"{cls_name}: loaded {type(cand).__name__} without .generate")
            del cand
            continue
        model, _ = cand, print(f"[e0] loaded via {cls_name}", flush=True)
        break
    if model is None:
        raise RuntimeError("no AutoModel class accepted this checkpoint:\n  " + "\n  ".join(errors))
    model.eval()
    if getattr(proc, "tokenizer", None) is not None and proc.tokenizer.padding_side != "left":
        proc.tokenizer.padding_side = "left"   # decoder-only batching needs left padding

    # Build the HF generate() kwargs once. do_sample follows temperature rather than being
    # hardcoded: the original sweep pinned do_sample=False, which silently neutered every
    # retry arm and, per several vendor cards, also risks degeneration (Qwen: repetition
    # loops, Gemma: collapse) on the JSON-formatted output this task needs.
    # None = do not pass the kwarg at all (leave the template's own default alone).
    think_flag = None if args.thinking == "auto" else (args.thinking == "on")
    temp = sampling.get("temperature", 0.0)
    gen_kwargs = {"max_new_tokens": args.max_tokens}
    if temp > 0.0:
        gen_kwargs.update(do_sample=True, temperature=temp)
        for key in ("top_p", "top_k", "repetition_penalty"):
            if key in sampling:
                gen_kwargs[key] = sampling[key]
    else:
        gen_kwargs["do_sample"] = False
    print(f"[e0] generate kwargs: {gen_kwargs}", flush=True)

    @torch.inference_mode()
    def generate(items, round_idx=0, on_chunk=None):
        """`on_chunk(n_done, outs)` fires after each batch, for mid-round checkpointing.

        The seeding stays where it is -- once per round, not per chunk. Chunking the round
        from the caller instead would re-seed on every chunk and make each one replay the
        same sampling trajectory, which is the bug --sampling was added to fix.
        """
        outs = []
        t0 = time.time()
        # Same reasoning as the vLLM path: reseed per round, never per run, or every retry
        # round replays the previous one.
        if args.seed is not None and gen_kwargs.get("do_sample"):
            transformers.set_seed(args.seed + round_idx)
        for start in range(0, len(items), args.batch_size):
            # Progress every ~200 items: a full 2500-image arm takes tens of minutes and
            # without this the log is silent throughout, so a hang looks like progress.
            if start and start % 200 == 0:
                rate = start / max(time.time() - t0, 1e-9)
                eta = (len(items) - start) / max(rate, 1e-9)
                print(f"[e0]   {start}/{len(items)} imgs  {rate:.2f} img/s  eta {eta/60:.1f}m",
                      flush=True)
            chunk = items[start:start + args.batch_size]
            msgs = [[{"role": "user", "content": [
                {"type": "image", "image": shrink(img, args.max_image_pixels)},
                {"type": "text", "text": text}]}] for text, img in chunk]
            enc = apply_template(proc, msgs, think_flag).to(model.device)
            gen = model.generate(**enc, **gen_kwargs)
            trimmed = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(proc.batch_decode(trimmed, skip_special_tokens=True))
            if on_chunk is not None:
                on_chunk(len(outs), outs)
        return outs

    return generate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=["paper_retry", "paper_once", "ours"])
    ap.add_argument("--dataset", default="$DATA/emonet/emonet-face-hq")
    ap.add_argument("--assets-dir", type=Path, default=Path("evaluation"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_e0"))
    ap.add_argument("--limit", type=int, default=0, help="run on N images instead of all")
    ap.add_argument("--limit-mode", choices=["random", "head"], default="random",
                    help="how --limit selects rows. random (default) draws a deterministic "
                         "subset seeded by --seed; head takes the first N, which skews "
                         "ethnicity and prompt coverage.")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--max-image-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--retries", type=int, default=PAPER_MAX_ATTEMPTS)
    ap.add_argument("--backend", choices=["transformers", "vllm"], default="transformers",
                    help="vLLM is faster but its current wheels need a CUDA 13 driver; "
                         "these A40 nodes run 12.9, so transformers is the default.")
    ap.add_argument("--batch-size", type=int, default=8, help="transformers backend only")
    ap.add_argument("--checkpoint-every", type=int, default=200,
                    help="write the .partial checkpoint roughly every N images. 0 disables "
                         "mid-round checkpointing, leaving only the round boundaries -- which "
                         "for a single-shot arm means no checkpoint at all. Costs one JSON "
                         "write per interval.")
    ap.add_argument("--resume", choices=["auto", "off"], default="auto",
                    help="auto: continue from <run>.json.partial if it exists, skipping "
                         "images already settled. Needed on the HPC cluster, where JobRequeue=1 means "
                         "a rebooted node re-runs the job automatically and a clean restart "
                         "would silently discard hours of completed generation.")
    ap.add_argument("--vram-interval", type=float, default=5.0,
                    help="seconds between NVML peak-VRAM samples; 0 disables sampling")
    ap.add_argument("--sampling", choices=["auto", "paper", "card", "greedy"], default="auto",
                    help="decoding policy. auto: 'paper' for paper_retry (uniform "
                         "temperature %.1f, matching LAION's gateway default -- the only "
                         "setting under which their rejection loop can converge), 'card' "
                         "for the single-shot arms (vendor-recommended per-model values). "
                         "greedy reproduces the original do_sample=False behaviour and is "
                         "rejected for multi-round arms." % PAPER_GATEWAY_TEMPERATURE)
    ap.add_argument("--temperature", type=float, default=None, help="override the policy")
    ap.add_argument("--top-p", type=float, default=None, help="override the policy")
    ap.add_argument("--top-k", type=int, default=None, help="override the policy")
    ap.add_argument("--seed", type=int, default=None,
                    help="base RNG seed; each retry round uses seed+round_idx so rounds "
                         "stay reproducible without replaying each other")
    ap.add_argument("--attn", choices=["flash", "sdpa", "default"], default="sdpa",
                    help="attention kernel. Default sdpa: torch dispatches it to the "
                         "FlashAttention-2 kernel for bf16 on Ampere, so the standalone "
                         "flash-attn package buys little and will not build here (system "
                         "CUDA 13.3 vs torch cu128). flash still works if that package is "
                         "ever installed; the impl actually loaded is logged and recorded "
                         "in the output JSON so a silent fallback cannot be mistaken for a "
                         "speedup.")
    ap.add_argument("--thinking", choices=["off", "on", "auto"], default="off",
                    help="reasoning mode for hybrid-thinking checkpoints. off (default) "
                         "passes enable_thinking=False where the chat template supports it: "
                         "GLM-4.6V and Qwen3.5 otherwise spend the whole token budget on "
                         "chain-of-thought and never emit the JSON (13.6%% and 4.5%% parse "
                         "rates). auto leaves the template default untouched.")
    ap.add_argument("--accept", choices=["auto", "paper25", "parse", "none"], default="auto",
                    help="acceptance test driving the retry loop, independent of --arm. "
                         "paper25: exactly 25 non-zero pairs (LAION's test, reproduction "
                         "only). parse: retry until the response parses -- recovers missing "
                         "data without imposing a cardinality. none: single shot. "
                         "auto = paper25 for paper_retry, none otherwise.")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset)["train"]
    # A --limit run must stay representative: the dataset is not sorted, but the first 250
    # rows still under-sample ethnicity category 5 (3.6% against 7.7% overall) and cover
    # only 117 of 207 prompt variants. Compliance is a formatting behaviour and barely
    # cares, but the same predictions get kappa-scored and demographically sliced, so a
    # deterministic random subset is strictly safer and costs nothing. Indices stay TRUE
    # dataset indices so downstream joins are unaffected.
    if args.limit and args.limit < len(ds):
        if args.limit_mode == "random":
            rng = random.Random(args.seed if args.seed is not None else 0)
            indices = sorted(rng.sample(range(len(ds)), args.limit))
        else:
            indices = list(range(args.limit))
    else:
        indices = list(range(len(ds)))
    n = len(indices)
    emotions = emotion_list(ds)
    print(f"[e0] model={args.model} arm={args.arm} images={n} emotions={len(emotions)}", flush=True)

    if args.sampling == "auto":
        args.sampling = "paper" if args.arm == "paper_retry" else "card"
    sampling = resolve_sampling(args)

    assets = load_paper_assets(args.assets_dir) if args.arm != "ours" else None
    generate = make_backend(args, sampling)
    vram = VramTracker(interval=args.vram_interval)
    vram.start()

    # Round 0 for every image; extra rounds only for the paper_retry arm.
    pending = list(indices)
    results: dict[int, dict] = {}
    attempts: dict[int, int] = {i: 0 for i in pending}
    last_seen: dict[int, dict] = {}   # most recent parse per image, for checkpointing
    # Raw text and round number that produced last_seen[i]. Kept so a fallback to an earlier
    # parse can also carry the text it came from -- pairing recovered scores with the final
    # round's raw_response would make the JSON self-contradictory.
    last_seen_raw: dict[int, str] = {}
    last_seen_round: dict[int, int] = {}
    # Round-1 outcome, kept whatever happens later. Under --accept parse the retry loop
    # overwrites raw_response with the attempt that finally succeeded, which would erase
    # the single-shot condition -- and the single-shot parse rate is a headline number
    # (Qwen3.5-9B 4.5%, GLM 13.6% before --thinking off). Keeping it means ONE run yields
    # both the honest first-try statistics and the retry-recovered predictions, instead of
    # paying for a separate paper_once arm.
    first: dict[int, dict] = {}
    started = time.time()
    accept = resolve_accept(args)
    max_rounds = args.retries if accept != "none" else 1
    start_round = 0
    resumed_from = None

    # Resume. the HPC cluster reboots nodes unannounced and runs JobRequeue=1, so a killed job comes
    # back on its own; on the 6 h epyc-gpu-test partition it may also simply hit the wall.
    # Without this the requeued job restarts from image 0, overwrites its own checkpoint and
    # reports the second attempt's numbers as if nothing happened.
    ckpt_path = args.out_dir / f"{args.model.strip('/').split('/')[-1]}__{file_label(args)}_eval.json.partial"
    if args.resume != "off" and ckpt_path.is_file():
        try:
            prev = json.load(open(ckpt_path))
            rows = {r["image_index"]: r for r in prev.get("results", [])}
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"[e0] checkpoint at {ckpt_path} unreadable ({type(exc).__name__}), "
                  f"starting clean", flush=True)
            rows = {}
        still_pending = []
        for i in list(indices):
            r = rows.get(i)
            if r is None:
                still_pending.append(i)
                continue
            attempts[i] = r.get("attempts", 0)
            if r.get("first_parse_success") is not None:
                first[i] = {"parse_success": r["first_parse_success"],
                            "predicted_emotions": r.get("first_predicted_emotions")
                            or {e: 0 for e in emotions},
                            "raw_response": r.get("first_raw_response", "")}
            if r.get("parse_success"):
                last_seen[i] = r["predicted_emotions"]
                last_seen_raw[i] = r.get("raw_response", "")
                last_seen_round[i] = r.get("parse_from_round") or r.get("attempts", 1)
            if r.get("success"):
                results[i] = r          # already settled, never regenerate it
            else:
                still_pending.append(i)
        if results or any(attempts.values()):
            pending = still_pending
            # Resume at the round the unfinished images had reached. Images that were caught
            # mid-round get at most ONE extra attempt, because the round is replayed for the
            # whole pending set rather than per image. That is a real deviation from a clean
            # run and is recorded in batch_info, not swallowed.
            start_round = min((attempts[i] for i in pending), default=0)
            resumed_from = {"checkpoint": str(ckpt_path), "settled": len(results),
                            "pending": len(pending), "start_round": start_round,
                            "max_extra_attempts": 1 if any(
                                attempts[i] > start_round for i in pending) else 0}
            print(f"[e0] RESUMED from {ckpt_path.name}: {len(results)}/{n} already settled, "
                  f"{len(pending)} pending, continuing at round {start_round + 1}",
                  flush=True)

    def absorb(i: int, raw_text: str, rnd: int, still: list[int]) -> None:
        """Record one image's outcome. Either settles it into `results` or defers it.

        Split out of the round loop so it can run per BATCH rather than per round, which is
        what makes mid-round checkpointing possible.
        """
        attempts[i] = rnd + 1
        scores, raw = parse_response(raw_text, args.arm, emotions)
        ok = scores is not None
        if ok:
            last_seen[i] = scores
            last_seen_raw[i] = raw_text
            last_seen_round[i] = rnd + 1
        # `i not in first` matters on a resume. An image caught mid-round-0 is replayed at
        # rnd 0, and without the guard its restored first-attempt record would be overwritten
        # by the SECOND sample -- silently corrupting the single-shot parse rate, which is a
        # headline number (Qwen3.5-9B 4.5%, GLM 13.6% before --thinking off).
        if rnd == 0 and i not in first:
            first[i] = {"parse_success": bool(ok),
                        "predicted_emotions": scores if ok else {e: 0 for e in emotions},
                        "raw_response": raw_text[:4000]}
        # The paper's acceptance test, reproduced verbatim.
        compliant = ok and raw is not None and len(raw) == PAPER_REQUIRED_PAIRS \
            and not any(v == 0 for v in raw.values() if isinstance(v, (int, float)))
        if not acceptable(accept, ok, compliant) and rnd + 1 < max_rounds:
            still.append(i)
            return
        # Fall back to the most recent successful parse when this round did not parse.
        # Without this, an image that parsed in round 1 and failed on the LAST round was
        # written out as an all-zero, parse-failed prediction: on the final round every
        # pending image settles through this branch (`still` is guarded by
        # `rnd + 1 < max_rounds`), so the last_seen fallback below it was dead code.
        # Measured cost of the bug: MiMo-VL lost all 59 of its round-1 parses and
        # reported parsed 0/250, Ministral lost 26, gemma-4-12B 18.
        fallback = not ok and i in last_seen
        results[i] = {
            "image_index": i,
            "image_path": f"index_{i}",
            "model": args.model,
            "success": True,
            "attempts": attempts[i],
            "paper_compliant": bool(compliant),
            "predicted_emotions": (last_seen[i] if fallback else
                                   scores if ok else {e: 0 for e in emotions}),
            "parse_success": bool(ok or fallback),
            "parse_from_round": last_seen_round.get(i) if (ok or fallback) else None,
            "raw_response": (last_seen_raw[i] if fallback else raw_text)[:4000],
            "final_raw_response": raw_text[:4000] if fallback else "",
            "first_parse_success": first.get(i, {}).get("parse_success"),
            "first_predicted_emotions": first.get(i, {}).get("predicted_emotions"),
            "first_raw_response": first.get(i, {}).get("raw_response", "")[:4000],
        }

    def checkpoint(still: list[int], rest: list[int]) -> None:
        """Write every image's current state to `<run>.json.partial`.

        `still` are images deferred to a later round, `rest` are images this round has not
        reached yet. Both are written with whatever parse they last produced, so a resume
        never loses a prediction -- and neither does the final write, which is what the
        pre-2026-08-06 checkpoint did by storing `raw_response: ""` for everything pending.
        """
        ckpt = dict(results)
        for i in still + rest:
            ckpt[i] = {"image_index": i, "image_path": f"index_{i}", "model": args.model,
                       "success": False, "attempts": attempts.get(i, 0),
                       "paper_compliant": False,
                       "predicted_emotions": last_seen.get(i, {e: 0 for e in emotions}),
                       "parse_success": i in last_seen,
                       "parse_from_round": last_seen_round.get(i),
                       "raw_response": last_seen_raw.get(i, "")[:4000],
                       "first_parse_success": first.get(i, {}).get("parse_success"),
                       "first_predicted_emotions": first.get(i, {}).get("predicted_emotions"),
                       "first_raw_response": first.get(i, {}).get("raw_response", "")[:4000]}
        write_results(args, [ckpt[i] for i in sorted(ckpt)], attempts, emotions,
                      n, started, suffix=".partial", vram=vram, sampling=sampling,
                      resumed_from=resumed_from)

    for rnd in range(start_round, max_rounds):
        if not pending:
            break
        batch = []
        for i in pending:
            text = build_prompt(args.arm, emotions, assets, f"{i:04d}.png")
            if rnd > 0:
                text += PAPER_AHA
            batch.append((text, ds[i]["path"]))

        still: list[int] = []
        absorbed = 0

        def on_chunk(n_done, outs, _rnd=rnd, _still=still):
            # Absorb everything the backend has finished but we have not recorded, then
            # checkpoint. Without this a job killed by the wall clock loses the whole round --
            # which for a single-shot arm is the whole run, since max_rounds is 1.
            nonlocal absorbed
            while absorbed < n_done:
                absorb(pending[absorbed], outs[absorbed], _rnd, _still)
                absorbed += 1
            if args.checkpoint_every and absorbed % args.checkpoint_every < args.batch_size:
                checkpoint(_still, pending[absorbed:])

        outs = generate(batch, rnd, on_chunk=on_chunk if args.checkpoint_every else None)
        # Stragglers: the vLLM backend never calls on_chunk, and the last transformers batch
        # may not have tripped the checkpoint interval.
        while absorbed < len(outs):
            absorb(pending[absorbed], outs[absorbed], rnd, still)
            absorbed += 1
        pending = still
        vram.mark(f"round_{rnd + 1}")
        print(f"[e0] round {rnd + 1}: {len(results)}/{n} settled, {len(pending)} retrying "
              f"({time.time() - started:.0f}s) vram_peak {vram.peak_mib / 1024:.1f} GB",
              flush=True)

        # Checkpoint at every round boundary as well as mid-round, so a resume never has to
        # replay a completed round.
        if pending:
            checkpoint(pending, [])

    # Retries exhausted. Keep the last parse rather than discarding it: under the paper's
    # acceptance test almost nothing ever complies, so treating non-compliant answers as
    # empty would throw away every prediction these models actually made.
    # Unreachable in practice -- the loop settles everything on its final round -- but kept
    # as the correct behaviour if the settle condition is ever loosened.
    for i in pending:
        results[i] = {
            "image_index": i, "image_path": f"index_{i}", "model": args.model,
            "success": False, "attempts": attempts[i], "paper_compliant": False,
            "predicted_emotions": last_seen.get(i, {e: 0 for e in emotions}),
            "parse_success": i in last_seen,
            "parse_from_round": last_seen_round.get(i),
            "raw_response": last_seen_raw.get(i, "")[:4000],
            "first_parse_success": first.get(i, {}).get("parse_success"),
            "first_predicted_emotions": first.get(i, {}).get("predicted_emotions"),
            "first_raw_response": first.get(i, {}).get("raw_response", "")[:4000],
        }

    ordered = [results[i] for i in sorted(results)]
    vram.stop()
    out_path, parsed, compliant, mean_nz = write_results(
        args, ordered, attempts, emotions, n, started, vram=vram, sampling=sampling,
        resumed_from=resumed_from)
    tag = args.model.strip("/").split("/")[-1]
    (args.out_dir / f"{tag}__{file_label(args)}_eval.json.partial").unlink(missing_ok=True)
    print(f"[e0] DONE {out_path}\n"
          f"     parsed {parsed}/{n} | paper-compliant {compliant}/{n} | "
          f"mean nonzero emotions {mean_nz:.2f} | {time.time() - started:.0f}s", flush=True)
    fp = next((r for r in ordered if r.get("first_parse_success") is not None), None)
    if fp is not None:
        f1 = sum(1 for r in ordered if r.get("first_parse_success"))
        print(f"     first-attempt parsed {f1}/{n} ({f1 / max(n, 1):.1%}) | "
              f"recovered by retry {parsed - f1}", flush=True)
        # Images whose kept prediction came from an earlier round than the last one they ran.
        # Before the last_seen fallback these were silently written out as all-zero.
        held = sum(1 for r in ordered
                   if r.get("parse_success") and r.get("parse_from_round")
                   and r["parse_from_round"] < r.get("attempts", 1))
        print(f"     kept from an earlier round {held}/{n} "
              f"(would have been zeroed before the last_seen fallback)", flush=True)
    # Grep-able single line: `grep VRAM_PEAK envlogs/*.out` gives the whole sweep's sizing
    # table, so e0_models.txt column 3 can be retuned without reopening the JSONs.
    v = vram.summary()
    if v["available"]:
        print(f"[e0] VRAM_PEAK {args.model} {args.arm} peak={v['peak_gb']}GB "
              f"after_load={v['after_load_gb']}GB requested={v['requested_shard_gb']}GB "
              f"recommend_shard={v['recommended_shard_gb']}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
