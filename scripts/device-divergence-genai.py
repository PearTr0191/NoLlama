r"""Does the OpenVINO GPU plugin diverge from CPU? — openvino_genai edition.

Companion to device-divergence.py, which drives the optimum backend. That
one is currently unusable as a control: transformers main (5.16.0.dev0) is
incompatible with optimum-intel in two separate places, and BOTH bite before
a single token is generated —

  OVModelForCausalLM        _optimize_model_for_decode() calls
                            get_experts_implementation(), not implemented
  OVModelForVisualCausalLM  the Qwen-VL prepare_inputs_for_generation()
                            does cache_position[0] on a None

Glimmer happens to sit on a code path that survives both, which is the only
reason it runs. Patching around either one would mean monkey-patching the
input preparation of the thing being measured — not acceptable in a
measurement.

openvino_genai has no transformers dependency, so it sidesteps all of that
and still exercises the same GPU plugin. The trade-off is stated plainly:
this measures the PLUGIN's CPU-vs-GPU behaviour through a different runtime
than Glimmer used. If the plugin is numerically identical across devices
here, that is evidence about the plugin, not proof about the optimum path.

  .\venv-nightly\Scripts\python.exe scripts\device-divergence-genai.py
  .\venv-nightly\Scripts\python.exe scripts\device-divergence-genai.py <model_dir> [max_new_tokens]

Use venv-nightly (OpenVINO 2026.4 nightly + genai), not venv-optimum*.
"""
import gc
import os
import sys

DEFAULT_MODEL = os.path.expanduser(r"~/models/SmolLM3-3B-int4-cw-ov")
MODEL = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 200

PROMPT = "Explain, in a few sentences, why the sky appears blue."

import openvino
import openvino_genai as ovg


def is_vlm(model_dir):
    """Mirror nollama.py: a VLM export ships separate vision-encoder IRs."""
    try:
        for fn in os.listdir(model_dir):
            low = fn.lower()
            if low.endswith(".xml") and ("vision" in low or "image_encoder" in low):
                return True
    except OSError:
        pass
    return False


if not os.path.isdir(MODEL):
    sys.exit(f"model dir not found: {MODEL}")

vlm = is_vlm(MODEL)
print("=" * 72)
print(f"openvino        {openvino.__version__}")
print(f"openvino-genai  {ovg.__version__}")
print(f"model           {MODEL}")
print(f"pipeline        {'VLMPipeline' if vlm else 'LLMPipeline'}")
print(f"max_new_tokens  {MAX_NEW}   greedy (do_sample=False)")
print("=" * 72)


def generate_once(device):
    """One greedy generation from a FRESHLY constructed pipeline.

    Rebuilding the pipeline per run is deliberate. Two generate() calls on a
    single pipeline instance are not necessarily independent — if any KV or
    internal state survives between calls, run 2 starts from a different
    place and the output differs for reasons that have nothing to do with
    plugin numerics. Measured 2026-08-15 on a B60: reusing one pipeline gave
    non-identical GPU runs (996 vs 989 chars) while CPU repeated exactly,
    which is precisely the shape a state-carryover bug would take. A fresh
    pipeline per run is the only way to isolate "is the plugin
    deterministic" from "is the pipeline stateless".
    """
    pipe = (ovg.VLMPipeline if vlm else ovg.LLMPipeline)(MODEL, device=device)
    cfg = ovg.GenerationConfig()
    cfg.max_new_tokens = MAX_NEW
    cfg.do_sample = False
    res = pipe.generate(PROMPT, cfg) if not vlm else pipe.generate(PROMPT, generation_config=cfg)
    text = res.texts[0] if hasattr(res, "texts") else str(res)
    del pipe
    gc.collect()
    return text


def run(device):
    """Two greedy generations, each from its own pipeline."""
    print(f"--- {device} ---", flush=True)
    outs = []
    for i in (1, 2):
        outs.append(generate_once(device))
        print(f"    run {i} (fresh pipeline): {len(outs[-1])} chars", flush=True)
    same = outs[0] == outs[1]
    print(f"    {device} run1 == run2: {same}"
          f"{'' if same else '   <-- NOT DETERMINISTIC even with a fresh pipeline'}\n")
    return outs[0], same


gpu_text, gpu_det = run("GPU")
cpu_text, cpu_det = run("CPU")

# genai returns decoded text, not ids, so re-encode to report a TOKEN index
# rather than a character offset — a character offset makes a one-token
# substitution look bigger or smaller than it is depending on word length.
tokzr = ovg.Tokenizer(MODEL)
gpu_ids = list(tokzr.encode(gpu_text).input_ids.data.flatten())
cpu_ids = list(tokzr.encode(cpu_text).input_ids.data.flatten())
n = min(len(gpu_ids), len(cpu_ids))
first = next((i for i in range(n) if gpu_ids[i] != cpu_ids[i]), None)

print("=" * 72)
if not (gpu_det and cpu_det):
    print("INCONCLUSIVE for the divergence question — but note WHICH device.")
    print("Each run built a fresh pipeline, so state carryover between")
    print("generate() calls is already excluded. A device that still differs")
    print("from itself is non-deterministic under greedy decoding, which is a")
    print("bigger finding than the CPU-vs-GPU question this script asks.")
elif gpu_text == cpu_text:
    print(f"IDENTICAL: GPU and CPU produced the same {len(gpu_ids)} tokens.")
    print("=> The GPU plugin shows NO divergence from CPU on this model.")
    print("   If Glimmer still diverges at ~40 under the optimum path, that")
    print("   is not simply 'different backends differ' and openvino#37419")
    print("   deserves a follow-up.")
else:
    idx = n if first is None else first
    print(f"DIVERGES at re-encoded token index {idx} (of {n} compared).")
    print(f"  GPU: ...{gpu_text[max(0, idx - 60):idx + 60]!r}")
    print(f"  CPU: ...{cpu_text[max(0, idx - 60):idx + 60]!r}")
    print("=> The plugin does differ across devices on a known-good model,")
    print("   which is the ordinary-numerics reading of Glimmer's behaviour.")
print("=" * 72)
