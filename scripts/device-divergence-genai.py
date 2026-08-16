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
import json
import os
import subprocess
import sys
import tempfile

# --worker DEVICE OUTFILE MODEL MAX_NEW  — internal, one generation per process.
WORKER = len(sys.argv) > 1 and sys.argv[1] == "--worker"
if WORKER:
    _, _, W_DEVICE, W_OUT, W_MODEL, W_MAX = sys.argv[:6]

DEFAULT_MODEL = os.path.expanduser(r"~/models/SmolLM3-3B-int4-cw-ov")
if WORKER:
    MODEL, MAX_NEW = W_MODEL, int(W_MAX)
else:
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

if WORKER:
    # One generation, then this process dies and the OS reclaims everything.
    pipe = (ovg.VLMPipeline if vlm else ovg.LLMPipeline)(MODEL, device=W_DEVICE)
    cfg = ovg.GenerationConfig()
    cfg.max_new_tokens = MAX_NEW
    cfg.do_sample = False
    res = pipe.generate(PROMPT, cfg) if not vlm else pipe.generate(PROMPT, generation_config=cfg)
    text = res.texts[0] if hasattr(res, "texts") else str(res)
    with open(W_OUT, "w", encoding="utf-8") as fh:
        json.dump({"text": text}, fh)
    sys.exit(0)

print("=" * 72)
print(f"openvino        {openvino.__version__}")
print(f"openvino-genai  {ovg.__version__}")
print(f"model           {MODEL}")
print(f"pipeline        {'VLMPipeline' if vlm else 'LLMPipeline'}")
print(f"max_new_tokens  {MAX_NEW}   greedy (do_sample=False)")
print("isolation       one subprocess per generation")
print("=" * 72)


def generate_once(device):
    """One greedy generation, in its own PROCESS.

    Two reasons, one methodological and one that cost a machine.

    Methodological: two generate() calls on one pipeline instance are not
    independent. Measured 2026-08-15 on a B60 — reusing a pipeline gave
    non-identical GPU runs (996 vs 989 chars) while CPU repeated exactly,
    the exact shape of state carryover. A fresh pipeline per run separates
    "is the plugin deterministic" from "is the pipeline stateless".

    Practical: `del pipe; gc.collect()` does NOT unload the GPU plugin from
    the process. Its allocations survive, and WDDM demotes them from
    dedicated to the shared segment rather than freeing them — ~13.7 GB
    still held, which is real and worth avoiding on a memory-tight box. A
    subprocess per generation is the only reliable teardown: when it exits,
    the OS reclaims everything, plugin included.

    CORRECTION (2026-08-16): an earlier version of this note, and the commit
    that introduced subprocess isolation, blamed that retention for crashing
    the test workstation mid-run. That was wrong. The same machine later
    went down while merely installing an npm package — no GPU, no model, no
    memory pressure — so the crashes are a hardware fault on that box, not
    this script. Two coincidences that made the wrong story fit: the crash
    landed at the GPU-to-CPU handover twice, and the machine had a 2 GB
    pagefile (commit limit ~34 GB), which made memory exhaustion look
    plausible and also explains why no crash dump was ever written. The
    isolation below is still correct on its own merits; the attribution was
    not. Reproducibility at the same point is not evidence of cause.
    """
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "result.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               device, out, MODEL, str(MAX_NEW)]
        proc = subprocess.run(cmd)
        if proc.returncode != 0 or not os.path.exists(out):
            sys.exit(f"worker for {device} failed (exit {proc.returncode}) — "
                     f"if this is an out-of-memory kill, lower max_new_tokens "
                     f"or use a smaller model.")
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)["text"]


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

    def window(ids, i, before=12, after=12):
        """Decode a TOKEN window. Slicing the text by a token index (the
        earlier bug here) shows the same characters for both devices and
        makes a real divergence look like none at all."""
        lo, hi = max(0, i - before), min(len(ids), i + after)
        return tokzr.decode(ids[lo:hi])

    def tok_at(ids, i):
        return f"{ids[i]} ({tokzr.decode([ids[i]])!r})" if i < len(ids) else "<past end>"

    print(f"DIVERGES at re-encoded token index {idx} (of {n} compared).")
    print(f"  agreed up to there: ...{tokzr.decode(gpu_ids[max(0, idx - 12):idx])!r}")
    print(f"  GPU token {idx}: {tok_at(gpu_ids, idx)}")
    print(f"  CPU token {idx}: {tok_at(cpu_ids, idx)}")
    print(f"  GPU continues: {window(gpu_ids, idx)!r}")
    print(f"  CPU continues: {window(cpu_ids, idx)!r}")
    print("=> The plugin does differ across devices on a known-good model,")
    print("   which is the ordinary-numerics reading of Glimmer's behaviour.")
    print("   Glimmer diverged around token 40 — compare that with the index")
    print("   above before concluding anything about it.")
print("=" * 72)
