r"""Does the OpenVINO GPU plugin diverge from CPU under greedy decoding?

Answers the open question left by openvino#37419: Glimmer's GPU and CPU
traces agree for ~40 tokens then diverge into different-but-coherent text.
That divergence is reproducible (repeat runs are byte-identical per device)
but UNEXPLAINED. If a known-good model diverges the same way, it is ordinary
cross-plugin numerics. If known-good models match exactly, Glimmer's
divergence is suspicious and worth reporting as a residue.

Compares TOKEN IDS, not decoded text — a text diff hides single-token
changes that decode to the same-looking string.

Run the same script on both models so the numbers are comparable:

  .\venv-optimum-nightly\Scripts\python.exe scripts\device-divergence.py
  .\venv-optimum-nightly\Scripts\python.exe scripts\device-divergence.py ~\models\Muse-Glimmer-30B-int4-ov

  usage: device-divergence.py [model_dir] [max_new_tokens]

Glimmer takes minutes on the CPU half (~1 tok/s); SmolLM3 takes seconds.
"""
import gc
import os
import sys

DEFAULT_MODEL = os.path.expanduser(r"~/models/SmolLM3-3B-int4-cw-ov")
MODEL = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 200

# Open-ended enough to generate a few hundred tokens, with no single
# obviously-correct continuation — the case most likely to sit near an
# argmax tie, which is where a cross-plugin difference would first show.
PROMPT = "Explain, in a few sentences, why the sky appears blue."

import openvino
import transformers
from transformers import AutoTokenizer


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


try:
    from importlib.metadata import version as _v
    optimum_ver = _v("optimum-intel")
except Exception:
    optimum_ver = "unknown"

if not os.path.isdir(MODEL):
    sys.exit(f"model dir not found: {MODEL}")

vlm = is_vlm(MODEL)
if vlm:
    from optimum.intel import OVModelForVisualCausalLM as OVModel
else:
    from optimum.intel import OVModelForCausalLM as OVModel

print("=" * 72)
print(f"openvino       {openvino.__version__}")
print(f"transformers   {transformers.__version__}")
print(f"optimum-intel  {optimum_ver}")
print(f"model          {MODEL}")
print(f"class          {OVModel.__name__}  ({'VLM' if vlm else 'LLM'} export)")
print(f"max_new_tokens {MAX_NEW}   greedy (do_sample=False, num_beams=1)")
print("=" * 72)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.chat_template:
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True)
else:
    inputs = tok(PROMPT, return_tensors="pt")
prompt_len = inputs["input_ids"].shape[1]
prompt_ids = inputs["input_ids"][0].tolist()
print(f"prompt         {prompt_len} tokens\n")

results = {}
for device in ("GPU", "CPU"):
    print(f"--- loading on {device} ---", flush=True)
    model = OVModel.from_pretrained(MODEL, device=device)
    runs = []
    # Twice on the same load: proves generate() itself is deterministic, so a
    # GPU-vs-CPU difference cannot be dismissed as run-to-run noise.
    for i in (1, 2):
        out = model.generate(**inputs, max_new_tokens=MAX_NEW,
                             do_sample=False, num_beams=1)
        seq = out[0].tolist()
        # Most classes return prompt + continuation, but not all do. Decide by
        # checking whether the sequence actually starts with the prompt rather
        # than guessing from the length — a run that stops early would fool a
        # length heuristic and silently shift every index that follows.
        if seq[:prompt_len] == prompt_ids:
            seq = seq[prompt_len:]
        runs.append(seq)
        print(f"    run {i}: {len(seq)} new tokens", flush=True)
    same = runs[0] == runs[1]
    print(f"    {device} run1 == run2: {same}"
          f"{'' if same else '   <-- NOT DETERMINISTIC, results below are meaningless'}\n")
    results[device] = runs[0]
    del model
    gc.collect()

gpu, cpu = results["GPU"], results["CPU"]
n = min(len(gpu), len(cpu))
first = next((i for i in range(n) if gpu[i] != cpu[i]), None)

def describe(seq, idx):
    """'<past end>' when one device stopped earlier than the other."""
    if idx >= len(seq):
        return "<past end>"
    return f"{seq[idx]} ({tok.decode([seq[idx]])!r})"

print("=" * 72)
print(f"GPU produced {len(gpu)} tokens, CPU produced {len(cpu)}.")
if first is None and len(gpu) == len(cpu):
    print(f"IDENTICAL: GPU and CPU agree on all {n} tokens.")
    print("=> This model shows NO cross-plugin divergence. If Glimmer still")
    print("   diverges under the same procedure, that is not 'just numerics'")
    print("   and the openvino#37419 observation deserves a closer look.")
else:
    # first is None here only when one run is a strict prefix of the other.
    idx = n if first is None else first
    print(f"DIVERGES at generated token index {idx} (of {n} compared).")
    print(f"  GPU: {describe(gpu, idx)}")
    print(f"  CPU: {describe(cpu, idx)}")
    print("  agreed prefix, last 20 tokens:")
    print(f"    ...{tok.decode(gpu[max(0, idx - 20):idx])!r}")
    print("=> Cross-plugin divergence happens on a known-good model too, which")
    print("   is the ordinary-numerics explanation for Glimmer's. Compare the")
    print("   index: much later here than Glimmer's ~40 would still be notable.")
print("=" * 72)
