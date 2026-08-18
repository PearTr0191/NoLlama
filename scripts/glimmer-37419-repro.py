r"""Re-run the openvino#37419 repro on a newer OpenVINO.

Identical in shape to the script in the issue body, so its output can be
diffed against what is already posted there. Two changes only:
  - loads from a LOCAL model dir (no 17 GB re-download)
  - prints the OpenVINO / transformers / optimum-intel versions, because
    "openvino nightly" is not a reproducible statement in a bug report

Greedy decoding (do_sample=False) so both devices are deterministic. Run
with the venv you want to test:

  .\venv-optimum-nightly\Scripts\python.exe glimmer-37419-repro.py
  .\venv-optimum\Scripts\python.exe glimmer-37419-repro.py        # 2026.3 control
"""
import gc
import os
import sys

MODEL = os.path.expanduser(r"~/models/Muse-Glimmer-30B-int4-ov")

import openvino
import transformers
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoTokenizer

try:
    from importlib.metadata import version as _v
    optimum_ver = _v("optimum-intel")
except Exception:
    optimum_ver = "unknown"

print("=" * 70)
print(f"openvino       {openvino.__version__}")
print(f"transformers   {transformers.__version__}")
print(f"optimum-intel  {optimum_ver}")
print(f"model          {MODEL}")
print("=" * 70)

if not os.path.isdir(MODEL):
    sys.exit(f"model dir not found: {MODEL}")

tok = AutoTokenizer.from_pretrained(MODEL)
inputs = tok.apply_chat_template(
    [{"role": "user", "content": 'Respond only with the text "HELLO!"'}],
    add_generation_prompt=True, return_tensors="pt", return_dict=True)
prompt_len = inputs["input_ids"].shape[1]

for device in ("GPU", "CPU"):
    print(f"\n=== {device} ===", flush=True)
    model = OVModelForVisualCausalLM.from_pretrained(MODEL, device=device)
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    text = tok.decode(out[0][prompt_len:], skip_special_tokens=False)
    print(text)
    print(f"--- {out.shape[1] - prompt_len} new tokens ---")
    del model
    gc.collect()
