"""Test OpenVINO's MoE disk offload (OFFLOAD_RATIO) on a GPU.

Usage:
    python scripts/offload-test.py <ratio> <model_dir> [device]

Run twice — ratio 0, then 90 — and compare `mem usm_device`:
  - drops sharply at 90  -> offload engaged (XMX GPU, fusable MoE IR)
  - identical            -> offload not engaged (most likely no XMX:
    OPTIMIZATION_CAPABILITIES must list GPU_HW_MATMUL — see TODONT.md)

Requires openvino + openvino-genai >= 2026.3. The property only ever
applies to MoE expert weights; dense models are unaffected.
"""
import sys, time, traceback

import openvino as ov
import openvino_genai as og

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(2)

RATIO = int(sys.argv[1])
MODEL = sys.argv[2]
DEVICE = sys.argv[3] if len(sys.argv) > 3 else "GPU"

print(f"genai {og.__version__}, {DEVICE}, OFFLOAD_RATIO={RATIO}", flush=True)
t0 = time.time()
try:
    if RATIO > 0:
        pipe = og.LLMPipeline(MODEL, DEVICE, OFFLOAD_RATIO=RATIO)
    else:
        pipe = og.LLMPipeline(MODEL, DEVICE)
    print(f"OK: loaded in {time.time()-t0:.1f}s", flush=True)
except Exception:
    print(f"LOAD FAILED after {time.time()-t0:.1f}s:", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    stats = ov.Core().get_property(DEVICE, "GPU_MEMORY_STATISTICS")
    for k, v in sorted(stats.items()):
        print(f"  mem {k}: {v/2**30:.2f} GB", flush=True)
except Exception as e:
    print(f"  (memory statistics unavailable: {e})", flush=True)

cfg = og.GenerationConfig()
cfg.max_new_tokens = 64
t1 = time.time()
try:
    out = pipe.generate("Say hello in one short sentence.", cfg)
    dt = time.time() - t1
    print(f"OK: 64 tokens in {dt:.1f}s -> {64/dt:.1f} tok/s", flush=True)
    print("OUTPUT:", str(out)[:200], flush=True)
    print(f"RESULT: ratio={RATIO} load+generate succeeded", flush=True)
except Exception:
    print(f"GENERATION FAILED after {time.time()-t1:.1f}s:", flush=True)
    traceback.print_exc()
    sys.exit(1)
