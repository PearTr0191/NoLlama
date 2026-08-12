"""Invoke optimum-cli's parser directly (AV blocks pip's .exe launcher stubs).

Usage: python run-export.py <src_model_dir> <out_dir>
Recipe: optimum-intel PR #1924 (Muse Glimmer int4).
"""
import sys

src, out = sys.argv[1], sys.argv[2]
sys.argv = [
    "optimum-cli", "export", "openvino",
    "-m", src, out,
    "--weight-format", "int4",
    "--task", "image-text-to-text",
    "--group-size", "64",
    "--group-size-fallback", "ignore",
]

# transformers 5.16.0.dev0 materializes checkpoint tensors in a thread pool
# (GLOBAL_WORKERS=4, no env knob). Parallel mmap copies of the 60 GB shards
# die with a native access violation (0xC0000005) on Windows under memory
# pressure — observed on 32 GB. Serialize: slower load, but it survives.
import transformers.core_model_loading as _cml

_cml.GLOBAL_WORKERS = 1

from optimum.commands.optimum_cli import main

main()
