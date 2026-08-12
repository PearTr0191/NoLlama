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

from optimum.commands.optimum_cli import main

main()
