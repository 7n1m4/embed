#!/usr/bin/env bash
# Wrapper script to launch the embedding service.
#
# Inference runs on a Vulkan GPU via ncnn's Vulkan backend, pinned by GPU name
# (see README.md "Why ncnn Vulkan"). Vulkan rather than CUDA is what makes this
# work on old NVIDIA and Radeon cards that CUDA no longer supports.
#
# --device vulkan is the default. If the GPU is unavailable or produces bad
# results for any reason, fall back with --device cpu (numerically identical,
# just slower and on the CPU).
#
# KEEP_ALIVE: idle seconds before the model is unloaded from the device
# (Ollama-style, default 300). An idle service holds no VRAM; the first request
# after an idle gap reloads the model. -1 keeps it resident for the life of
# the process, 0 unloads immediately after each request.

set -euo pipefail
cd "$(dirname "$0")"

exec python embedding_service.py --device "${DEVICE:-vulkan}" \
  --keep-alive "${KEEP_ALIVE:-300}"
