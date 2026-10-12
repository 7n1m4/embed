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

set -euo pipefail
cd "$(dirname "$0")"

exec python embedding_service.py --device "${DEVICE:-vulkan}"
