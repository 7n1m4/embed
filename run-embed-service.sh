#!/usr/bin/env bash
# Wrapper script to launch the embedding service on the GTX 1050.
# 1) Determine the CUDA device index via: 
#      nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader
#    Pick the index where name contains "GTX 1050".
# 2) Edit GPU_INDEX below to that number.
# 3) Run this script as the user that owns the working directory.

GPU_INDEX=1  # <--- replace with your GTX 1050's device index

exec python embedding_service.py --gpu-index "$GPU_INDEX"
