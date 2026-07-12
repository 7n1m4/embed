# AnimaEmbed: Embedding Service on GTX 1050

This project provides an OpenAI-compatible embedding service using `microsoft/unixcoder-base` model on a GTX 1050 (2GB VRAM) via ONNX Runtime CUDA. It is designed to work in dual-GPU setups, ensuring the target GPU (GTX 1050) is used exclusively while others (like a GTX 1060) remain idle.

## Prerequisites

1.  **Arch Linux**: With `python` (3.8+), `pip`.
2.  **CUDA Toolkit**: Installed and compatible with `onnxruntime-gpu`.
3.  **`onnxruntime-gpu`**: Installed matching your CUDA version.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Identify GPU Index**
   Find your GTX 1050's device index:
   ```bash
   nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader
   ```
   Note the index (e.g., `1`).

3. **Export Model**
   Convert the model to ONNX (CPU-only step):
   ```bash
   python export_model.py
   ```

4. **Run Service**
   Start the service (replace `N` with your GPU index):
   ```bash
   python embedding_service.py --gpu-index N
   ```

## Usage

The service is OpenAI-compatible at `http://localhost:8000/v1/embeddings`.

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "def hello(): print(\"world\")", "model": "unixcoder-base"}'
```

## Production

Use `run-embed-service.sh` or the provided systemd unit template in the runbook for persistent execution.
