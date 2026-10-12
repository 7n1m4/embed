"""OpenAI-compatible (/v1/embeddings) service for microsoft/unixcoder-base.

Inference runs through **ncnn** (see ncnn_backend.py). CUDA used to be the
backend, but the 1050 is Pascal (sm_61) and the installed driver only exposes
CUDA 13.0, which dropped Maxwell/Pascal/Volta support, so that path is closed
for good.

**Device: the GTX 1050 via ncnn's Vulkan backend.** Both GPUs expose Vulkan
1.4, so the device is pinned by *name* and index 0 is refused outright -- the
GTX 1060 used for desktop and gaming is never attached. Two config values are
load-bearing and must be set before ``load_param()``:

1. fp16 off, all three flags. The Vulkan ``MultiHeadAttention`` path rounds
   Q/K to fp16 before accumulation, which put embeddings at cosine ~0.16
   against torch. With fp16 off it is 0.999981+.
2. Each sample fed at its *own* real length, never padded to the batch max.
   Padding input to the graph drops the GPU to ~0.75 while CPU tolerates it.

``--device cpu`` remains available as a numerically equivalent fallback.

VRAM / RAM: unixcoder-base is ~124M params (~500 MB fp32) and ncnn keeps a
~374 MiB working set on the GPU. Do NOT raise uvicorn workers above 1: each
worker would load its own copy of the model.
"""
import os
import threading

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from ncnn_backend import (
    NcnnBackendError,
    NcnnEmbeddingBackend,
    PAD_TOKEN_ID,
)

GPU_NAME = "GTX 1050"
HIDDEN_SIZE = 768

HERE = os.path.dirname(os.path.abspath(__file__))
# fp32 variant: the embeddings are compared by cosine distance, so arithmetic
# fidelity matters more than the small speed win from fp16.
PARAM_PATH = os.path.join(HERE, "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.param")
BIN_PATH = os.path.join(HERE, "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.bin")

# The GTX 1050 through ncnn Vulkan. Both fixes this depends on live in
# ncnn_backend.py: fp16 disabled before load_param(), and per-sample
# unpadded lengths. Changing either reintroduces cosine-0.16 output.
DEFAULT_DEVICE = "vulkan"

# Mutable so `--device` can be applied before the backend is first built.
_device = {"name": DEFAULT_DEVICE}


def set_device(name: str) -> None:
    """Select the inference device. Call before the first request."""
    _device["name"] = name

app = FastAPI()

# The backend is created lazily on first use (see ensure_backend) and is shared
# by every request; a lock guards both creation and shutdown.
_backend_lock = threading.Lock()
_backend: NcnnEmbeddingBackend | None = None
_backend_error: str | None = None


def ensure_backend() -> NcnnEmbeddingBackend:
    """Return the shared ncnn backend, loading the model on first call."""
    global _backend, _backend_error

    if _backend is not None:
        return _backend

    with _backend_lock:
        if _backend is not None:
            return _backend
        if _backend_error is not None:
            raise HTTPException(status_code=503, detail=_backend_error)
        try:
            _backend = NcnnEmbeddingBackend(
                param_path=PARAM_PATH,
                bin_path=BIN_PATH,
                use_vulkan=_device["name"] == "vulkan",
            )
        except (NcnnBackendError, OSError) as exc:
            _backend_error = str(exc)
            raise HTTPException(status_code=503, detail=f"embedding backend unavailable: {exc}") from exc
        return _backend


@app.on_event("shutdown")
def _shutdown_backend() -> None:
    global _backend
    with _backend_lock:
        if _backend is not None:
            _backend.close()
            _backend = None


# ---- OpenAI-shaped models -------------------------------------------------

class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "unixcoder-base"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict


# ---- Pooling --------------------------------------------------------------
# The worker returns last_hidden_state as a NUMPY array, so pooling is pure
# numpy (no torch tensors after conversion).

def mean_pooling(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    # attention_mask is a 0/1 (batch, seq) mask: 1 = real token, 0 = pad.
    # Mean over real tokens only, matching the reference pooling in
    # export_model.py (mask.sum(-1) == (input_ids != pad_id).sum(-1) here).
    mask = attention_mask.astype(np.bool_)[:, :, None]
    # Zero out padding token contributions, sum over the sequence axis.
    sum_embeddings = np.sum(last_hidden_state * mask, axis=1)
    # Avoid division by zero on empty sequences.
    sum_mask = np.clip(mask.sum(axis=1), 1e-9, None)
    return sum_embeddings / sum_mask   # (batch, hidden)


# ---- Endpoint -------------------------------------------------------------
# NOTE: declared `def` (not `async def`) on purpose so FastAPI runs it in its
# threadpool. Inference is blocking work; an `async def` handler would block the
# event loop and serialize all concurrent clients.

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    texts = [req.input] if isinstance(req.input, str) else req.input

    if not texts or not all(isinstance(t, str) and t for t in texts):
        raise HTTPException(status_code=400, detail="input must be a non-empty string or list of non-empty strings")

    if not hasattr(app, "tokenizer"):
        app.tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    tokenizer = app.tokenizer

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")

    backend = ensure_backend()
    try:
        last_hidden = backend.last_hidden_state(inputs["input_ids"], inputs["attention_mask"])
    except NcnnBackendError as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    pooled = mean_pooling(last_hidden, inputs["attention_mask"])

    embeddings = pooled.tolist()
    data = [EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)]

    # Count real tokens only, never padding.
    prompt_tokens = int(inputs["attention_mask"].sum())
    usage = {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}

    return EmbeddingResponse(data=data, model=req.model, usage=usage)


# ---- Entry point ----------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=f"OpenAI-compatible embeddings via ncnn (target GPU: {GPU_NAME})")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="bind address (default 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="bind port (default 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="uvicorn workers; each loads its own copy of the model into VRAM (default 1, do not raise)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "vulkan"],
        default=DEFAULT_DEVICE,
        help=("inference device (default: %(default)s). ncnn's Vulkan "
              "MultiHeadAttention computes this graph wrongly -- cosine ~0.16 "
              "vs torch -- so 'cpu' is the accurate default"),
    )
    args = parser.parse_args()

    if args.workers != 1:
        raise SystemExit("--workers must be 1: each worker loads its own copy of the model into VRAM")

    # Must be applied before the backend is first built by a request handler.
    set_device(args.device)

    # Fail fast and loudly if the backend is broken, instead of serving 503s.
    ensure_backend()

    # Single worker by design: see the VRAM note in the module docstring.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
