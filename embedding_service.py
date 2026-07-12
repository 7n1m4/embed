"""OpenAI-compatible (/v1/embeddings) service for microsoft/unixcoder-base.

Runs on a SPECIFIC NVIDIA GPU chosen at startup via --gpu-index, which sets
CUDA_VISIBLE_DEVICES before any CUDA/ORT initialization. This makes only that
GPU visible to the process so we cannot accidentally run on a different card
(e.g. a GTX 1060 used for desktop/gaming).

VRAM budget: unixcoder-base is ~124M params (~500 MB fp32). Inference with ORT
CUDA easily fits in a 2 GB card. Do NOT increase uvicorn workers above 1: each
worker loads its own copy of the model into VRAM.
"""
import os
import argparse

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

app = FastAPI()


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
# ORTModelForFeatureExtraction returns last_hidden_state as a NUMPY array, not
# a torch tensor. Pooling must therefore be pure numpy.

def mean_pooling(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    # attention_mask: (batch, seq) -> (batch, seq, 1)
    mask = attention_mask[:, :, None]
    # Zero out padding token contributions, sum over the sequence axis.
    sum_embeddings = np.sum(last_hidden_state * mask, axis=1)
    # Avoid division by zero on empty sequences.
    sum_mask = np.clip(mask.sum(axis=1), 1e-9, None)
    return sum_embeddings / sum_mask   # (batch, hidden)


# ---- Endpoint -------------------------------------------------------------
# NOTE: declared `def` (not `async def`) on purpose so FastAPI runs it in its
# threadpool. ORT inference is blocking CPU/GPU work; in an `async def` handler
# it would block the event loop and serialize all concurrent clients.

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    texts = [req.input] if isinstance(req.input, str) else req.input

    if not hasattr(app, "tokenizer"):
        app.tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    tokenizer = app.tokenizer

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")

    if not hasattr(app, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded")

    outputs = app.model(**inputs)
    last_hidden = outputs.last_hidden_state          # numpy (batch, seq, hidden)
    pooled = mean_pooling(last_hidden, inputs["attention_mask"])

    embeddings = pooled.tolist()
    data = [EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)]

    # Corrected: use attention_mask.sum() for true token count, not padding tokens.
    prompt_tokens = int(inputs["attention_mask"].sum())
    usage = {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}

    return EmbeddingResponse(data=data, model=req.model, usage=usage)


# ---- Entry point ----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-index",
        type=int,
        required=True,
        help="CUDA device index of the target GPU (the GTX 1050). Find it with: "
             "nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader",
    )
    args = parser.parse_args()

    # Enforce GPU isolation BEFORE any CUDA/ORT import that initializes CUDA.
    # After this, the process sees only the chosen GPU (re-indexed to 0).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    app.model = ORTModelForFeatureExtraction.from_pretrained(
        "./unixcoder-onnx",
        provider="CUDAExecutionProvider",
    )

    import uvicorn
    # workers=1 default is intentional and mandatory: each worker loads the full
    # model into VRAM. One worker serves all clients through the threadpool.
    uvicorn.run(app, host="0.0.0.0", port=8000)
