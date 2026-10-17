# Repository Guidelines

## Project Overview
AnimaEmbed: OpenAI-compatible embedding service for `microsoft/unixcoder-base`.
It exists to feed the coding agent's memory store, which is why the HTTP contract
is fixed — the memory client needs zero changes and must keep working over the
network against `0.0.0.0:8000`.

Inference runs on any Vulkan-capable GPU through **ncnn's Vulkan backend**,
pinned by GPU name (index 0 is refused outright, so the card driving your desktop
stays free). Vulkan rather than CUDA is what makes the service usable on old
NVIDIA and Radeon hardware: NVIDIA dropped those architectures from recent CUDA
releases, so the CUDA path is closed for good on them. Two ncnn settings are
load-bearing and are detailed in `ncnn_backend.py` and `README.md`: fp16 must be
disabled **before** `load_param()`, and every sample must be fed at its own real
length, never padded to the batch max. Without both, the GPU runs at 90%+ and
produces embeddings at cosine ~0.16. With both, cosine is 0.999981+ and the
pinned card holds **~370–400 MiB at rest** following a restart (measured 370, 376,
then 402 MiB at steady state) — the fp32 weight floor is 501,356,844 bytes =
478.13 MiB, and fp16 is off because it would break the embeddings. Memory is a
ratchet: one very long request raised it 900 → **1661 MiB** and it never came
back, and further long requests took it to **~1962 MiB** of 2048 MiB, because
ncnn's `VkBlobAllocator` does not return blocks to the driver. **That ratchet is
permanent for the life of the process, and a restart clears it — which is why
the figures above are ~370 MiB now.** An idle unload also releases the card
(measured back to 357 MiB), but only by tearing the whole Vulkan instance down;
it does not reclaim the ratchet's high-water mark *within* a running process, so
a card pushed to 1962 MiB needs a restart either way. The footprint is bounded
by the longest request ever served, so `MAX_INPUT_TOKENS` (1024) is the cap that
bounds it; reason about the card's headroom accordingly.

**Scope in v1: one model, read-only.** `microsoft/unixcoder-base` is the only
model served; the request's `model` field is echoed back but is not used to
select anything — there is no registry, no `/v1/models`, and no way to serve a
second model without a code change. Embeddings are computed on request and never
stored: the service holds no index, no vector database, and no persistence. A
change to any of that is a scope expansion, not a fix, and should be agreed
first.

## Architecture & Data Flow
FastAPI app (`embedding_service.py`) → POST `/v1/embeddings` endpoint → ncnn
inference on the pinned Vulkan device (`ncnn_backend.py`) → mean pooling (NumPy)
→ OpenAI-shaped JSON response. Two-file backend split: `embedding_service.py`
owns the HTTP layer and pooling; `ncnn_backend.py` owns model loading, input
derivation, and the forward pass. ORT returns NumPy arrays, not torch tensors
(the original CUDA backend was removed; nothing in the current path uses ONNX
Runtime — `export_model.py` only uses it to produce the ONNX input to pnnx).

**Device pinning:** resolved by GPU *name* at backend construction, and index 0
is rejected explicitly, so there is no request shape that can move the model onto
the card driving the desktop. Targeting a different card means changing the
pinned-name constant in the backend — there is no CLI flag for it.

**Model lifecycle:** the model is **not** resident while the service is idle. It
loads on the first request after an idle gap and is unloaded again when the
service goes quiet, Ollama-style: an idle service holds no VRAM at all (357 MiB
vs 727 MiB resident, measured on the pinned card via `nvidia-smi`). The first
request after a gap pays the reload (~1 s on Vulkan) and gets a bit-identical
result (cosine 1.000000 across a load/unload/reload cycle).
`--keep-alive <seconds>` sets the idle timeout (default 300); `-1` keeps the
model resident for the life of the process, `0` unloads after every request.
Preflight still loads the model to verify the Vulkan binding, then releases it
before serving — the eager load is verification, not residency. The unload runs
under the same lock as inference (it cannot interrupt a forward pass) and the
process only ever holds one `ncnn.Net`, because `--workers 1` is enforced and
`ncnn.destroy_gpu_instance()` is process-global: tearing it down while a second
Net is loaded segfaults the process.

**Backend history (do not reopen the fp16 and padding questions):** the first
correct-load-order Vulkan measurement was cosine 0.157 vs torch. That is two
issues, both fixed here — fp16 rounding Q/K before score accumulation, and
padding the batch graph input. `describe()["vulkan_layers"]` reports the real
count: 128/131 with `set_vulkan_device()` called before `load_param()`, 4/131 if
it is called after (silent CPU while `use_vulkan_compute` still reports true).
The 3 missing layers are `Embed`, which has no Vulkan implementation anywhere in
ncnn.

Key data path:
1. Request received with `input` (str or list[str]) and `model`
2. Tokenizer (lazy-loaded on first request, stored on `app.tokenizer`) produces NumPy tensors
3. `ncnn_backend.NcnnEmbeddingBackend` derives `position_ids`/`token_type_ids`/mask from `input_ids`, trims each sample to its real length, and runs the ncnn graph per sample → `last_hidden_state`
4. `mean_pooling()` reduces (batch, seq, hidden) → (batch, hidden) embeddings
5. Response shaped as OpenAI Embeddings API JSON

## Response contract
OpenAI's `/v1/embeddings` shape, unchanged from git HEAD:
`object="list"`, top-level `model` and `usage`, and `data[i]` carrying
`object="embedding"`, `embedding` (768 floats), `index`. Do not add `model` to
`data[]` — it is a top-level field, and the contract must not change.

## Key Directories
Flat structure — all modules at repository root. No `src/`, `tests/`, or subdirectories.

```
animaembed/
├── embedding_service.py    # App logic, schemas, pooling, endpoint, entry point
├── ncnn_backend.py         # ncnn inference: load, input derivation, forward
├── export_model.py         # One-time model export (PyTorch → ONNX, CPU-only)
├── convert_pnnx.py         # One-time ONNX → ncnn conversion + reference embeddings
├── run-embed-service.sh    # Production launch wrapper (--device vulkan default)
├── mock_client.py          # OpenAI-shaped smoke client, exits 0
├── validate_pnnx.py        # CPU-vs-Vulkan drift probe used to find the fp16 bug
├── val_pnnx.cpp            # C++ accuracy validator for the ncnn graph
├── vulkan_probe.cpp        # C++ per-layer probe used to find the Vulkan bug
├── requirements.txt        # Python dependencies
├── README.md               # Setup and deployment documentation
├── AGENTS.md               # This file
└── .gitignore
unixcoder-onnx/             # Gitignored — generated by export_model.py + convert_pnnx.py
```

## Development Commands
| Action | Command |
|--------|---------|
| Install deps | `pip install -r requirements.txt` |
| Export ONNX model (one-time, CPU) | `CUDA_VISIBLE_DEVICES="" python export_model.py` |
| Convert ONNX → ncnn (one-time, CPU) | `CUDA_VISIBLE_DEVICES="" python convert_pnnx.py` |
| Build the C accuracy validator | `g++ -std=c++11 -O2 -o val_pnnx val_pnnx.cpp -I/usr/include/ncnn -lncnn` |
| Run service | `python embedding_service.py --device vulkan` (or `cpu`) |
| Production wrapper | `bash run-embed-service.sh` (`DEVICE=cpu` to fall back) |
| Test endpoint | `curl -X POST http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{"input": "text", "model": "unixcoder-base"}'` |

Run the service with `--workers 1` (default, and the only accepted value). Each
worker loads its own copy of the model into VRAM, and a low-VRAM card — which is
the situation this targets — cannot hold more than one. `--device vulkan` is the
shipped default and is numerically correct (cosine 0.999981); `--device cpu` is
the fallback, not the reverse.

## Code Conventions & Common Patterns
- **No `async def` for handlers**: Use plain `def` — ncnn inference blocks the CPU; `async def` would serialize concurrent requests (FastAPI runs `def` handlers in a threadpool automatically).
- **Pydantic v2 `BaseModel`** for all request/response schemas: `EmbeddingRequest`, `EmbeddingData`, `EmbeddingResponse`.
- **NumPy everywhere**: ncnn returns `np.ndarray`; pooling operates on NumPy arrays — no torch tensors after model conversion.
- **State on `app`**: Model and tokenizer are lazily initialized and stored as `app.model` / `app.tokenizer`. No DI container.
- **Backend is a separate module**: `ncnn_backend.NcnnEmbeddingBackend` owns ncnn specifics (device setup, blob discovery, input derivation, forward). `embedding_service.py` never touches the ncnn API directly.
- **Backend failure is surfaced, not hidden**: a bad backend raises `NcnnBackendError`, which the endpoint converts to a 503. Failures are loud.
- **Type hints** used throughout.

## Important Files
- **`embedding_service.py`** — Main FastAPI app, `/v1/embeddings` endpoint, mean pooling, CLI arg parsing. HTTP-layer changes go here.
- **`ncnn_backend.py`** — ncnn inference backend. Owns the model-load order, the 4-input graph contract (`in0` ids / `in1` mask / `in2` pos / `in3` tts), and the derivation of `position_ids`/`token_type_ids`/mask from `input_ids`. The contract is validated at load time, so a wrong graph fails immediately instead of returning garbage.
- **`convert_pnnx.py`** — Converts ONNX → ncnn and writes the reference embeddings used to verify accuracy.
- **`val_pnnx.cpp`** — Independent C++ accuracy check against `fixture_*.npy`, bypassing Python entirely.
- **`run-embed-service.sh`** — Production launch wrapper; runs `--device vulkan` by default, `DEVICE=cpu` to fall back.
- **`mock_client.py`** — OpenAI-shaped smoke client: single + batch requests, exits 0. Run it against a live service.

## Runtime/Tooling Preferences
- **Runtime**: Python 3.8+, `ncnn` (system package or pip). No CUDA needed for inference or conversion.
- **Package manager**: pip only — no pyproject.toml, setup.py, or conda
- **No build step** — Python scripts run directly
- **No containerization** — Docker not configured
- **Production deployment**: systemd unit template referenced in README but not present in repo (private runbook)

## Testing & QA
- **No pytest suite / no CI** — verification is by smoke run
- **No lint/format tools** configured (ruff, black, mypy, etc.)
- Accuracy is verified against `unixcoder-onnx/reference_embeddings.json`
  (cosine must be ≥0.999 for every input)
- `val_pnnx.cpp` is an independent C++ accuracy check that bypasses Python;
  `vulkan_probe.cpp` compares per-layer CPU vs Vulkan blobs to localise numeric
  divergence
- End-to-end check: curl the running service and compare against the reference
- **Both devices must be re-checked after any change to the backend.** Run a
  batch through the endpoint with `--device vulkan` and again with `--device cpu`;
  both must be ≥0.999 against `reference_embeddings.json`. A perf tweak that
  reintroduces fp16 or pads the graph input silently moves only the GPU number.
- GPU isolation check: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
  must not list the service pid on the card driving the desktop — check by
  matching the pid, not by hardcoded index, since which card is index 0 varies by
  machine
