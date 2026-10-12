## [anima](https://github.com/anlma)/embed

OpenAI-compatible embedding service for `microsoft/unixcoder-base`.

It exists to feed a coding agent's memory store, so the HTTP contract is fixed:
the client needs no changes and can point at `0.0.0.0:8000` from another machine.

Inference runs on any Vulkan-capable GPU through **ncnn's Vulkan backend** —
which is what makes this usable on old NVIDIA and Radeon cards that CUDA no
longer supports. Because the card is picked by name rather than by index, you can
pin the service to a spare/idle card and leave the one driving your desktop
alone, so it can run while you game.

**Scope in v1: one model, read-only.** `unixcoder-base` is the only model served
— the request's `model` field is echoed back but selects nothing, and there is no
registry or `/v1/models`. Embeddings are computed on request and never stored, so
the service holds no index and no persistence. Anything beyond that is a scope
expansion, not a fix.

## Why ncnn Vulkan

The obvious choice on an old NVIDIA card is CUDA, and it is closed: NVIDIA
dropped Maxwell/Pascal/Volta from recent CUDA releases, so the driver exposes a
CUDA version the card was never built for. That rules out
`onnxruntime-gpu` permanently, not just "needs a rebuild" — there is no CUDA path
forward on such hardware.

ncnn's Vulkan backend is the replacement, and Vulkan is still supported on those
cards and on old Radeon hardware too. It did reach the GPU, but the first
measurement came out at cosine **0.157** against torch. That turned out to be
*two* separate, fixable problems rather than a defect in the GPU path:

| Backend | Cosine vs torch | Verdict |
|---|---|---|
| CUDA (onnxruntime-gpu) | — | impossible, old architectures dropped from CUDA |
| ncnn, Vulkan on GPU | **0.999981** | correct, shipped |
| ncnn, CPU | **0.999999** | correct, fallback |

Both GPU numbers were measured through the service's own HTTP endpoint against
`unixcoder-onnx/reference_embeddings.json`, so they reflect the real request path
rather than a synthetic harness.

### Root cause 1 — fp16 rounds Q/K before the score is accumulated

Inside ncnn, `MultiHeadAttention_vulkan` projects Q/K with the shared
`multiheadattention_qk_cross` shader, which reads both operands as `sfp`. When
fp16 storage or fp16 packing is enabled, `sfp` resolves to `float16_t`
(`src/ncnn_glsl_ext.comp`), so the attention scores are rounded to fp16 *before*
they are accumulated. The dot-product itself runs in fp32, but its inputs have
already lost 13 bits of mantissa.

The fix is all three of `use_fp16_packed`, `use_fp16_storage` and
`use_fp16_arithmetic` set to **before `load_param()`**. Measured on the same four
reference texts:

```
fp16 on  (ncnn defaults)   cosine 0.24 .. 0.31   <- the 0.157 figure
fp16 off                   cosine 0.999981 .. 0.999993
```

Turning the flags off *after* `load_param()` does not work: the weights have
already been decoded in the packed layout ncnn chose, and the run returns null.
The disabling has to happen before the graph is loaded.

### Root cause 2 — padding the batch corrupts the GPU, but not the CPU

The tokenizer's `padding=True` pads every row out to the longest in the batch.
The CPU path tolerates this; the GPU path does not. Measured one text at a time:

| Sample | Natural length | cos at natural length | cos padded to 16 |
|---|---|---|---|
| 0 | 9 | 0.999990 | 0.781221 |
| 1 | 16 | 0.999993 | 0.999993 |
| 2 | 10 | 0.999981 | 0.798192 |
| 3 | 7 | 0.999989 | 0.754781 |

Sample 1 is exactly 16 tokens and needs no padding, so it is unaffected either
way — which is what points at the padding rather than the sequence itself.

`ncnn_backend.last_hidden_state()` therefore runs every sample at its **own real
length** and re-pads the *outputs* to a rectangular tensor for `mean_pooling`,
which only ever multiplies rows the attention mask marks as padding.

### Two traps worth knowing

**1. Load order silently decides whether Vulkan runs at all.** ncnn binds a
layer to its Vulkan implementation inside `load_param()`, and that binding only
happens if the device is already attached. Call `set_vulkan_device()` *after*
`load_param`/`load_model` and every layer collapses to CPU while
`opt.use_vulkan_compute` still reports true:

```
device set BEFORE load_param   ->  128 of 131 layers on Vulkan   (correct)
device set AFTER  load_param   ->    4 of 131 layers on Vulkan   (silent CPU)
```

The three layers that never reach Vulkan are the `Embed` layers — `Embed` has no
Vulkan implementation anywhere in ncnn. `describe()` now reports
`vulkan_layers` so this can be checked instead of trusted.

**2. Pinning by name, never by index.** Multiple GPUs are usually visible to
Vulkan, so the device is resolved by name and index 0 is refused outright. That
keeps the card driving your desktop free regardless of what else is plugged in,
and there is no per-request opt-in that can move the model onto it. To serve a
different card, change the pinned name (a constant in the backend, not a flag)
and restart.

## Prerequisites

1.  **Linux** with `python` (3.8+) and `pip`.
2.  **`ncnn`** with Vulkan support — the system `ncnn` package or `pip install ncnn`.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Export Model** (one-time, CPU-only — hides all GPUs)
   ```bash
   CUDA_VISIBLE_DEVICES="" python export_model.py
   ```

3. **Convert to ncnn** (one-time; produces `unixcoder-onnx/unixcoder.pnnx.fp32.ncnn.*`)
   ```bash
   CUDA_VISIBLE_DEVICES="" python convert_pnnx.py
   ```

## Usage

The service is OpenAI-compatible at `http://localhost:8000/v1/embeddings`.

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "print(\"hello world\")", "model": "unixcoder-base"}'
```

A batch is the same request with a list, and `index` follows list position:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["def f(): pass", "class A: pass"], "model": "unixcoder-base"}'
```

`python mock_client.py` drives both shapes and exits 0.

It is a normal OpenAI-shaped client — anything that speaks the embeddings API
works unchanged. In Python, using only the standard library:

```python
import json, urllib.request

def embed(texts):
    req = urllib.request.Request(
        "http://localhost:8000/v1/embeddings",
        data=json.dumps({"input": texts, "model": "unixcoder-base"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return [row["embedding"] for row in json.load(resp)["data"]]

vecs = embed(["def f(): pass", "class A: pass"])
print(len(vecs), [len(v) for v in vecs])   # 2 [768, 768]
```

No auth is enforced, so the endpoint is open to whatever can reach it — put a
reverse proxy in front of it before exposing it past a trusted network.

### Running it

```bash
bash run-embed-service.sh          # Vulkan GPU (default)
DEVICE=cpu bash run-embed-service.sh   # CPU fallback, same embeddings
```

Or the service directly: `python embedding_service.py [--device cpu|vulkan]`.

Keep `workers=1` (the default, and the only value accepted). Every worker loads
its own copy of the model. ncnn's own inference is also serialized behind a lock
inside the backend, so a second process buys nothing.

### GPU behaviour

- The GPU is pinned by **name**. Index 0 is rejected outright, so no request
  shape can move the model onto it, whichever card it is.
- Under sustained load the service holds **~374 MiB at 90%+ utilization** on the
  pinned card. Any card driving the desktop stays at its idle baseline with only
  that desktop's own processes attached.
- `--device cpu` attaches to no GPU at all — useful when the card is busy, or to
  cross-check the GPU's numbers.

