## [anima](https://github.com/anlma)/embed

OpenAI-compatible embedding service for `microsoft/unixcoder-base`.

Inference runs on the **GTX 1050 via ncnn's Vulkan backend**. The GTX 1060 used
for desktop/gaming is left completely untouched, so the service can run while you
game.

## Why ncnn Vulkan on the GTX 1050

The GTX 1050 is Pascal (`sm_61`). The installed driver only exposes CUDA 13.0,
which dropped Maxwell/Pascal/Volta — so **CUDA inference on the 1050 is
impossible**, permanently. ncnn's Vulkan backend was the replacement.

Vulkan did reach the GPU, but the first measurement came out at cosine **0.157**
against torch. That turned out to be *two* separate, fixable problems rather than
a defect in the GPU path:

| Backend | Cosine vs torch | Verdict |
|---|---|---|
| CUDA (onnxruntime-gpu) | — | impossible, Pascal dropped from CUDA 13 |
| ncnn, Vulkan on GTX 1050 | **0.999981** | correct, shipped |
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

**2. Pinning by name, never by index.** Both GPUs expose Vulkan, so the device is
resolved by the `GTX 1050` name and index 0 is refused outright. The GTX 1060
stays free for the desktop with no opt-in per request.

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

### Running it

```bash
bash run-embed-service.sh          # GTX 1050 via ncnn Vulkan (default)
DEVICE=cpu bash run-embed-service.sh   # CPU fallback, same embeddings
```

Or the service directly: `python embedding_service.py [--device cpu|vulkan]`.

Keep `workers=1` (the default, and the only value accepted). Every worker loads
its own copy of the model. ncnn's own inference is also serialized behind a lock
inside the backend, so a second process buys nothing.

### GPU behaviour

- The GTX 1050 is pinned by **name**. Index 0 is rejected outright, so there is
  no request shape that can put the model on the GTX 1060.
- Under sustained load the service holds **~374 MiB at 90%+ utilization** on the
  1050. The 1060 stays at its idle baseline with only the desktop's own
  processes (`vibepanel`, `hyprshell`, `chromium`) attached.
- `--device cpu` attaches to no GPU at all — useful when the card is busy, or to
  cross-check the GPU's numbers.

