"""Export microsoft/unixcoder-base to ONNX for ncnn (Vulkan) inference.

CPU-only: every GPU is hidden via ``CUDA_VISIBLE_DEVICES=""`` *before* torch is
imported, so this script cannot touch the GTX 1050 or the 1060.

The graph is exported from the raw ``AutoModel`` (not optimum's task wrapper,
which bakes extra pre/post-processing into the graph that ncnn cannot consume):

    inputs : input_ids         int64   [batch, seq]  (dynamic)
             attention_mask   int64   [batch, seq]  (dynamic)
    output : last_hidden_state float32 [batch, seq, 768]

Run once::

    CUDA_VISIBLE_DEVICES="" python export_model.py

Writes ``unixcoder-onnx/model.onnx`` and ``unixcoder-onnx/reference_embeddings.json``.
"""
import inspect
import json
import os

# Must happen before `import torch` — CUDA is never initialized on any GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "microsoft/unixcoder-base"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unixcoder-onnx")
ONNX_PATH = os.path.join(OUTPUT_DIR, "model.onnx")
REFERENCE_PATH = os.path.join(OUTPUT_DIR, "reference_embeddings.json")

OPSET = 17  # >= 17 keeps LayerNormalization a single ONNX node (ncnn-consumable)

# Fixed fixture list for the reference embeddings (shared contract).
FIXTURE_TEXTS = [
    "def hello(): print(1)",
    "class A:\n    def f(self):\n        return 1",
    "SELECT * FROM users WHERE id = 1",
    "print('hello world')",
]

# Extra shapes used to prove both dynamic axes work (batch and seq differ).
EXTRA_TEXTS = ["print(2)", "x = 1"]


class UnixCoderEncoder(torch.nn.Module):
    """Thin wrapper: run the raw base model, return only ``last_hidden_state``.

    Keeps the exported graph free of RoBERTa's unused return values (pooler
    output, attention maps) so it has exactly one output tensor named
    ``last_hidden_state``.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def mean_pool(hidden: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """sum(hidden * mask[..., None]) / mask.sum(-1, keepdim=True)  (matches the service)."""
    mask = (input_ids != pad_token_id).unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pad_token_id = tokenizer.pad_token_id

    print(f"[1/5] Loading {MODEL_ID} on CPU (float32)")
    # `dtype` is the modern kwarg; `torch_dtype` keeps the script runnable on
    # older transformers (>= 4.30 per requirements.txt).
    dtype_kwargs = {}
    if "dtype" in inspect.signature(AutoModel.from_pretrained).parameters:
        dtype_kwargs["dtype"] = torch.float32
    else:
        dtype_kwargs["torch_dtype"] = torch.float32
    model = AutoModel.from_pretrained(MODEL_ID, **dtype_kwargs)
    model.eval()
    import transformers

    print(f"      torch {torch.__version__}, transformers {transformers.__version__}, "
          f"hidden={model.config.hidden_size}, opset={OPSET}")

    inputs = tokenizer(FIXTURE_TEXTS, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    seq_len = int(input_ids.shape[1])
    print(f"[2/5] Tokenized {len(FIXTURE_TEXTS)} fixture texts -> input_ids "
          f"{tuple(input_ids.shape)} (seq_len={seq_len}, pad_token_id={pad_token_id})")

    encoder = UnixCoderEncoder(model).eval()
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    }

    print(f"[3/5] Exporting -> {ONNX_PATH}")
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            (input_ids, attention_mask),
            ONNX_PATH,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False,          # legacy TorchScript exporter: plain, ncnn-friendly graph
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
        )

    print("[4/5] Validating graph (onnx.checker + onnxruntime vs torch)")
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    n_ops = len(onnx_model.graph.node)
    print(f"      onnx.checker OK, opset={onnx_model.opset_import[0].version}, "
          f"nodes={n_ops}, size={os.path.getsize(ONNX_PATH) / 1e6:.1f} MB")

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    def run_ort(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return sess.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0]

    cases = [("fixture batch=4", input_ids, attention_mask)]
    extra = tokenizer(EXTRA_TEXTS, padding=True, truncation=True, return_tensors="pt")
    cases.append((f"extra batch=2 (different seq)", extra["input_ids"], extra["attention_mask"]))

    for label, ids, mask in cases:
        with torch.no_grad():
            torch_hidden = encoder(ids, mask).numpy()
        onnx_hidden = run_ort(ids.numpy(), mask.numpy())
        assert onnx_hidden.shape == tuple(ids.shape) + (768,), (
            f"{label}: unexpected ONNX output shape {onnx_hidden.shape}, want {tuple(ids.shape) + (768,)}"
        )
        diff = float(np.abs(onnx_hidden - torch_hidden).max())
        print(f"      {label}: ONNX shape {onnx_hidden.shape}, max abs diff = {diff:.3e}")
        if not diff < 1e-3:
            raise RuntimeError(f"{label}: torch/ONNX max abs diff {diff} >= 1e-3")

    print("[5/5] Writing reference embeddings (mean-pooled torch CPU hidden states)")
    with torch.no_grad():
        hidden = encoder(input_ids, attention_mask)
    pooled = mean_pool(hidden, input_ids, pad_token_id).numpy()
    assert pooled.shape == (len(FIXTURE_TEXTS), 768), pooled.shape

    reference = {
        "texts": FIXTURE_TEXTS,
        "embeddings": [[float(v) for v in row] for row in pooled],
    }
    with open(REFERENCE_PATH, "w") as f:
        json.dump(reference, f)

    total = sum(len(r) for r in reference["embeddings"])
    print(f"      wrote {REFERENCE_PATH} "
          f"({len(reference['embeddings'])} entries, {total} floats total)")
    print(f"OK  seq_len={seq_len}  opset={OPSET}  torch={torch.__version__}")


if __name__ == "__main__":
    main()
