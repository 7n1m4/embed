"""Convert microsoft/unixcoder-base from PyTorch to ncnn via pnnx (fp16 + fp32).

The onnx2ncnn path is unusable (377 unsupported/placeholder ops, no LayerNorm).
pnnx (Tencent's official PyTorch -> ncnn tool) emits ncnn-native ops, so the same
torch module is converted directly.

Three details matter for a *correct* and *clean* conversion:

1. **The transformers 5.x mask path is untraceable.** `masking_utils` builds the
   4D attention mask with `_non_vmap_expansion_sdpa` + `aten::index` on the live
   attention mask; TorchScript records those index tensors as constants, so pnnx
   emits `pnnx_fold_result0`/`Tensor.index`/`pnnx.Expression` and the resulting
   ncnn graph **ignores the runtime mask entirely**. Worse, the traced module does
   not even replay: the generated `*_pnnx.py` dies with
   "expanded size (16) must match existing size (4) at non-singleton dimension 3".

2. **Everything data-dependent is fed in as plain tensors.** The wrapper therefore
   takes three explicit inputs and lets transformers pass each through untouched:
     - `input_ids`   int64/int32 [batch, seq]
     - `attn_mask`   float32    [batch, 1, 1, seq]  (0.0 keep, -1e9 drop; ncnn SDPA *adds* the mask)
     - `position_ids` int64/int32 [batch, seq]  (RoBERTa incremental positional ids)
   position_ids is what the `torch.ne`/`cumsum`/`mul` chain computed; token_type_ids
   is all-zero for a RoBERTa encoder, so no gather is needed either. Result: the
   ncnn graph contains **no** pnnx custom ops -- every layer is stock ncnn
   (Embed/LayerNorm/InnerProduct/SDPA/GELU/Reshape/Permute/Split/BinaryOp).

3. **CPU only.** Every GPU is hidden via ``CUDA_VISIBLE_DEVICES=""`` *before*
   torch is imported, so this script never initializes CUDA on the GTX 1050 or
   the GTX 1060 gaming GPU.

Run::
    CUDA_VISIBLE_DEVICES="" python convert_pnnx.py
    g++ -std=c++11 -O2 -o val_pnnx val_pnnx.cpp -I/usr/include/ncnn -lncnn
    ./val_pnnx unixcoder-onnx/unixcoder.pnnx.fp32.ncnn.{param,bin}   # numbers
"""
import inspect
import os
import sys

# Must happen before `import torch` - CUDA is never initialized on any GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from pnnx import export as pnnx_export  # noqa: E402

# Reused contract fixtures from the ONNX export slice.
from export_model import FIXTURE_TEXTS, mean_pool  # noqa: E402

MODEL_ID = "microsoft/unixcoder-base"
OUTPUT_DIR = os.path.join(HERE, "unixcoder-onnx")
PT_PATH = os.path.join(OUTPUT_DIR, "unixcoder_pnnx.pt")

# -1e9 keeps fp16 usable (-inf works too, but -FLT_MAX padding is what
# transformers' eager path uses); 0.0 means "attend".
MASK_FILL = -1.0e9

VARIANTS = [
    # (tag, fp16 flag)
    ("unixcoder.pnnx", True),
    ("unixcoder.pnnx.fp32", False),
]


class UnixCoderEncoder4D(torch.nn.Module):
    """Wrapper: run the raw base model, return only ``last_hidden_state``.

    Takes the 2D ``[batch, seq]`` attention mask (as a real client would) and
    unsqueezes it to 4D inside the graph. This keeps the mask *dynamic*
    through pnnx instead of baking it in at trace time.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = attention_mask.to(torch.bool)[:, None, None, :]
        return self.model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state


class UnixCoderEncoderPlain(torch.nn.Module):
    """Wrapper with every data-dependent tensor passed in explicitly.

    ``transformers`` forward-args that we now supply ourselves:
      input_ids      -- word ids
      attention_mask -- 4D float mask (0 keep / MASK_FILL drop), passed through as-is
      position_ids   -- RoBERTa incremental positional ids
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor,
                position_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        ).last_hidden_state


class UnixCoderEncoderMHA(torch.nn.Module):
    """Wrapper that re-expresses every attention block so pnnx fuses it.

    ``transformers`` computes q/k/v with ``.view(B, S, H, D).transpose(1, 2)``.
    pnnx's ``F_scaled_dot_product_attention -> MultiHeadAttention`` fusion
    requires ``Tensor.permute(dims=(0, 2, 1, 3))`` instead, otherwise it emits a
    raw ``SDPA`` graph whose head/batch layout does not match ncnn's SDPA layer
    (it reads ``num_heads`` from the channel axis) -- that graph is what produced
    the NaN / crash. Re-implementing the two reshapes with ``permute`` makes
    pnnx fold query/key/value/out_proj into ncnn's native ``MultiHeadAttention``.

    Everything else is taken verbatim from the original ``RobertaLayer`` /
    ``RobertaEmbeddings`` computation so the weights and the math are unchanged.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor,
                position_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        H = self.model.config.num_attention_heads
        D = self.model.config.hidden_size // H

        e = self.model.embeddings
        x = e.word_embeddings(input_ids) + e.token_type_embeddings(token_type_ids) \
            + e.position_embeddings(position_ids)
        # reproduce LayerNorm verbatim (eps from the module, not a literal)
        x = e.LayerNorm(x)

        # The 4D mask is passed in already prepared (0.0 keep / MASK_FILL drop);
        # it must reach SDPA verbatim -- deriving it here would invert it.
        mask = attn_mask

        for layer in self.model.encoder.layer:
            a = layer.attention.self
            q = a.query(x).view(B, S, H, D).permute(0, 2, 1, 3)
            k = a.key(x).view(B, S, H, D).permute(0, 2, 1, 3)
            v = a.value(x).view(B, S, H, D).permute(0, 2, 1, 3)
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            attn = layer.attention.output.dense(o.permute(0, 2, 1, 3).reshape(B, S, self.model.config.hidden_size))
            attn = layer.attention.output.dropout(attn) if self.model.training else attn
            x = layer.attention.output.LayerNorm(x + attn)

            h = layer.intermediate.dense(x)
            h = torch.nn.functional.gelu(h)
            h = layer.output.dense(h)
            h = layer.output.dropout(h) if self.model.training else h
            x = layer.output.LayerNorm(x + h)

        return x


def load_encoder(wrapper) -> torch.nn.Module:
    if "dtype" in inspect.signature(AutoModel.from_pretrained).parameters:
        dtype_kwargs = {"dtype": torch.float32}
    else:
        dtype_kwargs = {"torch_dtype": torch.float32}
    model = AutoModel.from_pretrained(MODEL_ID, **dtype_kwargs)
    model.eval()
    return wrapper(model).eval()


def fixture_tensors(tokenizer):
    """Tokenize the fixture texts and build the three graph inputs + reference."""
    inputs = tokenizer(
        FIXTURE_TEXTS, padding=True, truncation=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # RoBERTa position ids: cumsum(non_pad) * non_pad + padding_idx
    keep = (input_ids != tokenizer.pad_token_id)
    position_ids = (torch.cumsum(keep.long(), dim=1) * keep.long()) + tokenizer.pad_token_id

    # 4D float mask: 0.0 attend, MASK_FILL masked out
    mask4d = torch.where(
        keep[:, None, None, :], torch.zeros(()), torch.full((), MASK_FILL)
    ).to(torch.float32)

    # RoBERTa token_type ids are all-zero for a plain encoder
    token_type_ids = torch.zeros_like(input_ids)

    return input_ids, attention_mask, mask4d, position_ids, token_type_ids


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids, attention_mask, mask4d, position_ids, token_type_ids = fixture_tensors(tokenizer)
    seq_len = int(input_ids.shape[1])
    print(f"tokenized {len(FIXTURE_TEXTS)} fixture texts -> "
          f"input_ids {tuple(input_ids.shape)} (pad_token_id={tokenizer.pad_token_id})",
          flush=True)

    # ---- sanity: torch-side reference against the JSON reference -----------
    encoder = load_encoder(UnixCoderEncoderMHA)
    with torch.no_grad():
        hidden = encoder(input_ids, mask4d, position_ids, token_type_ids)
        pooled = mean_pool(hidden, input_ids, tokenizer.pad_token_id).numpy()
    ref = np.array(_load_reference())
    diff = float(np.abs(pooled - ref).max())
    cos = [
        float(pooled[i] @ ref[i] / np.linalg.norm(pooled[i]) / np.linalg.norm(ref[i]))
        for i in range(len(ref))
    ]
    print(f"torch graph check: max_abs_diff={diff:.3e} min_cosine={min(cos):.6f}",
          flush=True)
    if diff > 1e-3:
        raise RuntimeError(f"wrapper disagrees with reference: max diff {diff}")
    print(f"torch hidden: {tuple(hidden.shape)} {hidden.dtype}", flush=True)

    # ---- persist the fixture inputs for the C++ runner ---------------------
    if False:
        dump = {
        "seq_len": seq_len,
        "batch": int(input_ids.shape[0]),
        "pad_token_id": int(tokenizer.pad_token_id),
        "input_ids": input_ids.numpy(),
        "attention_mask_2d": attention_mask.numpy(),
        "attn_mask_4d": mask4d.numpy(),
        "position_ids": position_ids.numpy(),
        "token_type_ids": token_type_ids.numpy(),
    }
    # plain .npy files (consumed directly by val_pnnx.cpp)
    np.save(os.path.join(OUTPUT_DIR, "fixture_input_ids.npy"), input_ids.numpy())
    np.save(os.path.join(OUTPUT_DIR, "fixture_attention_mask.npy"), attention_mask.numpy())
    np.save(os.path.join(OUTPUT_DIR, "fixture_attn_mask_4d.npy"), mask4d.numpy())
    np.save(os.path.join(OUTPUT_DIR, "fixture_position_ids.npy"), position_ids.numpy())
    np.save(os.path.join(OUTPUT_DIR, "fixture_token_type_ids.npy"), token_type_ids.numpy())
    print("wrote unixcoder-onnx/fixture_*.npy", flush=True)

    # ---- pnnx -> ncnn ------------------------------------------------------
    for tag, fp16 in VARIANTS:
        ncnnparam = os.path.join(OUTPUT_DIR, tag + ".ncnn.param")
        ncnnbin = os.path.join(OUTPUT_DIR, tag + ".ncnn.bin")
        pt_path = os.path.join(OUTPUT_DIR, f"unixcoder_pnnx.{tag.split('.')[-1]}.pt")
        print(f"--- pnnx fp16={fp16} -> {tag}.ncnn.param/bin", flush=True)
        pnnx_export(
            encoder,
            pt_path,
            inputs=(input_ids, mask4d, position_ids, token_type_ids),
            ncnnparam=ncnnparam,
            ncnnbin=ncnnbin,
            fp16=fp16,
            optlevel=2,
            check_trace=False,
        )
        print(f"    param={os.path.getsize(ncnnparam)} bytes "
              f"bin={os.path.getsize(ncnnbin)} bytes", flush=True)

    _report_ops()


def _load_reference():
    import json
    with open(os.path.join(OUTPUT_DIR, "reference_embeddings.json")) as f:
        return json.load(f)["embeddings"]


CUSTOM_OPS = ("torch.ne", "torch.gather", "Tensor.to", "Tensor.type_as",
              "Tensor.index", "pnnx.Expression", "CumulativeSum", "Unsupported")


def _report_ops() -> None:
    for tag, _fp16 in VARIANTS:
        path = os.path.join(OUTPUT_DIR, tag + ".ncnn.param")
        if not os.path.exists(path):
            continue
        lines = open(path).read().split("\n")
        counts = {}
        bad = []
        for i in range(2, len(lines)):
            if not lines[i].strip():
                continue
            t = lines[i].split()
            counts[t[0]] = counts.get(t[0], 0) + 1
            if t[0] in CUSTOM_OPS:
                bad.append(f"{t[0]}:{t[1]}")
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
        print(f"[ops] {tag}: layers={counts.get('7767517', 0) or sum(counts.values())} {summary}")
        print(f"[ops] {tag}: custom/unsupported -> {bad if bad else 'NONE (all stock ncnn)'}")


if __name__ == "__main__":
    main()
