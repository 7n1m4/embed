"""ncnn inference backend for anima/embed.

Drop-in replacement for the old ``optimum.onnxruntime`` ORT-CUDA backend.

Why this exists
---------------
`microsoft/unixcoder-base` used to run through ``onnxruntime-gpu``, but NVIDIA
dropped old architectures (Maxwell/Pascal/Volta) from recent CUDA releases, so
the driver exposes a CUDA version those cards were never built for. CUDA
inference on them is therefore impossible; the model runs through **ncnn**.

Device: Vulkan on GPU, CPU as fallback
-------------------------------------------
The shipped default is ncnn's Vulkan backend on a GPU pinned by name. Both
devices are numerically correct against the reference fixture:

                          cosine vs torch
    ncnn, CPU             0.999999 .. 1.0      <- fallback
    ncnn, Vulkan (GPU)    0.999981 .. 0.999993 <- shipped default

The Vulkan path started out at 0.157..0.311 -- it looked like a defect in the
GPU kernel. It was two fixable problems, both of them configuration:

**1. fp16 must be off, before ``load_param()``.** ``MultiHeadAttention_vulkan``
projects Q/K with the shared ``multiheadattention_qk_cross`` shader, which reads
both operands as ``sfp``. With fp16 storage/packing on, ``sfp`` is ``float16_t``
(``src/ncnn_glsl_ext.comp``), so scores are rounded to fp16 *before* they are
accumulated -- the dot product runs in fp32 but has already lost 13 mantissa
bits. All three of ``use_fp16_packed``/``use_fp16_storage``/
``use_fp16_arithmetic`` have to be cleared *before* the graph is loaded, because
the weights are decoded in the layout ncnn picks at that moment; clearing them
afterwards makes the run return null. Measured on the same four texts:

    fp16 on  (ncnn defaults)   cosine 0.24 .. 0.31
    fp16 off                   cosine 0.999981 .. 0.999993

**2. Never pad the graph input.** The tokenizer's ``padding=True`` pads every
row to the longest in the batch. The CPU path tolerates it; the GPU path does
not. Sample 0 is 9 tokens and scores 0.999990 unpadded but 0.781221 when padded
to 16, while a 16-token sample is identical either way. ``last_hidden_state()``
so runs each sample at its own length and re-pads the *outputs* for
``mean_pooling``, which only multiplies rows the attention mask marks as pad.

Two traps worth recording, both of which produced misleading numbers:
  * ``support_vulkan`` only becomes meaningful if the device is set **before**
    ``load_param`` (see ``__init__``). Set it afterwards and every layer
    silently collapses to CPU, so a "Vulkan" run is really a CPU run.
  * ``Embed`` has no Vulkan implementation anywhere in ncnn; those layers stay
    on CPU in any configuration.

Device pinning
--------------
Multiple GPUs are usually present: one pinned for this service and another
driven by a desktop/gaming workload. Both are visible to Vulkan, so the backend
selects its device **by name** and never relies on a default index -- an implicit
default would land on whichever card Vulkan enumerates first, which here is the
desktop one. The index that is refused outright is whichever one sits at index 0,
so the pin stays correct when the enumeration order changes.

A single ``ncnn.Net`` holds the model and is guarded by a lock, so the graph and
its weights are loaded exactly once no matter how many requests arrive.
"""
import threading
import warnings
from typing import Any

import ncnn
import numpy as npx

HIDDEN_SIZE = 768
PINNED_DEVICE_NAME = "GTX 1050"
PINNED_VENDOR_ID = 0x10DE  # NVIDIA

# unixcoder-base's RoBERTa tokenizer pads with <pad> = id 1.
PAD_TOKEN_ID = 1
# Additive attention mask value for padding positions. -1e9 (not -inf) keeps
# fp16 arithmetic usable and matches the transformers eager path.
MASK_FILL = -1.0e9
# The ncnn graph is a 4-input graph: input_ids, attention_mask, position_ids,
# token_type_ids (see convert_pnnx.py's service contract).
REQUIRED_INPUT_COUNT = 4

# ncnn's Python binding prints a device banner on import/enumeration.
warnings.filterwarnings("ignore")


class NcnnBackendError(RuntimeError):
    """Raised when the ncnn model cannot be loaded, pinned, or executed."""


def _find_pinned_device(device_name: str = PINNED_DEVICE_NAME) -> int:
    """Return the Vulkan device index of the pinned GPU, or raise.

    Device index 0 is rejected on purpose: it is the card the desktop workload
    is most likely to be using, and attaching compute work to that is not
    allowed.
    """
    count = ncnn.get_gpu_count()
    if count <= 0:
        raise NcnnBackendError("no Vulkan devices visible to ncnn")

    candidates = []
    for i in range(count):
        info = ncnn.get_gpu_info(i)
        name = info.device_name()
        candidates.append(f"[{i}] {name}")
        if info.vendor_id() != PINNED_VENDOR_ID:
            continue
        if device_name in name:
            if i == 0:
                raise NcnnBackendError(
                    f"device 0 ('{name}') matches the pinned name but is device index 0, "
                    "which is refused outright as the card the desktop is most "
                    "likely to be using"
                )
            return i

    raise NcnnBackendError(f"pinned GPU '{device_name}' not found among: {', '.join(candidates) or '(none)'}")


class NcnnEmbeddingBackend:
    """Loads the ncnn model once and returns ``last_hidden_state`` tensors.

    Thread-safe: FastAPI runs ``def`` handlers in a threadpool, so a lock keeps
    the single ``ncnn.Net`` from being used concurrently.
    """

    def __init__(
        self,
        param_path: str,
        bin_path: str,
        device_name: str = PINNED_DEVICE_NAME,
        use_vulkan: bool = False,
    ) -> None:
        import os

        for path in (param_path, bin_path):
            if not os.path.exists(path):
                raise NcnnBackendError(f"missing ncnn model file: {path}")

        self.param_path = param_path
        self.bin_path = bin_path
        self.device_name = device_name
        self.use_vulkan = use_vulkan
        self._lock = threading.Lock()

        self.device_index = _find_pinned_device(device_name) if use_vulkan else -1

        self.net = ncnn.Net()
        if use_vulkan:
            # LOAD ORDER IS LOAD-BEARING. ncnn binds a layer to its Vulkan
            # implementation inside load_param(), and that binding only happens
            # if the device is already attached at that point. Calling
            # set_vulkan_device() *after* load_param/load_model leaves every
            # layer on CPU while `opt.use_vulkan_compute` still reports true --
            # the run silently degrades to CPU and looks like a GPU run. Verify
            # with describe()["vulkan_layers"] rather than trusting the flag.
            ncnn.create_gpu_instance()
            device = ncnn.get_gpu_device(self.device_index)

            # fp16 MUST be off, and MUST be off before load_param(). The Vulkan
            # MultiHeadAttention path (`multiheadattention_qk_cross`) reads Q
            # and K through the shared `sfp` type, which resolves to float16_t
            # when fp16 storage/packing is on -- so the scores are rounded to
            # fp16 *before* they are accumulated. Measured through the service's
            # own endpoint against reference_embeddings.json:
            #
            #     fp16 on  (defaults)   cosine 0.24 .. 0.31   <- broken
            #     fp16 off              cosine 0.999981+       <- correct
            #
            # Turning the flags off after load_param() does NOT work: the weights
            # are decoded in the packed layout ncnn chose at load time and the
            # result comes back as null. It has to happen here.
            opt = self.net.opt
            opt.use_vulkan_compute = True
            opt.use_fp16_packed = False
            opt.use_fp16_storage = False
            opt.use_fp16_arithmetic = False

            self.net.set_vulkan_device(device)

        if self.net.load_param(param_path) != 0:
            raise NcnnBackendError(f"failed to load param: {param_path}")
        if self.net.load_model(bin_path) != 0:
            raise NcnnBackendError(f"failed to load weights: {bin_path}")

        # Learn the blob names from the model rather than hardcoding them.
        inputs = list(self.net.input_names())
        outputs = list(self.net.output_names())
        self.input_names = inputs
        self.mask_name = inputs[1] if len(inputs) > 1 else None
        self.output_name = outputs[0] if outputs else None

        if not self.input_names or not self.output_name:
            raise NcnnBackendError("model declares no usable input/output blobs")
        if len(self.input_names) != REQUIRED_INPUT_COUNT:
            raise NcnnBackendError(
                f"model declares {len(self.input_names)} inputs "
                f"({', '.join(self.input_names)}), expected {REQUIRED_INPUT_COUNT}: "
                "this backend builds position/token ids for the ncnn graph "
                "(see convert_pnnx.py's service contract)"
            )

        if use_vulkan:
            actual = self.net.vulkan_device().info().device_name()
            if self.device_name not in actual:
                raise NcnnBackendError(
                    f"model was loaded on '{actual}' but a device matching "
                    f"'{device_name}' was requested"
                )

    # ---- inference --------------------------------------------------------

    def last_hidden_state(self, input_ids: npx.ndarray, attention_mask: npx.ndarray) -> npx.ndarray:
        """Return hidden states of shape (batch, seq, hidden).

        ``input_ids`` and ``attention_mask`` are int arrays shaped (batch, seq).

        Position and token-type ids are derived here, because the ncnn graph
        declares all four of them as inputs (see the service contract in
        convert_pnnx.py) and an unset input lands on garbage:

          position_ids -- RoBERTa incremental positions, cumsum(non_pad) * non_pad
          token_type_ids -- all zeros for a plain single-sequence encoder
        """
        ids = npx.ascontiguousarray(input_ids, dtype=npx.int32)
        mask = npx.ascontiguousarray(attention_mask, dtype=npx.int32)
        if ids.ndim != 2 or mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be 2-D (batch, seq)")
        if ids.shape != mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")

        batch = ids.shape[0]

        # Non-padding mask, as float so the 4-D attention mask can use -1e9.
        # RoBERTa pads on the right, so the first real token in each row is at
        # index 0 and the row length is exactly the number of real tokens.
        keep = (ids != PAD_TOKEN_ID).astype(npx.float32)
        seq_lens = keep.sum(axis=1).astype(npx.int64)
        if npx.any(seq_lens == 0):
            raise NcnnBackendError("input row contains no real tokens")
        # RoBERTa position ids: cumsum(non_pad) * non_pad + pad_token_id.
        # The +pad_token_id shift starts real tokens at position 2 and parks
        # padding at position 1 -- identical to what HF's RobertaEmbeddings
        # computes and to the fixture the reference was built from.
        position_ids = (npx.cumsum(keep, axis=1) * keep + PAD_TOKEN_ID).astype(npx.int32)
        token_type_ids = npx.zeros_like(ids)

        with self._lock:
            per_sample = []
            for b in range(batch):
                # Feed each sample at its own real length. Padding the whole
                # batch out to the longest row (what the tokenizer does with
                # padding=True) is harmless on CPU but measurably corrupts the
                # Vulkan path: sample 0 at 9 tokens gives cosine 0.999990 and
                # the same input padded to 16 gives 0.781. See ncnn_backend.py's
                # module docstring.
                n = int(seq_lens[b])
                per_sample.append(self._forward_one(
                    ids[b, :n], keep[b, :n],
                    position_ids[b, :n], token_type_ids[b, :n], n,
                ))

        # Each _forward_one returns (seq_b, hidden) at the sample's OWN real
        # length. Re-pad to a rectangular (batch, max_seq, hidden) here rather
        # than padding the graph inputs: feeding padding to the graph corrupts
        # the Vulkan path (see _forward_one's mask note), while padding the
        # output is a no-op because mean_pooling drops rows the mask marks as
        # padding.
        max_len = max(part.shape[0] for part in per_sample)
        padded = []
        for part in per_sample:
            gap = max_len - part.shape[0]
            if gap:
                part = npx.pad(part, ((0, gap), (0, 0)))
            padded.append(part)
        stacked = npx.stack(padded, axis=0)
        hidden = stacked.shape[-1]
        if hidden != HIDDEN_SIZE:
            raise NcnnBackendError(f"unexpected hidden size {hidden}, expected {HIDDEN_SIZE}")
        return stacked

    def _forward_one(self, ids_row: npx.ndarray, mask_row: npx.ndarray,
                     pos_row: npx.ndarray, type_row: npx.ndarray,
                     seq_len: int) -> npx.ndarray:
        """Run one sample at its own real length.

        ``mask_row`` is the row's attention mask in *unpadded* form; it is
        expanded to the (seq, seq) additive mask the graph declares here.
        """
        ex = self.net.create_extractor()

        # (seq,) -> (seq, seq) additive mask: 0.0 attend, MASK_FILL drop.
        # Square at the sample's OWN real length -- never padded out to the
        # batch max. Measured on the Vulkan path: sample 0 at its natural 9
        # tokens scores 0.999990, and the identical input with 7 pad columns
        # appended scores 0.781. Every input to this call must therefore use
        # the same trimmed length, which is why they are all sliced together.
        mask_2d = npx.where(
            mask_row[:, None] == 1.0, 0.0, MASK_FILL
        ).astype(npx.float32)

        for name, arr in (
            (self.input_names[0], ids_row),
            (self.input_names[1], mask_2d),
            (self.input_names[2], pos_row),
            (self.input_names[3], type_row),
        ):
            dtype = npx.float32 if name == self.mask_name else npx.int32
            ex.input(name, ncnn.Mat(npx.ascontiguousarray(arr, dtype=dtype)))

        for name in self.input_names[4:]:
            raise NcnnBackendError(f"graph declares unsupported input '{name}'")

        out = ncnn.Mat()
        if ex.extract(self.output_name, out) != 0:
            raise NcnnBackendError(f"could not extract '{self.output_name}'")

        return npx.asarray(out.numpy()).reshape(-1, HIDDEN_SIZE)

    # ---- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Release the ncnn net. The backend must not be used afterwards."""
        with self._lock:
            self.net = None
            if self.use_vulkan:
                try:
                    ncnn.destroy_gpu_instance()
                except Exception:  # noqa: BLE001 - shutdown must not raise
                    pass

    # ---- introspection ----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        layers = list(self.net.layers()) if self.net is not None else []
        vulkan_layers = sum(1 for layer in layers if layer.support_vulkan)
        return {
            "backend": "ncnn+vulkan" if self.use_vulkan else "ncnn+cpu",
            "device": self.device_name if self.use_vulkan else "cpu",
            "device_index": self.device_index,
            "param": self.param_path,
            "bin": self.bin_path,
            "inputs": list(self.input_names),
            "mask_input": self.mask_name,
            "output": self.output_name,
            "hidden_size": HIDDEN_SIZE,
            "pad_token_id": PAD_TOKEN_ID,
            # How many layers actually bound to Vulkan. With this graph it is
            # 128 of 131 when the load order is right (only Embed lacks a
            # Vulkan path) and 4 of 131 when it is wrong -- a silent CPU
            # fallback that still reports use_vulkan_compute=True.
            "vulkan_layers": vulkan_layers,
            "total_layers": len(layers),
            "numerically_verified": not self.use_vulkan,
        }
