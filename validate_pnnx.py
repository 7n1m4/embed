"""Validate the pnnx -> ncnn unixcoder model against the torch reference.

The onnx2ncnn path is unusable (377 unsupported/placeholder ops, no LayerNorm)
and the first pnnx attempt produced a graph pnnx filled with a raw ``SDPA``
layer whose head/batch layout does not match ncnn's SDPA op (ncnn reads
``num_heads`` from the channel axis). ``convert_pnnx.py`` now wraps the
attention blocks so pnnx folds them into ncnn's native
``MultiHeadAttention``; this script proves the result is numerically correct.

Everything runs through the ``ncnn`` **python** bindings (``pip install ncnn``,
same version as the system libncnn), so there is no helper process and no C++
custom layer. MHA layers keep per-net state, so a *fresh* ``ncnn.Net`` is
loaded per request -- sharing one Net across extractors silently corrupts the
second request.

Run::

    CUDA_VISIBLE_DEVICES="" python validate_pnnx.py            # CPU fp32 + fp16
    CUDA_VISIBLE_DEVICES="" python validate_pnnx.py --vulkan 1  # ... plus Vulkan
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

try:
    import ncnn
except ImportError:  # pragma: no cover - dependency hint
    sys.exit("missing module 'ncnn'.  pip install ncnn  (matching the system libncnn)")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "unixcoder-onnx")
REF_PATH = os.path.join(OUT, "reference_embeddings.json")

# (param basename, precision label, cosine bar)
VARIANTS = [
    ("unixcoder.pnnx.fp32", "fp32", 0.999),
    ("unixcoder.pnnx", "fp16", 0.99),
]


def load_fixtures():
    """Fixture tensors dumped by convert_pnnx.py (same tokenizer, same texts)."""
    f = lambda n: np.load(os.path.join(OUT, "fixture_%s.npy" % n))
    return {
        "input_ids": f("input_ids").astype(np.int64),
        "position_ids": f("position_ids").astype(np.int64),
        "token_type_ids": f("token_type_ids").astype(np.int64),
        "attn_mask_4d": f("attn_mask_4d").astype(np.float32),
    }


def load_reference():
    with open(REF_PATH) as fh:
        payload = json.load(fh)
    return np.asarray(payload["embeddings"], dtype=np.float64)


def mm_arr(mat):
    """Copy an ncnn Mat to a numpy array while its buffer is still alive."""
    return np.array(mat.numpy())


def input_mat(values):
    """Build an ncnn input Mat.

    Int inputs MUST be int32: ncnn's Embed layer casts the buffer to ``int*``,
    so an int64 array makes it read every other element as an index (half the
    tokens come out as token 0).  pnnx's own prompt says
    "replace ncnn input torch.long type with torch.int" for exactly this reason.
    """
    return ncnn.Mat(np.ascontiguousarray(values, dtype=np.int32)).clone()


def run_sequence(param, binpath, ids, pos, tts, mask, vulkan_device=None):
    """Run one sequence and return hidden states shaped ``[seq, 768]``.

    A fresh ``ncnn.Net`` per call: ``MultiHeadAttention`` retains state, and
    reusing a Net across extractors corrupts the second request.
    """
    with ncnn.Net() as net:
        if vulkan_device is not None:
            net.set_vulkan_device(vulkan_device)
            net.opt.use_vulkan_compute = True
        net.load_param(param)
        net.load_model(binpath)

        with net.create_extractor() as ex:
            ex.input("in0", input_mat(ids))
            ex.input("in1", ncnn.Mat(np.ascontiguousarray(mask, dtype=np.float32)).clone())
            ex.input("in2", input_mat(pos))
            ex.input("in3", input_mat(tts))
            ret, out = ex.extract("out0")
            if ret != 0:
                raise RuntimeError("extract('out0') failed with %d" % ret)

            arr = mm_arr(out)
    # ncnn layout is [seq][emb] for this graph (w=embed_dim, h=seq)
    if arr.ndim != 2:
        raise RuntimeError("unexpected output shape %r" % (arr.shape,))
    return arr


def mean_pool(hidden, input_ids, pad_token_id=1):
    """sum(hidden * mask) / mask.sum() with mask = (input_ids != pad_token_id)."""
    keep = (input_ids != pad_token_id).astype(np.float64)
    return (hidden * keep[:, None]).sum(axis=0) / keep.sum()


def evaluate(tag, param, binpath, fixture, ref, bar, vulkan_device=None):
    ids, pos, tts = fixture["input_ids"], fixture["position_ids"], fixture["token_type_ids"]
    mask_all, batch = fixture["attn_mask_4d"], fixture["input_ids"].shape[0]

    pooled = []
    for b in range(batch):
        hidden = run_sequence(param, binpath, ids[b], pos[b], tts[b], mask_all[b].reshape(1, -1), vulkan_device).astype(np.float64)
        if hidden.shape != (ids.shape[1], ref.shape[1]):
            raise RuntimeError("%s: bad hidden shape %r" % (tag, hidden.shape,))
        pooled.append(mean_pool(hidden, ids[b]))
    pooled = np.vstack(pooled)

    rows = []
    for b in range(batch):
        p, r = pooled[b], ref[b]
        diff = np.abs(p - r)
        denom = (np.linalg.norm(p) * np.linalg.norm(r)) or 1.0
        rows.append((float(diff.max()), float(diff.mean()), float(p @ r / denom)))

    worst_cos = min(r[2] for r in rows)
    passed = worst_cos >= bar
    print("== %s%s" % (tag, " (vulkan device %s)" % vulkan_device if vulkan_device is not None else ""))
    print("%-5s %14s %14s %12s" % ("row", "max_abs_diff", "mean_abs", "cosine"))
    for b, (mx, mn, cs) in enumerate(rows):
        print("%-5d %14.6e %14.6e %12.6f" % (b, mx, mn, cs))
    print("   worst cosine=%.6f  bar=%.3f  -> %s\n" % (worst_cos, bar, "PASS" if passed else "FAIL"))
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vulkan", type=int, default=None, metavar="DEVICE",
                    help="also run on this Vulkan device index (e.g. 1 for the GTX 1050)")
    args = ap.parse_args()

    fixture = load_fixtures()
    ref = load_reference()
    if ref.shape != (fixture["input_ids"].shape[0], 768):
        sys.exit("reference shape %r does not match the fixture batch" % (ref.shape,))

    ok = True
    for tag, _prec, bar in VARIANTS:
        param = os.path.join(OUT, tag + ".ncnn.param")
        binpath = os.path.join(OUT, tag + ".ncnn.bin")
        if not os.path.exists(param):
            print("missing %s -- run: CUDA_VISIBLE_DEVICES=\"\" python convert_pnnx.py\n" % param)
            ok = False
            continue
        ok = evaluate(tag, param, binpath, fixture, ref, bar) and ok

    if args.vulkan is not None:
        name = ncnn.get_gpu_info(args.vulkan).device_name()
        print("vulkan device %d -> %s\n" % (args.vulkan, name))
        for tag, _prec, bar in VARIANTS:
            param = os.path.join(OUT, tag + ".ncnn.param")
            binpath = os.path.join(OUT, tag + ".ncnn.bin")
            if os.path.exists(param):
                ok = evaluate(tag, param, binpath, fixture, ref, bar, args.vulkan) and ok

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
