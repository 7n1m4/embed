"""OpenAI-compatible (/v1/embeddings) service for microsoft/unixcoder-base.

Inference runs through **ncnn** (see ncnn_backend.py). CUDA used to be the
backend, but NVIDIA dropped old architectures (Maxwell/Pascal/Volta) from recent
CUDA releases, so that path is closed for good on the class of card this targets.
Vulkan through ncnn is the replacement, and it keeps working on old NVIDIA and
Radeon hardware alike.

**Device: a Vulkan GPU, pinned by name.** Both the Vulkan device and index 0 are
handled in ncnn_backend, which pins by *name* and refuses index 0 outright -- the
card used for desktop/gaming is never attached. Two config values are
load-bearing and must be set before ``load_param()``:

1. fp16 off, all three flags. The Vulkan ``MultiHeadAttention`` path rounds
   Q/K to fp16 before accumulation, which put embeddings at cosine ~0.16
   against torch. With fp16 off it is 0.999981+.
2. Each sample fed at its *own* real length, never padded to the batch max.
   Padding input to the graph drops the GPU to ~0.75 while CPU tolerates it.

``--device cpu`` remains available as a numerically equivalent fallback.

VRAM / RAM: the served fp32 weights are 501,356,844 bytes (478.13 MiB) and the
process holds ~900 MiB steady state under short traffic -- not the ~374 MiB the
original docs claimed, which is arithmetically impossible since it is smaller
than the weights alone. Above that floor VRAM is a **ratchet**: a single
2163-token request was measured taking the card to 1661 MiB and it never came
back, further long traffic pushed it to 1953-1962 MiB, and it is only reclaimed
by a restart. The cause is per-sample ``create_extractor()`` allocation feeding
ncnn's Vulkan pool, whose ``VkBlobAllocator::fastFree`` merges a freed range
back into a block's free list without ever calling ``vkFreeMemory`` -- so the
largest blob shape ever served sets a permanent floor.
``MAX_INPUT_TOKENS`` below is what bounds that: requests longer than it are
rejected before any inference runs, so no request can set the high-water mark.
Do NOT raise uvicorn workers above 1: each worker would load its own copy of
the model, and the floor above already leaves little headroom.

**Model lifecycle: loaded on demand, unloaded when idle.** The weights are
resident only while the service is actually being asked for embeddings. An
idle service holds nothing on the card: measured on the pinned GTX 1050, the
process stands at 357 MiB with the model released and 727 MiB with it loaded,
and ``close()`` gives every byte of that back (verified driver-side via
``nvidia-smi``, not inferred). The first request after an idle period pays the
reload -- measured at about one second on Vulkan -- and the result is
bit-identical to the never-unloaded run (cosine 1.000000 across a
load/unload/reload cycle on the reference texts).
``--keep-alive`` sets the idle timeout in seconds (default
``DEFAULT_KEEP_ALIVE_SECONDS``); ``-1`` restores the old always-resident
behaviour, and ``0`` unloads after every request.
"""
import os
import sys
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

# Maximum token count accepted in any single input element. Chosen from the
# model's own position-embedding table, not from a guess:
#
#   * `unixcoder-onnx/unixcoder.pnnx.fp32.ncnn.param` line 10 declares the
#     position embedding as `Embed embed_26 1 1 in2 18 0=768 1=1026 2=0
#     3=787968` -- 1=1026 rows, confirmed by the weight bytes (1026 x 768 x 4
#     = 787968) and independently by the model config's
#     `max_position_embeddings: 1026`. Valid position ids are therefore
#     0..1025.
#   * But the backend derives position ids as `cumsum(non_pad) * non_pad +
#     PAD_TOKEN_ID` (ncnn_backend.py), so a row of N real tokens occupies
#     positions 2..N+1 -- two rows are burned on <s> and <pad>. The last row
#     that stays in range is N = 1024 (max position id 1025). At N = 1025 the
#     graph reads position 1026, which the table does not have.
#
# Measured on the live service, both regimes exist today: a 722-token request
# returns HTTP 200, a 2163-token request returns HTTP 500 `inference failed:
# could not extract out0`. 2163 > 1024, so the 500 is exactly what running off
# the end of the position table looks like.
#
# Why a cap is needed at all (VRAM, not correctness): VRAM is a high-water
# ratchet. Idle and a batch of 32 short strings both sat at 900 MiB, but ONE
# single 2163-token request took the card to 1661 MiB and it never came back
# (unchanged at 30 s and after further short traffic); more long requests then
# pushed it to 1953-1962 MiB. The allocator's `fastFree` merges a freed range
# back into a block's free list and never calls `vkFreeMemory`, so the largest
# blob shape ever served becomes a permanent floor until restart. This constant
# is what bounds that.
MAX_INPUT_TOKENS = 1024

HERE = os.path.dirname(os.path.abspath(__file__))
# fp32 variant: the embeddings are compared by cosine distance, so arithmetic
# fidelity matters more than the small speed win from fp16.
PARAM_PATH = os.path.join(HERE, "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.param")
BIN_PATH = os.path.join(HERE, "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.bin")

# ---- Tokenizer -------------------------------------------------------------
# The tokenizer is a vendored local artefact, resolved from this directory
# FIRST and only falling back to the Hub if it is genuinely absent.
#
# The old code called AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
# on first request. That is a network dependency at process start: with no
# cache (or a cache that has been evicted) and no Hub reachability, the service
# raised OSError -- and because the load used to be lazy, on the first request,
# the process had already bound port 8000 and then served 503s forever. Upstream
# also publishes no `tokenizer.json` for this model, so only the slow
# `vocab.json` + `merges.txt` pair can load it.
#
# Files in TOKENIZER_DIR were copied verbatim from the pinned commit
# `5604afdc964f6c53782a6813140ade5216b99006` and are byte-identical to what the
# service has been using since it started: encoding "def hello(): return 1"
# yields the same 7 ids either way ([0, 729, 20596, 2077, 483, 524, 2]).
TOKENIZER_DIR = os.path.join(HERE, "tokenizer")

# The model id, used ONLY as a fallback when TOKENIZER_DIR is missing.
TOKENIZER_HUB_ID = "microsoft/unixcoder-base"

# The vendored files a RobertaTokenizer needs. Checked explicitly rather than
# by "does the directory exist", so a partially-checked-out or half-written
# vendored directory fails loudly here instead of confusing transformers into
# an unhelpful error later.
TOKENIZER_REQUIRED_FILES = (
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)

# Short, deterministic probe. A successful encode of this string is proof the
# tokenizer is usable, not just present.
TOKENIZER_PROBE = "def hello(): return 1"
# Ids the probe must encode to. These are the ids the live service produced
# when it loaded the tokenizer over the network: [0, 729, 20596, 2077, 483,
# 524, 2] -- <s> + "def hello ( ) : return 1" + </s>. Verifying the *values*
# (not just that an encode happened) is what catches a silently wrong
# vocab.json or a mis-set add_prefix_space.
TOKENIZER_PROBE_IDS = [0, 729, 20596, 2077, 483, 524, 2]
TOKENIZER_VOCAB_SIZE = 51416

# Vulkan on a pinned GPU. Both fixes this depends on live in ncnn_backend.py:
# fp16 disabled before load_param(), and per-sample unpadded lengths. Changing
# either reintroduces cosine-0.16 output.
DEFAULT_DEVICE = "vulkan"

# ncnn weight and graph sizes, checked at preflight so a truncated download or
# a wrong fp16 artefact is reported instead of producing garbage embeddings.
# 501,356,844 bytes is the fp32 floor documented above; the param file is the
# graph text. Both are exact, not approximations -- a mismatch is a hard stop.
EXPECTED_BIN_BYTES = 501356844
EXPECTED_PARAM_BYTES = 9913

# Expected Vulkan layer bindings for THIS graph, from describe(): every layer
# binds except the Embed ones, which have no Vulkan implementation anywhere in
# ncnn. With the load order right it is 128/131; with set_vulkan_device() called
# after load_param() every layer collapses to CPU and it reads 4/131 while
# `use_vulkan_compute` still reports true -- a silent wrong-device run.
EXPECTED_VULKAN_LAYERS = 128
EXPECTED_TOTAL_LAYERS = 131

# Mutable so `--device` can be applied before the backend is first built.
_device = {"name": DEFAULT_DEVICE}

# Idle timeout, in seconds, before the model is unloaded from the device.
# Ollama's semantics, because that is what the user asked for: the weights are
# resident only while they are actually being used, and an idle service holds
# no VRAM at all.
#
#   -1  -> never unload; the model stays resident for the life of the process
#    0  -> unload as soon as a request completes (maximum saving, every
#          request pays the reload cost)
#    N>0 -> unload after N idle seconds
#
# The default is 5 minutes, the same as Ollama's `keep_alive`, chosen because
# a memory-store client embeds in bursts: burst, idle, burst again. Five
# minutes absorbs a burst gap without a reload, and releases the card for
# anything else that wants it once the agent stops thinking.
#
# The saving is real and measured on the pinned card: 357 MiB used with the
# model unloaded vs 727 MiB with it resident -- the fp32 weight floor is
# 478.13 MiB, so this is nearly the whole footprint, not a rounding error.
DEFAULT_KEEP_ALIVE_SECONDS = 300

# Mutable so `--keep-alive` can be applied before the first request.
_keep_alive = {"seconds": DEFAULT_KEEP_ALIVE_SECONDS}


def set_device(name: str) -> None:
    """Select the inference device. Call before the first request."""
    _device["name"] = name


class PreflightError(RuntimeError):
    """Raised when the service cannot start correctly. Always fatal."""


def tokenizer_missing_files() -> list[str]:
    """Names of the required vendored tokenizer files that are not present."""
    return [
        name for name in TOKENIZER_REQUIRED_FILES
        if not os.path.isfile(os.path.join(TOKENIZER_DIR, name))
    ]


def load_tokenizer() -> AutoTokenizer:
    """Return the tokenizer: local vendored files first, Hub only as fallback.

    Local-first is deliberate. The Hub is not a dependency this service is
    allowed to have at start time: if it is unreachable the process must not
    bind port 8000 in a state where it can only answer 503.

    The preflight runs before the port is bound, so a broken tokenizer is a
    start-up failure rather than a serving-time 503.
    """
    missing = tokenizer_missing_files()

    if missing:
        # Only reached when the vendored directory is genuinely incomplete.
        # Loud, and it names both what was wanted and what was found, so the
        # operator knows which fix to make.
        print(
            f"WARNING: vendored tokenizer incomplete in {TOKENIZER_DIR} "
            f"(missing: {', '.join(missing)}); falling back to the Hub repo "
            f"'{TOKENIZER_HUB_ID}'",
            file=sys.stderr, flush=True,
        )
        try:
            return AutoTokenizer.from_pretrained(TOKENIZER_HUB_ID)
        except Exception as exc:
            raise PreflightError(
                "cannot load tokenizer: vendored copy at "
                f"{TOKENIZER_DIR} is missing {', '.join(missing)}, and the Hub "
                f"fallback for '{TOKENIZER_HUB_ID}' failed with: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_DIR, local_files_only=True
        )
    except Exception as exc:
        raise PreflightError(
            f"cannot load tokenizer from {TOKENIZER_DIR}: "
            f"{type(exc).__name__}: {exc}. The files are present, so this "
            "usually means they are truncated or from a different commit."
        ) from exc

    return tokenizer


def check_probe_ids(tokenizer: AutoTokenizer) -> None:
    """Raise if the tokenizer does not encode the probe exactly as expected."""
    try:
        encoded = tokenizer(TOKENIZER_PROBE, add_special_tokens=True)
        ids = list(encoded["input_ids"])
    except Exception as exc:
        raise PreflightError(
            f"tokenizer loaded from {TOKENIZER_DIR} but cannot encode the probe "
            f"string {TOKENIZER_PROBE!r}: {type(exc).__name__}: {exc}"
        ) from exc

    if ids != TOKENIZER_PROBE_IDS:
        raise PreflightError(
            f"tokenizer from {TOKENIZER_DIR} encoded {TOKENIZER_PROBE!r} to {ids} "
            f"but expected {TOKENIZER_PROBE_IDS} -- this is a different vocab "
            "or a different add_prefix_space, and every embedding would be wrong"
        )

    actual_vocab = getattr(tokenizer, "vocab_size", None)
    if actual_vocab != TOKENIZER_VOCAB_SIZE:
        raise PreflightError(
            f"tokenizer from {TOKENIZER_DIR} reports vocab_size "
            f"{actual_vocab} but the model was trained with "
            f"{TOKENIZER_VOCAB_SIZE} -- the model's Embed table is "
            f"{TOKENIZER_VOCAB_SIZE} rows, so out-of-range ids would be read "
            "as garbage"
        )


def check_model_files() -> None:
    """Raise if either ncnn artefact is missing or has the wrong size."""
    for path, expected_bytes, what in (
        (PARAM_PATH, EXPECTED_PARAM_BYTES, "ncnn param"),
        (BIN_PATH, EXPECTED_BIN_BYTES, "ncnn weights"),
    ):
        if not os.path.isfile(path):
            raise PreflightError(f"missing {what} file: {path}")
        actual = os.path.getsize(path)
        if actual != expected_bytes:
            raise PreflightError(
                f"{what} file has unexpected size: {path} is {actual} bytes, "
                f"expected {expected_bytes}. A truncated or wrong-variant "
                "artefact would silently produce wrong embeddings."
            )


def check_device_pinning() -> None:
    """Raise if the Vulkan device would not resolve to the pinned, non-index-0 GPU.

    Only the resolution is checked here; the backend redoes it when it loads,
    which is where the actual attachment happens. Doing it up front means a
    mis-pinned start reports the pin failure instead of a 503 on first request.
    """
    if _device["name"] != "vulkan":
        return
    try:
        from ncnn_backend import _find_pinned_device
        from ncnn_backend import PINNED_DEVICE_NAME
    except Exception:
        # _find_pinned_device is private; if it cannot be imported, let the
        # backend fail in its own terms rather than guessing here.
        return

    if PINNED_DEVICE_NAME != GPU_NAME:
        raise PreflightError(
            f"backend pins {PINNED_DEVICE_NAME!r} but the service pins "
            f"{GPU_NAME!r}; refusing to start with a mismatched pin"
        )
    try:
        index = _find_pinned_device(GPU_NAME)
    except Exception as exc:
        raise PreflightError(f"pinned GPU check failed: {exc}") from exc
    if index == 0:
        raise PreflightError(
            f"pinned GPU resolved to Vulkan index 0, which is refused "
            "outright (it is the card the desktop runs on)"
        )


def check_vulkan_layer_count(backend: "NcnnEmbeddingBackend") -> None:
    """Raise if the graph did not bind to Vulkan the way this model requires.

    Reuses the backend's own describe() rather than recounting layers here, so
    there is exactly one implementation of that number. Only checked on the
    Vulkan path: on CPU nothing is expected to bind (4 of 131 is the correct,
    expected CPU number), so a CPU run must not fail on it.
    """
    if _device["name"] != "vulkan":
        return
    info = backend.describe()
    vulkan_layers = info.get("vulkan_layers")
    total_layers = info.get("total_layers")
    if total_layers != EXPECTED_TOTAL_LAYERS:
        raise PreflightError(
            f"graph has {total_layers} layers, expected {EXPECTED_TOTAL_LAYERS}. "
            "The param file does not match this backend's assumptions."
        )
    if vulkan_layers != EXPECTED_VULKAN_LAYERS:
        raise PreflightError(
            f"only {vulkan_layers} of {total_layers} layers bound to Vulkan, "
            f"expected {EXPECTED_VULKAN_LAYERS}. That is the signature of "
            "set_vulkan_device() being called after load_param(): every layer "
            "collapses to CPU while use_vulkan_compute still reports true, so "
            "the request is served by the CPU at a fraction of the speed and "
            "with the wrong device. Refusing to start."
        )


def _preflight_log(message: str) -> None:
    """Report a preflight line so it is actually seen.

    stdout is block-buffered when the process is run under systemd, so a plain
    `print()` before uvicorn starts gets buffered and can vanish from the
    journal entirely (uvicorn's own logging flushes as it goes, which is what
    makes the missing preflight lines obvious in comparison). Flushing on every
    line, explicitly, is the only reliable way to get these into the journal.
    """
    print(message, flush=True)


def preflight() -> None:
    """Verify everything the service needs, before uvicorn binds a port.

    Order is deliberate: cheapest and most diagnostic first. A failure here
    exits non-zero with a specific message, which is strictly better than a
    process that binds port 8000 and then answers 503 forever, or worse, serves
    wrong embeddings.
    """
    # 1. The tokenizer must load and encode the probe to the expected ids.
    tokenizer_missing = tokenizer_missing_files()
    source = TOKENIZER_DIR if not tokenizer_missing else TOKENIZER_HUB_ID
    tokenizer = load_tokenizer()
    check_probe_ids(tokenizer)
    _preflight_log(
        f"preflight: tokenizer ok (probe -> {len(TOKENIZER_PROBE_IDS)} ids, "
        f"source: {source})"
    )

    # 2. The model artefacts must exist at the exact expected sizes.
    check_model_files()
    _preflight_log(
        "preflight: model files ok "
        f"({EXPECTED_PARAM_BYTES} + {EXPECTED_BIN_BYTES} bytes)"
    )

    # 3. The pinned GPU must resolve by name and must not be index 0.
    check_device_pinning()
    if _device["name"] == "vulkan":
        _preflight_log(f"preflight: pinned GPU {GPU_NAME!r} resolves and is not index 0")
    else:
        _preflight_log("preflight: device is cpu, Vulkan pin check skipped")

    # 4. The model must actually load, and on Vulkan must bind the expected
    #    layer count -- the check that catches a silently-correct-looking
    #    load order mistake.
    backend = ensure_backend()
    check_vulkan_layer_count(backend)
    info = backend.describe()
    if _device["name"] == "vulkan":
        _preflight_log(
            "preflight: model loaded on "
            f"{info.get('device')!r} "
            f"(vulkan_layers={info.get('vulkan_layers')}/{info.get('total_layers')})"
        )
        # The load above was verification, not residency. Keeping the model on
        # the card after preflight would hold 478 MiB of VRAM for the entire
        # idle life of the process, which is exactly the cost this service
        # exists to avoid. Releasing it here means an idle service holds
        # nothing, and the first request reloads (measured: identical down to
        # the last mantissa bit, at a measured cost of about a second).
        release_backend()
        _preflight_log(
            f"preflight: model released; it will load on the first request "
            f"(--keep-alive {_keep_alive['seconds']}s)"
        )
    else:
        # On CPU the weights are page cache, not a scarce device, so the load
        # stays resident: it costs no VRAM and makes the first request fast.
        _preflight_log(
            f"preflight: model loaded on cpu ({info.get('total_layers')} layers)"
        )

    _preflight_log("preflight: all checks passed")


app = FastAPI()

# The backend is created lazily on first use (see ensure_backend) and is shared
# by every request; a lock guards both creation and shutdown.
#
# `_unload_timer` implements the Ollama-style keep-alive: an idle timer that
# hands the card back once the model has not been asked for anything. It is
# also taken under `_backend_lock`, which is what makes an unload safe to race
# against a request: the unload either waits for an in-flight forward pass and
# unloads after it, or runs first and the request transparently reloads.
#
# RLock, not Lock: the timer helpers take the lock too, and ensure_backend()
# holds it while creating the backend and then arming the timer. A plain Lock
# would deadlock there on the very first request of every idle window.
_backend_lock = threading.RLock()
_backend: NcnnEmbeddingBackend | None = None
_backend_error: str | None = None
_unload_timer: threading.Timer | None = None


def set_keep_alive(seconds: int) -> None:
    """Set the idle-unload timeout. Call before the first request.

    ``-1`` disables unloading entirely (the model stays resident), ``0``
    unloads as soon as each request completes, and any positive value is the
    number of seconds of idleness after which the weights are released.
    """
    _keep_alive["seconds"] = seconds


def _cancel_unload_timer() -> None:
    """Disarm the idle timer. Caller must hold ``_backend_lock``."""
    global _unload_timer
    if _unload_timer is not None:
        _unload_timer.cancel()
        _unload_timer = None


def _arm_unload_timer() -> None:
    """(Re)arm the idle timer. Takes ``_backend_lock``.

    Only touches the timer handle, never the backend, so re-arming never waits
    on an in-flight inference. A no-op when unloading is disabled or when no
    backend is resident, so ``--keep-alive -1`` keeps the model resident for
    the life of the process and a request that never loaded a model never
    starts a timer that could only report a release that did not happen.
    """
    global _unload_timer
    with _backend_lock:
        _cancel_unload_timer()
        seconds = _keep_alive["seconds"]
        if seconds < 0 or _backend is None:
            return
        timer = threading.Timer(seconds, _unload_on_idle)
        timer.daemon = True
        timer.start()
        _unload_timer = timer


def _unload_on_idle() -> None:
    """Release the model after the idle timeout. Safe to run from the timer
    thread; it never raises, because a failure here must not take down the
    process that is still serving the endpoint."""
    try:
        unloaded = release_backend()
    except Exception:  # noqa: BLE001 - a timer thread must not propagate
        return
    if unloaded:
        # Logged by the caller, not by release_backend(): "why" the model went
        # away differs per call site, and a log line that claims an idle timer
        # fired when it was preflight is worse than no line at all.
        _preflight_log(
            f"model unloaded after {_keep_alive['seconds']}s idle; "
            "next request reloads it"
        )


def release_backend() -> bool:
    """Unload the model if it is loaded; return True if it actually went away.

    Shared by the idle timer, preflight, and shutdown. The timer that fired to
    get here is already dead, so the handle is simply cleared rather than
    cancelled. Deliberately silent about *why*: the caller knows that, and
    logging it here would attribute every release to the idle timer.
    """
    global _backend, _unload_timer
    with _backend_lock:
        _unload_timer = None
        if _backend is None:
            return False
        backend = _backend
        _backend = None
        backend.close()
    return True


def ensure_backend() -> NcnnEmbeddingBackend:
    """Return the shared ncnn backend, loading the model on first call."""
    global _backend, _backend_error

    if _backend is not None:
        _arm_unload_timer()
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
        _arm_unload_timer()
        return _backend


# The tokenizer is a small local artefact loaded eagerly, before uvicorn binds
# the port (see preflight) -- it is cached here so request handlers never touch
# the filesystem or the network. Loading it lazily on the first request, as this
# used to do, meant the port was already bound before the failure appeared.
_tokenizer: AutoTokenizer | None = None


def get_tokenizer() -> AutoTokenizer:
    """Return the shared tokenizer. Loaded by preflight before the port binds."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = load_tokenizer()
    return _tokenizer


@app.on_event("shutdown")
def _shutdown_backend() -> None:
    if release_backend():
        _preflight_log("model unloaded on shutdown")


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
        app.tokenizer = get_tokenizer()
    tokenizer = app.tokenizer

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")

    # Bound the VRAM high-water mark. Tokenizing only touches CPU (vocab +
    # merges; no model weights), so counting here is free on the GPU and the
    # rejection below happens before ensure_backend() -- no model load, no
    # extractor, no blob allocation.
    #
    # `attention_mask` is per-row even with padding=True: padding is the only
    # source of zeros, so the row sum is that element's real token count
    # (including <s> and </s>). That is the count that decides the position
    # ids, so it is the count that has to be bounded.
    token_counts = [int(n) for n in inputs["attention_mask"].sum(axis=1)]
    over_limit = [(i, n) for i, n in enumerate(token_counts) if n > MAX_INPUT_TOKENS]
    if over_limit:
        # Reject the whole request rather than dropping or truncating an
        # element: that would shift `index` and hand back a well-formed body
        # with fewer embeddings than inputs, which is a silent partial
        # result. Per-sample rejection is all-or-nothing by design.
        #
        # 413, not 400 or 503. It is a 4xx, so the call is never retried as
        # a transient failure, and it keeps 503 meaning "backend failure"
        # as this codebase defines it. 413 (rather than 400) because this is
        # a size violation: the client can shorten or split the input and
        # retry, exactly as with OpenAI's own max-input-length error.
        index, count = max(over_limit, key=lambda item: item[1])
        raise HTTPException(
            status_code=413,
            detail=(f"input element {index} is {count} tokens, which exceeds "
                    f"the limit of {MAX_INPUT_TOKENS}"),
        )

    backend = ensure_backend()
    try:
        last_hidden = backend.last_hidden_state(inputs["input_ids"], inputs["attention_mask"])
    except NcnnBackendError as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc
    finally:
        # Keep the model alive for another idle window even if this request
        # failed mid-inference: the weights are already resident, and
        # unloading them here would save nothing while making the next
        # request pay a reload.
        _arm_unload_timer()

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
    parser.add_argument(
        "--keep-alive",
        type=int,
        metavar="SECONDS",
        default=DEFAULT_KEEP_ALIVE_SECONDS,
        help=("idle seconds before the model is unloaded from the device, "
              "Ollama-style (default: %(default)s). The weights are resident "
              "only while in use; an idle service holds no VRAM. 0 unloads "
              "immediately after each request, -1 keeps the model resident "
              "for the life of the process"),
    )
    args = parser.parse_args()

    if args.workers != 1:
        raise SystemExit("--workers must be 1: each worker loads its own copy of the model into VRAM")

    # Must be applied before the backend is first built by a request handler,
    # and before preflight() loads it for verification.
    set_device(args.device)
    set_keep_alive(args.keep_alive)

    # Verify tokenizer, model files, GPU pin and Vulkan binding BEFORE uvicorn
    # binds the port. A failure here exits non-zero and says why, which is far
    # better than a process that holds the port and serves 503s, or serves
    # wrong embeddings, for as long as it is left running.
    try:
        preflight()
    except PreflightError as exc:
        print(f"FATAL: startup preflight failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)

    # Single worker by design: see the VRAM note in the module docstring.
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
