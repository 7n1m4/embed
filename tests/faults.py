"""Fault-injection fixtures for preflight / startup tests.

These tests need to start the real ``embedding_service.py`` with *one* thing
broken, in an isolated copy that cannot touch the live service, the real
model files, or the real tokenizer. Everything is copied into a temp dir
under ``tests/_faulttmp/`` (never ``/tmp``) and removed afterwards.

The trick that makes the copy runnable at all: ``embedding_service.py`` and
``ncnn_backend.py`` are thin (30 KB total, no model data) and import each other
by module name, so the copy needs only the Python sources -- and a
``sitecustomize.py`` shim (``tests/shim/``) that puts the real repo on
``sys.path`` so ``ncnn_backend``, ``ncnn``, ``numpy`` and ``transformers``
still resolve while the copy's own ``__main__`` drives its own constants.

The model files are never copied. Instead each fault test points the copy's
``EXPECTED_*`` constants at the *real* files via ``--real`` paths, so the copy
loads the same 478 MiB of weights from ``unixcoder-onnx/`` -- read-only, and
with no second copy on disk. That is also why the size checks can be broken
without touching the originals.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SHIM = os.path.join(HERE, "shim")
# Scratch lives inside the repo's tests/ dir, never /tmp (explicit constraint).
TMP = os.path.join(HERE, "_faulttmp")


def tmpdir(name: str) -> str:
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def cleanup() -> None:
    if os.path.isdir(TMP):
        shutil.rmtree(TMP, ignore_errors=True)


def _dropin_copy(dest: str) -> None:
    """Copy the importable Python sources of the service into ``dest``."""
    for name in ("embedding_service.py", "ncnn_backend.py", "convert_pnnx.py"):
        src = os.path.join(REPO, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
    # A tokenizer directory that the copy owns: same vendored bytes, but the
    # copy can be mutated to simulate a broken tokenizer without any risk to
    # the real one.
    shutil.copytree(
        os.path.join(REPO, "tokenizer"), os.path.join(dest, "tokenizer")
    )


def _patch_const(path: str, name: str, value: str) -> None:
    """Rewrite one ``NAME = ...`` constant in a copied python file."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    needle = f"{name} = "
    for i, line in enumerate(lines):
        if line.startswith(needle) and not line.lstrip().startswith(("#",)):
            indent = " " * (len(line) - len(line.lstrip()))
            lines[i] = f"{indent}{needle}{value}"
            break
    else:
        raise AssertionError(f"constant {name} not found in {path}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def fresh(dest_name: str, bind_port: int, real_model: bool = True) -> str:
    """Create a fresh, runnable copy of the service.

    With ``real_model=True`` (the default) the copy's ``PARAM_PATH``/``BIN_PATH``
    are pointed at the *real* artefacts in the repo, so the copy loads the same
    478 MiB of weights from disk without copying them. The originals are only
    ever read. Faults that need a *damaged* or *wrongly-sized* artefact then
    break the copy's own ``EXPECTED_*`` constants or a stub path, never the
    repo's files.
    """
    dest = tmpdir(dest_name)
    _dropin_copy(dest)
    # The copy never binds the service port 8000/8001; use a throwaway one so
    # the "port never bound" assertion is meaningful and collision-free.
    _patch_const(os.path.join(dest, "embedding_service.py"), "TOKENIZER_DIR",
                 repr(os.path.join(dest, "tokenizer")))
    if real_model:
        _patch_const(
            os.path.join(dest, "embedding_service.py"), "PARAM_PATH",
            repr(os.path.join(REPO, "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.param"))
        )
        _patch_const(
            os.path.join(dest, "embedding_service.py"), "BIN_PATH",
            repr(os.path.join(REPO, "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.bin"))
        )
    with open(os.path.join(dest, "PORT"), "w") as fh:
        fh.write(str(bind_port))
    return dest


def _port_is_listening(port: int, host: str = "127.0.0.1") -> bool:
    """Independent check that something is (still) accepting on ``port``."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.75)
        try:
            sock.connect((host, port))
        except OSError:
            return False
    return True


def run_service(dest: str, extra_args=(), env_extra=None, timeout=60.0):
    """Run the copy's ``embedding_service.py`` and capture the outcome.

    Returns a dict with ``returncode``, ``stdout``, ``port_bound`` (the
    service's own uvicorn banner appeared) and ``port_listening`` (an
    independent TCP connect succeeded). The process is always terminated
    before returning; a process that survives past ``timeout`` is killed and
    its ``returncode`` is negative, which the caller treats as a failure.
    """
    env = dict(os.environ)
    # Only the shim directory, so nothing in the parent environment can mask
    # the copy's own files.
    env["PYTHONPATH"] = SHIM
    env.pop("PYTEST_CURRENT_TEST", None)
    if env_extra:
        env.update(env_extra)

    port = int(open(os.path.join(dest, "PORT")).read())
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(dest, "embedding_service.py"),
         "--device", "cpu", "--host", "127.0.0.1", "--port", str(port)]
        + list(extra_args),
        cwd=dest, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    started = time.time()
    port_bound = False
    out = ""
    try:
        # Read lines until the process either exits, announces uvicorn is
        # serving, or overruns the deadline. A preflight failure is the
        # common case and exits in ~5 s; the healthy run would serve forever,
        # so the banner is also the stop signal.
        while True:
            if proc.poll() is not None:
                # Drain whatever is left without blocking.
                try:
                    out += proc.stdout.read() or ""
                except Exception:
                    pass
                break
            line = proc.stdout.readline()
            if line:
                out += line
                if "Uvicorn running on" in line:
                    port_bound = True
                    break
            elif time.time() - started > timeout:
                proc.kill()
                break
            elif _port_is_listening(port):
                # Fallback detection if the banner never arrives on the pipe.
                port_bound = True
                break
            else:
                time.sleep(0.02)
                if time.time() - started > timeout:
                    proc.kill()
                    break

        # Stop the process before doing anything else: nothing survives this
        # call, which is the whole point of the harness.
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        port_listening = _port_is_listening(port)
        return {
            "returncode": proc.returncode,
            "stdout": out,
            "stderr": "",
            "port_bound": port_bound,
            "port_listening": port_listening,
            "bind_wait_seconds": round(time.time() - started, 3),
            "pid": proc.pid,
            "port": port,
        }
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass


# ---- the faults ----------------------------------------------------------

def _with_param_size_broken(name: str, bind_port: int) -> str:
    """Copy whose expected param size no longer matches the real artefact."""
    dest = fresh(name, bind_port)
    real_param = os.path.join(REPO, "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.param")
    _patch_const(
        os.path.join(dest, "embedding_service.py"), "PARAM_PATH", repr(real_param)
    )
    _patch_const(
        os.path.join(dest, "embedding_service.py"),
        "EXPECTED_PARAM_BYTES", "12345",
    )
    return dest


def _with_bin_size_broken(name: str, bind_port: int) -> str:
    dest = _with_param_size_broken(name, bind_port)
    real_bin = os.path.join(REPO, "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.bin")
    _patch_const(
        os.path.join(dest, "embedding_service.py"), "BIN_PATH", repr(real_bin)
    )
    _patch_const(
        os.path.join(dest, "embedding_service.py"),
        "EXPECTED_BIN_BYTES", "12345",
    )
    return dest


def _with_bin_truncated(name: str, bind_port: int, keep: int = 4096) -> str:
    """Copy whose param passes but whose .bin is a truncated head of the real one."""
    dest = fresh(name, bind_port)
    real_bin = os.path.join(REPO, "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.bin")
    stub = os.path.join(dest, "truncated.bin")
    with open(real_bin, "rb") as src, open(stub, "wb") as dst:
        remaining = keep
        while remaining > 0:
            chunk = src.read(min(1 << 20, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    _patch_const(os.path.join(dest, "embedding_service.py"), "BIN_PATH", repr(stub))
    # Keep the param authentic; only the weights are damaged.
    return dest


def _with_missing_bin(name: str, bind_port: int) -> str:
    dest = fresh(name, bind_port)
    _patch_const(
        os.path.join(dest, "embedding_service.py"), "BIN_PATH",
        repr(os.path.join(dest, "does-not-exist.bin")),
    )
    return dest


def _with_tokenizer_empty(name: str, bind_port: int) -> str:
    """An empty vendored tokenizer dir.

    Two distinct behaviours matter here and both are asserted:
      * offline, the Hub fallback is unavailable and preflight must fail;
      * online, the fallback loads the *identical* upstream tokenizer and the
        service still starts and serves, but the log must carry the WARNING.
    """
    dest = fresh(name, bind_port)
    for fname in os.listdir(os.path.join(dest, "tokenizer")):
        os.remove(os.path.join(dest, "tokenizer", fname))
    with open(os.path.join(dest, "MODE"), "w") as fh:
        fh.write("tokenizer_empty")
    return dest


def _with_tokenizer_missing_files(name: str, bind_port: int, drop) -> str:
    dest = fresh(name, bind_port)
    for fname in drop:
        os.remove(os.path.join(dest, "tokenizer", fname))
    return dest


def _with_tokenizer_vocab_corrupt(name: str, bind_port: int) -> str:
    """A truncated vocab: the file is present but not valid JSON."""
    dest = fresh(name, bind_port)
    path = os.path.join(dest, "tokenizer", "vocab.json")
    with open(path) as fh:
        raw = fh.read()
    # Cut in the middle of the JSON, so the file exists and is non-empty but
    # cannot be parsed. The size check on the .bin/param does not apply to the
    # tokenizer, so this is exactly the "silently truncated vendored file" the
    # docs warn about.
    with open(path, "w") as fh:
        fh.write(raw[: len(raw) // 2])
    return dest


def _with_tokenizer_vocab_small(name: str, bind_port: int) -> str:
    """A *valid* vocab with far fewer entries: still loads, wrong size."""
    dest = fresh(name, bind_port)
    path = os.path.join(dest, "tokenizer", "vocab.json")
    with open(path) as fh:
        vocab = json.load(fh)
    keep = dict(sorted(vocab.items(), key=lambda kv: kv[1])[:3000])
    with open(path, "w") as fh:
        json.dump(keep, fh)
    return dest


def _with_tokenizer_prefix_space(name: str, bind_port: int) -> str:
    """A add_prefix_space flip: still loads, still right size, wrong ids."""
    dest = fresh(name, bind_port)
    path = os.path.join(dest, "tokenizer", "tokenizer_config.json")
    with open(path) as fh:
        cfg = json.load(fh)
    cfg["add_prefix_space"] = True
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return dest


def _with_tokenizer_merges_corrupt(name: str, bind_port: int) -> str:
    dest = fresh(name, bind_port)
    path = os.path.join(dest, "tokenizer", "merges.txt")
    with open(path, "a") as fh:
        fh.write("this line is not a merge\n")
    return dest


def _with_gpu_access_blocked(name: str, bind_port: int) -> str:
    """Force the Vulkan path with no devices: the pin must fail."""
    dest = fresh(name, bind_port)
    ncnn_mod = os.path.join(dest, "ncnn_backend.py")
    with open(ncnn_mod) as fh:
        src = fh.read()
    src = src.replace(
        "count = ncnn.get_gpu_count()",
        "count = 0 if os.environ.get('FAULT_NO_GPU') else ncnn.get_gpu_count()",
        1,
    )
    src = src.replace("import threading", "import os\nimport threading", 1)
    with open(ncnn_mod, "w") as fh:
        fh.write(src)
    return dest


def _with_vulkan_layer_count_wrong(name: str, bind_port: int) -> str:
    """A graph that loads but binds the wrong number of Vulkan layers."""
    dest = fresh(name, bind_port)
    svc = os.path.join(dest, "embedding_service.py")
    _patch_const(svc, "EXPECTED_VULKAN_LAYERS", "4")
    return dest


def _with_total_layers_wrong(name: str, bind_port: int) -> str:
    dest = fresh(name, bind_port)
    svc = os.path.join(dest, "embedding_service.py")
    _patch_const(svc, "EXPECTED_TOTAL_LAYERS", "99")
    return dest


def _with_probe_ids_wrong(name: str, bind_port: int) -> str:
    dest = fresh(name, bind_port)
    svc = os.path.join(dest, "embedding_service.py")
    _patch_const(
        svc, "TOKENIZER_PROBE_IDS", "[0, 729, 20596, 2077, 483, 524, 999]"
    )
    return dest


def _with_vocab_size_wrong(name: str, bind_port: int) -> str:
    dest = fresh(name, bind_port)
    svc = os.path.join(dest, "embedding_service.py")
    _patch_const(svc, "TOKENIZER_VOCAB_SIZE", "51417")
    return dest


FAULTS = {
    "pristine": lambda n, p: fresh(n, p),
    "param_size_mismatch": _with_param_size_broken,
    "bin_size_mismatch": _with_bin_size_broken,
    "bin_truncated": _with_bin_truncated,
    "bin_missing": _with_missing_bin,
    "tokenizer_empty": _with_tokenizer_empty,
    "tokenizer_missing_vocab": lambda n, p: _with_tokenizer_missing_files(
        n, p, drop=["vocab.json"]),
    "tokenizer_vocab_corrupt": _with_tokenizer_vocab_corrupt,
    "tokenizer_vocab_small": _with_tokenizer_vocab_small,
    "tokenizer_prefix_space": _with_tokenizer_prefix_space,
    "tokenizer_merges_corrupt": _with_tokenizer_merges_corrupt,
    "probe_ids_mismatch": _with_probe_ids_wrong,
    "vocab_size_mismatch": _with_vocab_size_wrong,
    "total_layers_mismatch": _with_total_layers_wrong,
    "vulkan_layers_mismatch": _with_vulkan_layer_count_wrong,
}
