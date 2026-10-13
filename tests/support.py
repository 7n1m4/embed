"""Shared fixtures for the anima/embed test suite.

Everything here is deliberately explicit and small:

* ``SERVICE`` launches exactly ONE `embedding_service.py --device cpu` on port
  8001, waits for it to answer, and guarantees it is dead before the module
  ends. It never touches port 8000 (the live service) and never starts a
  second instance concurrently.
* ``memory()`` samples MemAvailable/SwapFree so heavy tests can report the
  RAM they cost (RAM is a stated concern on this machine).
* ``reference`` is the torch reference embedding set in
  ``unixcoder-onnx/reference_embeddings.json``.

The suite is stdlib ``unittest`` (pytest is not installed on this machine).
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

SERVICE_PORT = 8001
TEST_PORT = SERVICE_PORT
SERVICE_HOST = "127.0.0.1"
# The live, systemd-managed, Vulkan service. Read-only for the whole suite.
LIVE_PORT = 8000
SERVICE_PROCESS_NAME = "embedding_service_test_instance"
REFERENCE_PATH = os.path.join(REPO, "unixcoder-onnx", "reference_embeddings.json")

# A single token, repeated. Used to build inputs of an exact token count:
# " foo" encodes to exactly one id, so N copies plus <s>/</s> is N + 2.
UNIT_TOKEN = " foo"
PROBE_TEXT = "def hello(): print(1)"

# Trailing slash matters: the service mounts the route as "/v1/embeddings".
BASE_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}"


def text_of_exact_length(n: int) -> str:
    """A string the tokenizer encodes to exactly ``n`` ids."""
    if n < 3:
        raise ValueError("minimum representable length is 3 (<s>, token, </s>)")
    return UNIT_TOKEN * (n - 2)


def memory() -> dict:
    """MemAvailable/SwapFree in MiB, straight from /proc/meminfo."""
    out = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, value = line.partition(":")
            if key in ("MemAvailable", "SwapFree"):
                out[key] = int(value.strip().split()[0]) // 1024  # KiB -> MiB
    return out


def post(payload: dict, path: str = "/v1/embeddings", timeout: float = 300.0):
    """POST JSON to the service. Returns (status, parsed_body_or_raw_text)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path, data=body,
        headers={"content-type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, _maybe_json(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        return exc.code, _maybe_json(raw)


def get(path: str, timeout: float = 60.0):
    req = urllib.request.Request(BASE_URL + path, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _maybe_json(raw: str):
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def load_reference() -> tuple:
    """Return (texts, embeddings) from the torch reference file."""
    with open(REFERENCE_PATH) as fh:
        payload = json.load(fh)
    return list(payload["texts"]), [
        [float(v) for v in row] for row in payload["embeddings"]
    ]


_tok_cache = {}


def tokenizer():
    """The service's own tokenizer, loaded once from the vendored directory.

    Used as an independent oracle: any count the endpoint reports can be
    checked against ``len(tokenizer()(text)["input_ids"])`` rather than being
    assumed. Loaded with ``local_files_only`` so the test can never silently
    fall back to a different Hub snapshot.
    """
    tok = _tok_cache.get("tok")
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            os.path.join(REPO, "tokenizer"), local_files_only=True
        )
        _tok_cache["tok"] = tok
    return tok


def UnitTokenCount(text: str) -> int:
    """How many ids the tokenizer produces for ``text`` (with special tokens)."""
    enc = tokenizer()(text, add_special_tokens=True)
    return len(enc["input_ids"])


def cosine(a, b) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def port_is_free(port: int = SERVICE_PORT, host: str = SERVICE_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.75)
        try:
            sock.connect((host, port))
        except OSError:
            return True
    return False


def _cmdline(pid) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _status_field(pid, prefix: str) -> str:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith(prefix):
                    return line[len(prefix):].strip()
    except OSError:
        pass
    return ""


def _is_ours(pid) -> bool:
    return _status_field(pid, "Uid:").split()[0] == str(os.getuid()) \
        if _status_field(pid, "Uid:") else False


def _child_of(pid, root, seen=None) -> bool:
    """True if ``pid`` is ``root`` or a descendant of it."""
    pid = str(pid)
    if pid == str(root):
        return True
    seen = seen if seen is not None else set()
    if pid in seen or not pid.isdigit():
        return False
    seen.add(pid)
    ppid = _status_field(pid, "PPid:")
    return _child_of(ppid, root, seen) if ppid else False


def service_processes() -> list:
    """Every ``embedding_service.py`` process owned by this user."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        if "embedding_service.py" not in _cmdline(entry):
            continue
        if not _is_ours(entry):
            continue
        found.append(entry)
    return found


def live_service_pid() -> str | None:
    """The pid of the live, systemd-managed service on port 8000."""
    for pid in service_processes():
        cmd = _cmdline(pid)
        if "--port" in cmd and str(LIVE_PORT) in cmd:
            return pid
    return None


def live_service_rss_kb() -> int:
    """Total VmRSS of the live port-8000 service, in KiB."""
    total = 0
    for pid in service_processes():
        cmd = _cmdline(pid)
        if "--port" in cmd and str(LIVE_PORT) in cmd:
            rss = _status_field(pid, "VmRSS:")
            if rss:
                total += int(rss.split()[0])
    return total


def kill_strays() -> list:
    """Kill leftover TEST instances only (never the live service on 8000).

    A process is a stray only if it runs ``embedding_service.py`` on the test
    port 8001 and belongs to this user. The systemd-managed service on port
    8000 therefore can never match, whatever else is on its command line.
    """
    import signal

    victims = []
    for pid in service_processes():
        cmd = _cmdline(pid)
        if str(SERVICE_PORT) not in cmd:
            continue
        victims.append(int(pid))
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return victims


class ServiceCase(unittest.TestCase):
    """Base case that owns the single CPU service instance.

    The instance is started once for the whole run (``ensure_started``) and
    torn down by ``stop()``. Tests MUST NOT start their own instance.

    The process handle lives on this class, never on a subclass: a subclass
    attribute assignment would shadow it and make every module after the
    first believe no instance is running.
    """

    _proc = None
    _log = None

    @classmethod
    def ensure_started(cls) -> None:
        if ServiceCase._proc is not None:
            return
        assert port_is_free(), (
            f"port {SERVICE_PORT} is already in use -- refusing to start a "
            "second test instance"
        )
        env = dict(os.environ)
        env.pop("PYTEST_CURRENT_TEST", None)
        # A clean, private HF cache is not needed: the tokenizer loads from
        # the vendored local directory, offline-first.
        ServiceCase._log = open(os.path.join(HERE, "_service.log"), "w")
        ServiceCase._proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(REPO, "embedding_service.py"),
             "--device", "cpu", "--host", SERVICE_HOST,
             "--port", str(SERVICE_PORT)],
            cwd=REPO, env=env, stdout=ServiceCase._log, stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 120.0
        while time.time() < deadline:
            if ServiceCase._proc.poll() is not None:
                with open(os.path.join(HERE, "_service.log")) as fh:
                    tail = fh.read()[-2000:]
                raise AssertionError(
                    f"test service exited early with "
                    f"{ServiceCase._proc.returncode}:\n{tail}"
                )
            try:
                status, _ = post({"input": "warmup"}, timeout=10.0)
                if status == 200:
                    return
            except Exception:
                time.sleep(0.25)
        raise AssertionError("test service did not become ready in time")

    @classmethod
    def stop(cls) -> None:
        if ServiceCase._proc is None:
            return
        if ServiceCase._proc.poll() is None:
            ServiceCase._proc.terminate()
            try:
                ServiceCase._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                ServiceCase._proc.kill()
                ServiceCase._proc.wait(timeout=20)
        ServiceCase._proc = None
        if ServiceCase._log is not None:
            ServiceCase._log.close()
            ServiceCase._log = None

    def setUp(self) -> None:
        self.ensure_started()
        self.mem_before = memory()

    def tearDown(self) -> None:
        pass

    # ---- helpers ----------------------------------------------------------

    def embed(self, payload: dict):
        return post(payload)

    def assertOpenAIShape(self, body: dict, expected_count: int) -> None:
        """The full documented response contract."""
        self.assertIsInstance(body, dict)
        self.assertEqual(body.get("object"), "list")
        self.assertIn("model", body)
        self.assertIsInstance(body["model"], str)

        data = body.get("data")
        self.assertIsInstance(data, list)
        self.assertEqual(
            len(data), expected_count,
            f"expected exactly {expected_count} data entries, got {len(data)}",
        )
        for i, item in enumerate(data):
            self.assertIsInstance(item, dict)
            self.assertEqual(item.get("object"), "embedding")
            # Documented invariant: `model` is a top-level field only.
            self.assertNotIn(
                "model", item,
                "data[i] must not carry a 'model' key (documented invariant)",
            )
            emb = item.get("embedding")
            self.assertIsInstance(emb, list)
            self.assertEqual(len(emb), 768, f"data[{i}] is not 768-dim")
            self.assertEqual(
                item.get("index"), i,
                f"data[{i}].index={item.get('index')!r}",
            )
            for value in emb:
                self.assertIsInstance(value, float)

        usage = body.get("usage")
        self.assertIsInstance(usage, dict)
        self.assertIsInstance(usage.get("prompt_tokens"), int)
        self.assertIsInstance(usage.get("total_tokens"), int)
        self.assertGreaterEqual(usage["prompt_tokens"], 0)
        self.assertGreaterEqual(usage["total_tokens"], 0)
        self.assertEqual(usage["prompt_tokens"], usage["total_tokens"])
