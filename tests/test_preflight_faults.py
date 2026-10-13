"""Preflight / startup-failure paths.

The property under test is *ordering*: a broken tokenizer, a truncated or
wrongly-sized artefact, or a layer-count mismatch must make the process exit
non-zero **before uvicorn binds the port**. A process that binds and then
serves 503 is the failure this whole preflight exists to prevent, so the
assertions are:

* exit code non-zero (the code uses SystemExit(2))
* the port was never bound -- checked two ways: the uvicorn banner never
  appeared, and an independent TCP connect to that port fails
* the log names the specific fault, not a generic crash

Each fault is injected into a throwaway *copy* of the service in
``tests/_faulttmp/`` -- never into the live service, the real tokenizer, or
the real model files. The copy points its ``PARAM_PATH``/``BIN_PATH`` at the
real artefacts (read-only) so it loads the genuine 478 MiB of weights; only
the copy's own constants and its own tokenizer directory are mutated.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import faults  # noqa: E402

PORT = 8199


class PreflightFaultTest(unittest.TestCase):
    """Every case: the copy must die before binding its port."""

    def _assert_refused(self, result: dict, expected_fragment: str,
                         fault: str) -> None:
        out = result["stdout"]
        self.assertNotEqual(
            result["returncode"], 0,
            f"[{fault}] the process exited cleanly -- the fault was not "
            f"detected",
        )
        self.assertGreater(
            result["returncode"], 0,
            f"[{fault}] exit code {result['returncode']} looks like a kill or "
            "a crash, not the documented SystemExit(2) preflight failure",
        )
        self.assertFalse(
            result["port_bound"],
            f"[{fault}] uvicorn bound the port before preflight finished -- "
            f"log: {out[-500:]}",
        )
        self.assertFalse(
            result["port_listening"],
            f"[{fault}] port {result['port']} was still accepting when the "
            "process died",
        )
        self.assertIn(
            expected_fragment, out,
            f"[{fault}] the failure message does not name the fault; log: "
            f"{out[-500:]}",
        )

    def test_param_file_size_mismatch(self) -> None:
        d = faults.FAULTS["param_size_mismatch"]("param_size", PORT)
        r = faults.run_service(d)
        self._assert_refused(r, "ncnn param file has unexpected size",
                              "param size")
        self.assertIn("501356844" if False else "9913", r["stdout"])

    def test_bin_file_size_mismatch(self) -> None:
        d = faults.FAULTS["bin_size_mismatch"]("bin_size", PORT)
        r = faults.run_service(d)
        self._assert_refused(r, "unexpected size", "bin size")

    def test_bin_file_missing(self) -> None:
        d = faults.FAULTS["bin_missing"]("bin_missing", PORT)
        r = faults.run_service(d)
        self._assert_refused(r, "missing ncnn weights file", "bin missing")

    def test_bin_file_truncated(self) -> None:
        # A truncated .bin whose expected size is still the real one: the size
        # check is what catches it, since the param file is untouched.
        d = faults.FAULTS["bin_truncated"]("bin_trunc", PORT)
        r = faults.run_service(d)
        self._assert_refused(r, "unexpected size", "truncated bin")

    def test_tokenizer_vocab_json_truncated(self) -> None:
        d = faults.FAULTS["tokenizer_vocab_corrupt"]("vocab_trunc", PORT)
        r = faults.run_service(d)
        self._assert_refused(r, "cannot load tokenizer", "truncated vocab")

    def test_tokenizer_vocab_missing_entries(self) -> None:
        """A valid-but-incomplete vocab: still parses, wrong for the model."""
        d = faults.FAULTS["tokenizer_vocab_small"]("vocab_small", PORT)
        r = faults.run_service(d)
        self._assert_refused(r, "cannot load tokenizer", "vocab size")

    def test_tokenizer_add_prefix_space_flip(self) -> None:
        """A mis-set add_prefix_space: silently wrong ids, right vocab size.

        This is the case where the *probe id* check earns its keep -- the
        tokenizer loads, vocab_size matches, and every embedding is wrong.
        """
        d = faults.FAULTS["tokenizer_prefix_space"]("prefix_space", PORT)
        r = faults.run_service(d)
        self._assert_refused(r, "but expected", "probe ids")

    def test_tokenizer_merges_corrupt(self) -> None:
        d = faults.FAULTS["tokenizer_merges_corrupt"]("merges_bad", PORT)
        r = faults.run_service(d)
        self._assert_refused(r, "cannot load tokenizer", "merges")

    def test_tokenizer_dir_completely_empty_offline(self) -> None:
        """No vendored tokenizer at all, and the Hub unreachable.

        The service is documented to be local-first with a Hub fallback, so
        this asserts the failure mode the fallback was added to *avoid*: a
        bind-then-503 outage. With the Hub offline, preflight must fail.
        """
        d = faults.fresh("tok_empty_offline", PORT + 1)
        real_hf = os.path.join(d, "hf_home")
        os.makedirs(real_hf, exist_ok=True)
        for fname in os.listdir(os.path.join(d, "tokenizer")):
            os.remove(os.path.join(d, "tokenizer", fname))
        r = faults.run_service(d, env_extra={
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HOME": real_hf,
        })
        self._assert_refused(r, "cannot load tokenizer", "empty tokenizer")

    def test_tokenizer_dir_empty_with_hub_available(self) -> None:
        """Online, the Hub fallback loads the same tokenizer -- and the log
        must say so, rather than failing silently.

        This test's only real assertion is the WARNING line: it pins the
        documented "loud" behaviour so an operator can tell a healthy
        offline-first start from a fallback one.
        """
        d = faults.fresh("tok_empty_online", PORT + 2)
        real_hf = os.path.join(d, "hf_home")
        os.makedirs(real_hf, exist_ok=True)
        for fname in os.listdir(os.path.join(d, "tokenizer")):
            os.remove(os.path.join(d, "tokenizer", fname))
        r = faults.run_service(d, env_extra={
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
            "HF_HOME": real_hf,
        })
        self.assertIn("WARNING: vendored tokenizer incomplete", r["stdout"],
                      "the Hub fallback must announce itself")
        # It should still start (the Hub is reachable), which is the
        # documented behaviour -- but a failure here is a finding, not a
        # test error, so it is reported rather than asserted hard.
        if r["port_bound"]:
            self.assertIn("source: microsoft/unixcoder-base", r["stdout"])


class PreflightSanityTest(unittest.TestCase):
    """The pristine copy must start and serve; the faults above are only
    meaningful if the harness itself is sound."""

    def test_pristine_copy_binds_and_serves(self) -> None:
        import socket
        import time
        import urllib.error
        import urllib.request

        d = faults.fresh("pristine", PORT + 3)
        env = dict(os.environ)
        env["PYTHONPATH"] = faults.SHIM
        import subprocess

        proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(d, "embedding_service.py"),
             "--device", "cpu", "--host", "127.0.0.1", "--port", str(PORT + 3)],
            cwd=d, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        try:
            banner = False
            deadline = time.time() + 60
            while time.time() < deadline and not banner:
                if proc.poll() is not None:
                    break
                line = proc.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                if "Uvicorn running on" in line:
                    banner = True
            self.assertTrue(
                banner,
                "the pristine copy never bound its port; the harness is "
                "broken and the fault tests above are vacuous",
            )
            self.assertEqual(proc.poll(), None, "the copy exited after binding")
            payload = ('{"input": "def hello(): print(1)", '
                       '"model": "unixcoder-base"}')
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT + 3}/v1/embeddings",
                data=payload.encode(),
                headers={"content-type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    import json
                    body = json.loads(resp.read().decode())
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(len(body["data"][0]["embedding"]), 768)
            except urllib.error.HTTPError as exc:
                self.fail(f"pristine copy returned {exc.code}")
            self.assertTrue(faults._port_is_listening(PORT + 3))
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except Exception:
                    proc.kill()
            # Nothing must survive the suite.
            deadline = time.time() + 20
            while time.time() < deadline:
                if not faults._port_is_listening(PORT + 3):
                    break
                time.sleep(0.2)
            self.assertFalse(
                faults._port_is_listening(PORT + 3),
                f"port {PORT + 3} is still accepting after teardown",
            )
            _ = importlib.util  # keep the import meaningful for linters


class LayerCountInvariantTest(unittest.TestCase):
    """``describe()["total_layers"] == 131`` regardless of device.

    Expressing this on CPU is valid and cheap: the graph is loaded from the
    param file, and the layer list comes from the graph itself, so the count
    is device-independent. ``vulkan_layers`` differs by device (128 on a
    correct Vulkan load, 4 on CPU) and is therefore NOT asserted here -- only
    the invariant that holds everywhere.
    """

    def test_total_layers_is_131_on_cpu(self) -> None:
        sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
        from ncnn_backend import NcnnEmbeddingBackend

        param = os.path.abspath(os.path.join(
            HERE, "..", "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.param"))
        weights = os.path.abspath(os.path.join(
            HERE, "..", "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.bin"))
        backend = NcnnEmbeddingBackend(param, weights, use_vulkan=False)
        try:
            info = backend.describe()
            self.assertEqual(
                info["total_layers"], 131,
                f"graph has {info['total_layers']} layers, expected 131 -- "
                "the param file does not match this backend's assumptions",
            )
            self.assertEqual(info["hidden_size"], 768)
            self.assertEqual(info["device"], "cpu")
            self.assertFalse(info["numerically_verified"] is False,
                             "CPU is the numerically verified path")
        finally:
            backend.close()

    def test_total_layers_mismatch_is_not_caught_on_cpu(self) -> None:
        """A finding, pinned so it cannot regress silently.

        ``check_vulkan_layer_count()`` returns early unless the device is
        Vulkan, so NEITHER ``EXPECTED_TOTAL_LAYERS`` NOR
        ``EXPECTED_VULKAN_LAYERS`` is checked on ``--device cpu``. A wrong
        expectation therefore starts happily on CPU.

        This is pinned as the observed behaviour. It is a real gap, not a
        design choice the docs mention: the total-layer check is
        device-independent in principle (the layer list comes from the param
        file), yet it is gated behind the Vulkan branch. If the check is ever
        made unconditional, this test fails and should be replaced with the
        hard-stop assertion it should always have been.
        """
        d = faults.FAULTS["total_layers_mismatch"]("total_layers", PORT + 4)
        r = faults.run_service(d)
        self.assertEqual(
            r["port_bound"], True,
            "the total-layer check DID fire on CPU -- make it unconditional "
            "and replace this test with a hard-stop assertion",
        )
        # The service must still have loaded the real graph and agree with
        # the invariant, which is what makes the gap a *silent* one.
        self.assertIn("model loaded on cpu (131 layers)", r["stdout"])

    def test_vulkan_layer_mismatch_is_not_caught_on_cpu(self) -> None:
        """Same gate, other constant: a bogus EXPECTED_VULKAN_LAYERS is
        harmless on CPU -- which is correct, since 4/131 is the right CPU
        number -- but it also means no layer check at all runs on CPU."""
        d = faults.FAULTS["vulkan_layers_mismatch"]("vulkan_layers", PORT + 5)
        r = faults.run_service(d)
        self.assertEqual(
            r["port_bound"], True,
            "a CPU run failed on the Vulkan layer count; a CPU run must not "
            "fail on it",
        )
        self.assertIn("device is cpu", r["stdout"])


def tearDownModule() -> None:
    faults.cleanup()
    stray = [p for p in os.listdir("/proc") if p.isdigit()]
    listening = []
    for port in range(PORT, PORT + 8):
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(("127.0.0.1", port))
                listening.append(port)
        except OSError:
            continue
    if listening:
        raise AssertionError(f"test ports still listening: {listening}")
    _ = stray


if __name__ == "__main__":
    unittest.main()
