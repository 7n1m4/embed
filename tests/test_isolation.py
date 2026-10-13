"""Process / isolation hygiene.

These guard the constraints the suite itself is run under, so they are the
tests that prove the suite did not do the thing it must never do:

* the live service on port 8000 is untouched, still serving, and still
  agrees with the torch reference (checked on the Vulkan path it runs on)
* the test instance is the ONLY service process, always ``--device cpu``,
  always port 8001
* no Vulkan device is attached anywhere in this test run
* when the run ends, port 8001 is free and no instance survives

The last two are the ones that make the rest trustworthy: a suite that left
a Vulkan-bound process running would be indistinguishable from a suite that
touched the user's GPU, and an "isolation" test that cannot fail would prove
nothing.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
import urllib.request

from .support import (
    LIVE_PORT,
    PROBE_TEXT,
    ServiceCase,
    TEST_PORT,
    _child_of,
    _cmdline,
    _is_ours,
    cosine,
    load_reference,
    live_service_rss_kb,
    memory,
    port_is_free,
    post,
    service_processes,
)


def _live_post(body: dict):
    req = urllib.request.Request(
        f"http://127.0.0.1:{LIVE_PORT}/v1/embeddings",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.status, json.loads(resp.read().decode())


class LiveServiceUntouchedTest(unittest.TestCase):
    """The live service must keep answering, unchanged."""

    def test_live_service_still_answers_on_8000(self) -> None:
        status, body = _live_post({"input": PROBE_TEXT})
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "list")
        self.assertEqual(len(body["data"][0]["embedding"]), 768)
        self.assertNotIn("model", body["data"][0])

    def test_live_service_matches_reference_on_its_vulkan_path(self) -> None:
        """Independent accuracy check of the SHIPPED configuration.

        This is the strongest available evidence that the test run did not
        perturb the live service: the same four reference texts must still
        score >= 0.999 cosine on Vulkan, which is the bar AGENTS.md sets.
        """
        texts, reference = load_reference()
        status, body = _live_post({"input": texts})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["data"]), len(texts))
        for i, (got, expected) in enumerate(zip(body["data"], reference)):
            c = cosine(got["embedding"], expected)
            self.assertGreaterEqual(
                c, 0.999,
                f"live service (Vulkan) ref[{i}] {texts[i]!r}: cosine "
                f"{c:.6f} < 0.999 -- the shipped configuration degraded",
            )

    def test_live_service_process_is_the_systemd_service(self) -> None:
        live = [p for p in service_processes()
                if str(LIVE_PORT) in _cmdline(p)]
        self.assertTrue(live, "no port-8000 service found -- the live service "
                              "is gone")
        for pid in live:
            self.assertIn(
                "--device", _cmdline(pid),
                f"live service {pid} has no --device flag: {_cmdline(pid)}",
            )
            self.assertIn(
                "vulkan", _cmdline(pid),
                f"live service {pid} is not running its shipped Vulkan "
                f"configuration: {_cmdline(pid)}",
            )


class SingleInstanceTest(unittest.TestCase):
    """Exactly one service process: the live one, plus at most one of ours."""

    def test_no_more_than_one_test_instance(self) -> None:
        mine = str(os.getpid())
        ours = [p for p in service_processes() if _child_of(p, mine)]
        # A test instance would be a child of this process tree.
        self.assertLessEqual(
            len(ours), 1,
            "more than one test service instance is alive: "
            + ", ".join(f"{p}:{_cmdline(p).strip()}" for p in ours),
        )

    def test_test_instance_is_cpu_only_and_on_8001(self) -> None:
        mine = str(os.getpid())
        for pid in service_processes():
            if not _child_of(pid, mine):
                continue  # the live service, not ours to assert on
            cmd = _cmdline(pid)
            self.assertIn(str(TEST_PORT), cmd,
                          f"test instance {pid} is not on port 8001: {cmd}")
            self.assertIn("--device cpu", cmd,
                          f"test instance {pid} is not CPU-only: {cmd}")
            self.assertNotIn("vulkan", cmd,
                             f"test instance {pid} mentions Vulkan: {cmd}")

    def test_test_instance_is_ours_and_our_child(self) -> None:
        """A stray instance left by an earlier run must not be mistaken
        for this run's instance."""
        mine = str(os.getpid())
        ours = [p for p in service_processes() if _child_of(p, mine)]
        if ours:
            for pid in ours:
                self.assertTrue(_is_ours(pid))


class NoVulkanAttachTest(unittest.TestCase):
    """Nothing in this test run may have a Vulkan library mapped in.

    The suite imports ncnn only to build CPU backends. If a Vulkan library
    were mapped into the test process, that would mean a device was created
    -- which is exactly what the constraints forbid. Read from /proc/PID/maps.
    """

    VULKAN_LIBS = ("libvulkan.so", "libnvidia", "libvkd3d", "libGLX_nvidia")

    def test_test_process_has_no_vulkan_mapping(self) -> None:
        mine = str(os.getpid())
        pids = [mine] + [p for p in service_processes() if _child_of(p, mine)]
        offenders = []
        for pid in pids:
            try:
                with open(f"/proc/{pid}/maps") as fh:
                    maps = fh.read()
            except OSError:
                continue
            for lib in self.VULKAN_LIBS:
                if lib in maps:
                    offenders.append((pid, lib))
                    break
        self.assertEqual(
            offenders, [],
            "a process in this test run has a Vulkan library mapped -- the "
            f"suite attached a GPU device: {offenders}",
        )


class MemoryFootprintTest(ServiceCase):
    """RAM is a stated constraint, so the cost of a heavy batch is reported
    as evidence rather than merely asserted."""

    def test_heavy_batch_cost_and_live_service_unchanged(self) -> None:
        before = memory()
        live_before = live_service_rss_kb()

        lengths = (3, 9, 64, 256, 512)
        payload = {"input": [" foo" * (n - 2) for n in lengths]}
        status, body = self.embed(payload)
        self.assertEqual(status, 200)
        self.assertOpenAIShape(body, len(lengths))

        after = memory()
        live_after = live_service_rss_kb()
        report = (
            f"MEM before/after: MemAvailable {before['MemAvailable']} -> "
            f"{after['MemAvailable']} MiB, SwapFree {before['SwapFree']} -> "
            f"{after['SwapFree']} MiB | live pid-8000 RSS {live_before} -> "
            f"{live_after} KiB"
        )
        print(report, flush=True)
        # The live service must not have grown because of this test run.
        self.assertLessEqual(
            live_after, live_before + 262144,
            "the live service grew by more than 256 MiB during the suite -- "
            "the test run interfered with it",
        )
        # And the box must not have been pushed into swap by one batch.
        self.assertGreaterEqual(
            after["MemAvailable"], 256,
            "MemAvailable fell under 256 MiB during a single test batch",
        )


def tearDownModule() -> None:
    # Nothing may survive: stop the instance even if a test above raised.
    ServiceCase.stop()
    mine = str(os.getpid())
    survivors = [p for p in service_processes() if _child_of(p, mine)]
    if survivors:
        raise AssertionError(
            "test service instances survived teardown: "
            + ", ".join(f"{p} ({_cmdline(p).strip()})" for p in survivors)
        )
    if not port_is_free(TEST_PORT):
        raise AssertionError(
            f"port {TEST_PORT} is still in use after the suite finished"
        )


if __name__ == "__main__":
    unittest.main()
