"""Test runner for the anima/embed suite.

Why not pytest: it is not installed on this machine and the constraints forbid
`pip install`, so the suite is stdlib ``unittest``. What is available and what
the suite uses:

    python 3.14.7, numpy 2.5.4, fastapi 0.141.1 + starlette (TestClient),
    httpx 0.28.1, transformers 5.19.0, ncnn (system), requests, uvicorn

Run it:

    cd /home/vi/embed && python tests/run_tests.py            # everything
    python tests/run_tests.py -k guard                        # filter
    python tests/run_tests.py -v                             # verbose
    python tests/run_tests.py -v tests/test_numeric_accuracy.py

It enforces the two safety constraints itself:

* exactly ONE service instance, `--device cpu`, port 8001; started once and
  killed at exit, even on failure or Ctrl-C;
* it refuses to start if port 8001 is already taken, so it can never open a
  second instance alongside a leftover one.
"""
from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import support  # noqa: E402
from support import ServiceCase, port_is_free  # noqa: E402

# Modules in dependency order. The HTTP-facing modules share one live
# instance; the preflight module spawns its own isolated copies and runs
# after them so the single-instance rule is never at risk mid-batch.
MODULES = [
    "tests.test_response_contract",
    "tests.test_numeric_accuracy",
    "tests.test_token_limit_guard",
    "tests.test_mean_pooling",
    "tests.test_isolation",
    "tests.test_preflight_faults",
]


def _cleanup() -> None:
    ServiceCase.stop()
    import shutil

    faults_cleanup = os.path.join(HERE, "_faulttmp")
    if os.path.isdir(faults_cleanup):
        shutil.rmtree(faults_cleanup, ignore_errors=True)
    if not port_is_free():
        support.kill_strays()
        deadline = time.time() + 10
        while time.time() < deadline and not port_is_free():
            time.sleep(0.25)
    if not port_is_free():
        print("WARNING: port 8001 is still in use after cleanup", file=sys.stderr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=None,
                        help="test module paths (default: the whole suite)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-k", dest="pattern", default=None,
                        help="unittest -k style pattern")
    args = parser.parse_args(argv)

    if not port_is_free():
        strays = support.kill_strays()
        deadline = time.time() + 10
        while time.time() < deadline and not port_is_free():
            time.sleep(0.25)
        if not port_is_free():
            if strays:
                message = (
                    f"ERROR: port 8001 is still in use after killing pids "
                    f"{strays}. Refusing to start a second instance."
                )
            else:
                message = (
                    "ERROR: port 8001 is in use and no test instance was "
                    "found to kill. Refusing to start a second instance."
                )
            print(message, file=sys.stderr)
            return 2

    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: (_cleanup(), sys.exit(130)))
        except (ValueError, OSError):
            pass

    print(f"MEM before: {support.memory()}", flush=True)
    print(f"live service RSS (port 8000): {support.live_service_rss_kb()} KiB",
          flush=True)

    testmodules = args.paths or MODULES
    start = time.time()
    loader = unittest.TestLoader()
    loader.testNamePatterns = [f"*{args.pattern}*"] if args.pattern else None
    suite = unittest.TestSuite()
    for name in testmodules:
        try:
            module = __import__(name, fromlist=["*"])
        except ImportError as exc:
            print(f"cannot import {name}: {exc}", file=sys.stderr)
            return 2
        suite.addTests(loader.loadTestsFromModule(module))

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    elapsed = time.time() - start

    print(f"elapsed: {elapsed:.1f}s", flush=True)
    print(f"MEM after: {support.memory()}", flush=True)
    print("port 8001 free:", port_is_free(), flush=True)
    print(f"live service RSS (port 8000): {support.live_service_rss_kb()} KiB",
          flush=True)

    _cleanup()
    atexit.unregister(_cleanup)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    sys.exit(main())
