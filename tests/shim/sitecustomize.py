"""Test-only sys.path shim.

Fault-injection tests launch `embedding_service.py` as a subprocess from a
temporary copy of the repo (see tests/faults.py). In the copy, `__main__` is
still `/tmp/.../embedding_service.py`, so Python puts the copy's directory on
sys.path[0] -- but the `ncnn_backend` module lives in the *real* repo. This
shim makes the real repo importable from the copy without touching the code
under test.

It is only ever loaded when PYTHONPATH names this directory explicitly, which
only the test runner does.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_REAL_REPO = os.path.abspath(os.path.join(_here, "..", ".."))
if _REAL_REPO not in sys.path:
    sys.path.insert(0, _REAL_REPO)
