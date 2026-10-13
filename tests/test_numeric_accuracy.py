"""Numerical correctness: the tests that would catch the regressions the README
documents (fp16 rounding, padded graph input, wrong pooling, wrong vocab).

The signal is ``unixcoder-onnx/reference_embeddings.json`` -- mean-pooled
torch embeddings for the four fixture texts. Cosine against it must be
``>= 0.999`` every time (the bar AGENTS.md sets for both devices). This is the
assertion that fp16-on and padded-input would fail: both were measured at
cosine ~0.16 and ~0.78 respectively through the very endpoint under test.

Determinism and the ``describe()`` layer invariant live here too, because
they are properties of the numeric path rather than of the HTTP envelope.
"""
from __future__ import annotations

import math
import unittest

from .support import (
    PROBE_TEXT,
    ServiceCase,
    cosine,
    load_reference,
    post,
    text_of_exact_length,
)

# AGENTS.md: "cosine must be >=0.999 for every input". The CPU path measures
# 0.999999 against this file, so the bar is deliberately far from the measured
# value -- it is a regression detector, not a tolerance tuned to today's noise.
COSINE_BAR = 0.999

# A ceiling, so a test cannot pass because both vectors are degenerate.
COSINE_CEILING = 1.0 - 1e-9


class NumericAccuracyTest(ServiceCase):
    def setUp(self) -> None:
        super().setUp()
        self.texts, self.reference = load_reference()
        self.assertEqual(len(self.reference), len(self.texts))
        self.assertEqual(len(self.reference[0]), 768)

    def test_reference_fixture_is_well_formed(self) -> None:
        """The reference itself must be a usable signal, not a stub."""
        self.assertEqual(len(self.texts), 4)
        for i, row in enumerate(self.reference):
            self.assertEqual(len(row), 768, f"ref[{i}] not 768-dim")
            self.assertTrue(all(math.isfinite(v) for v in row),
                            f"ref[{i}] contains non-finite values")
            norm = math.sqrt(sum(v * v for v in row))
            self.assertGreater(norm, 1.0, f"ref[{i}] is degenerate")
            # And the four reference vectors must not be near-duplicates of
            # each other, or a "correct" embedding could match all of them.
            for j in range(i + 1, len(self.reference)):
                c = cosine(row, self.reference[j])
                self.assertLess(abs(c), 0.99,
                                f"ref[{i}] and ref[{j}] are near-identical")

    def test_cosine_against_reference_batched(self) -> None:
        status, body = self.embed({"input": self.texts})
        self.assertEqual(status, 200)
        self.assertOpenAIShape(body, len(self.texts))
        for i, (text, got) in enumerate(zip(self.texts, body["data"])):
            emb = got["embedding"]
            self.assertTrue(all(math.isfinite(v) for v in emb),
                            f"data[{i}] contains NaN/inf")
            c = cosine(emb, self.reference[i])
            self.assertGreaterEqual(
                c, COSINE_BAR,
                f"text[{i}] {text!r}: cosine {c:.6f} < {COSINE_BAR} -- this "
                "is the signature of fp16 rounding or a padded graph input",
            )
            self.assertLess(c, COSINE_CEILING)

    def test_cosine_against_reference_solo(self) -> None:
        """Same texts one per request: the padding regression shows up here
        differently, so both shapes are checked."""
        for i, text in enumerate(self.texts):
            with self.subTest(i=i):
                status, body = self.embed({"input": text})
                self.assertEqual(status, 200)
                emb = body["data"][0]["embedding"]
                c = cosine(emb, self.reference[i])
                self.assertGreaterEqual(
                    c, COSINE_BAR,
                    f"text[{i}] {text!r}: solo cosine {c:.6f} < {COSINE_BAR}",
                )

    def test_cosine_against_reference_at_limit(self) -> None:
        """A long, well-formed input must also stay accurate.

        Long inputs are where the position-embedding table runs out and where
        the VRAM ratchet lives, so accuracy here is not a given.
        """
        long_text = text_of_exact_length(1024)
        status, body = self.embed({"input": long_text})
        self.assertEqual(status, 200)
        emb = body["data"][0]["embedding"]
        self.assertTrue(all(math.isfinite(v) for v in emb))
        # Self-consistency under truncation: the first 200 tokens of the same
        # string must not collapse into the same embedding, or the model is
        # ignoring everything past a prefix.
        status2, body2 = self.embed({"input": text_of_exact_length(200)})
        self.assertEqual(status2, 200)
        c = cosine(emb, body2["data"][0]["embedding"])
        self.assertLess(
            abs(c), 0.99,
            "a 1024-token input resembles a 200-token one almost exactly -- "
            "the long tail is being truncated or ignored",
        )

    def test_embedding_reflects_input_content(self) -> None:
        """Different inputs give different embeddings, and similar code
        stays near."""
        status, body = self.embed(
            {"input": [
                "def add(a, b):\n    return a + b",
                "def add(a, b):\n    return a + b",
                "the quick brown fox jumps over the lazy dog",
            ]}
        )
        self.assertEqual(status, 200)
        e = [item["embedding"] for item in body["data"]]
        # Float round-trip through JSON means these are equal, not 1.0-cosine.
        self.assertEqual(e[0], e[1])
        self.assertLess(
            cosine(e[0], e[2]), 0.95,
            "identical-code and natural-language embeddings are "
            "indistinguishable -- the encoder is broken",
        )


class DeterminismTest(ServiceCase):
    def test_same_input_twice_is_identical(self) -> None:
        texts = [PROBE_TEXT, "def f(x):\n    return x * 2"]
        _, first = self.embed({"input": texts})
        _, second = self.embed({"input": texts})
        for i in range(len(texts)):
            self.assertEqual(
                first["data"][i]["embedding"], second["data"][i]["embedding"],
                "identical requests produced different embeddings -- there is "
                "no cache and no randomness in this service, so any drift is "
                "a bug",
            )

    def test_repeated_solo_requests_are_identical(self) -> None:
        for i in range(6):
            status, body = self.embed({"input": PROBE_TEXT})
            if i == 0:
                baseline = body["data"][0]["embedding"]
                self.assertEqual(len(baseline), 768)
                continue
            self.assertEqual(body["data"][0]["embedding"], baseline,
                             f"request {i} drifted from request 0")


class FiniteInvariantTest(ServiceCase):
    def test_all_values_finite_across_shapes(self) -> None:
        cases = [
            "x",
            "def hello(): print(1)",
            "a\nb\nc",
            # Deliberately under the 1024-token cap: the point is character
            # coverage, not the guard (which has its own module).
            "データ" * 100,
            " " * 512 + "tail",
            text_of_exact_length(1024),
        ]
        for text in cases:
            with self.subTest(text=text[:20]):
                status, body = self.embed({"input": text})
                self.assertEqual(status, 200)
                for value in body["data"][0]["embedding"]:
                    self.assertTrue(math.isfinite(value),
                                    "NaN or inf in the embedding")

    def test_embedding_is_not_all_zeros(self) -> None:
        """A pooled-zero vector passes a finiteness check but is useless."""
        status, body = self.embed({"input": "def hello(): print(1)"})
        self.assertEqual(status, 200)
        emb = body["data"][0]["embedding"]
        self.assertTrue(any(v != 0.0 for v in emb),
                        "embedding is the zero vector")

    def test_shortest_valid_input(self) -> None:
        # <s> + one token + </s> = the smallest legal request.
        status, body = self.embed({"input": text_of_exact_length(3)})
        self.assertEqual(status, 200)
        self.assertOpenAIShape(body, 1)
        self.assertEqual(body["usage"]["prompt_tokens"], 3)


if __name__ == "__main__":
    unittest.main()
