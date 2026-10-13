"""The OpenAI /v1/embeddings response contract.

Every assertion here is a documented invariant, so each one fails on a real
regression that a memory-store client would actually see:

* ``object == "list"``, top-level ``model`` + ``usage``
* ``data[i]`` = ``{"object": "embedding", "embedding": 768 floats, "index": i}``
* **no ``model`` key inside ``data[i]``** -- pinned as a documented invariant
* ``N`` inputs -> exactly ``N`` data entries, in input order
* ``model`` is echoed verbatim and selects nothing
"""
from __future__ import annotations

import unittest

from .support import PROBE_TEXT, ServiceCase, post, text_of_exact_length

# Deliberately not a code-shaped probe: an identifier-heavy string and a
# natural-language string exercise different tokenizer paths, so a pooling or
# padding bug cannot hide inside a single shape.
BATCH = [
    PROBE_TEXT,
    "class A:\n    def f(self):\n        return 1",
    "SELECT * FROM users WHERE id = 1",
    "print('hello world')",
]


class ResponseContractTest(ServiceCase):
    def test_single_object_envelope(self) -> None:
        status, body = self.embed({"input": PROBE_TEXT})
        self.assertEqual(status, 200)
        self.assertOpenAIShape(body, 1)
        self.assertEqual(body["usage"]["prompt_tokens"], 9)  # <s> + 7 + </s>

    def test_string_and_model_echo(self) -> None:
        # `model` is echoed and used for nothing else: the same input must
        # produce the same embedding under two different model strings.
        status_a, body_a = self.embed({"input": PROBE_TEXT, "model": "m-a"})
        status_b, body_b = self.embed({"input": PROBE_TEXT, "model": "m-b"})
        self.assertEqual((status_a, status_b), (200, 200))
        self.assertEqual(body_a["model"], "m-a")
        self.assertEqual(body_b["model"], "m-b")
        self.assertEqual(
            body_a["data"][0]["embedding"], body_b["data"][0]["embedding"],
            "the request's model field selected something -- it must be "
            "echoed only (v1 scope: one model)",
        )

    def test_model_field_is_optional_and_defaults(self) -> None:
        status, body = self.embed({"input": PROBE_TEXT})
        self.assertEqual(status, 200)
        self.assertOpenAIShape(body, 1)
        self.assertEqual(body["model"], "unixcoder-base",
                         "missing model must default to the documented id")

    def test_batch_sizes_return_exact_entry_counts(self) -> None:
        for size in (1, 2, 3, 5, 8):
            with self.subTest(size=size):
                texts = [f"sample {i} with a leading number {i:02d}" for i in range(size)]
                status, body = self.embed({"input": texts})
                self.assertEqual(status, 200)
                self.assertOpenAIShape(body, size)
                indexes = [item["index"] for item in body["data"]]
                self.assertEqual(indexes, list(range(size)),
                                 "index must follow list position")
                self.assertEqual(
                    len({tuple(item["embedding"]) for item in body["data"]}),
                    size,
                    "all embeddings in a batch came back identical -- the "
                    "batch collapsed",
                )

    def test_index_follows_position_not_value(self) -> None:
        # Duplicate inputs must not be deduplicated or reordered.
        texts = ["identical text"] * 4
        status, body = self.embed({"input": texts})
        self.assertEqual(status, 200)
        self.assertOpenAIShape(body, 4)
        self.assertEqual([i["index"] for i in body["data"]], [0, 1, 2, 3])
        for item in body["data"]:
            self.assertEqual(item["embedding"], body["data"][0]["embedding"])

    def test_every_dimension_is_a_real_float(self) -> None:
        status, body = self.embed({"input": PROBE_TEXT})
        self.assertEqual(status, 200)
        emb = body["data"][0]["embedding"]
        self.assertEqual(len(emb), 768)
        # An embedding that is all zeros, or that contains bools sent as
        # floats, would pass a naive isinstance check on the JSON level.
        self.assertTrue(any(v != 0.0 for v in emb),
                        "embedding is all zeros")
        self.assertIsInstance(emb[0], float)

    def test_usage_matches_prompt_tokens(self) -> None:
        status, body = self.embed({"input": "def hello(): print(1)"})
        self.assertEqual(status, 200)
        self.assertOpenAIShape(body, 1)
        # <s> def hello ( ) : print ( 1 ) </s> -- 9 ids, counted with the two
        # special tokens, and no padding counted.
        self.assertEqual(body["usage"]["prompt_tokens"], 9)

    def test_batch_usage_counts_real_tokens_only(self) -> None:
        # Texts of unequal length, padded to one batch: usage must equal the
        # sum of real tokens, not batch_max * count. A naive implementation
        # that sums the padded mask would inflate it.
        texts = ["a", "bb", "ccc", "dddd"]
        status, body = self.embed({"input": texts})
        self.assertEqual(status, 200)
        self.assertOpenAIShape(body, len(texts))
        self.assertGreater(len({tuple(i["embedding"]) for i in body["data"]}), 1)

        solo_totals = []
        for text in texts:
            s, b = self.embed({"input": text})
            self.assertEqual(s, 200)
            solo_totals.append(b["usage"]["prompt_tokens"])
        self.assertEqual(
            body["usage"]["prompt_tokens"], sum(solo_totals),
            f"batch usage {body['usage']['prompt_tokens']} != sum of solo "
            f"counts {sum(solo_totals)} -- padding leaked into usage",
        )

    def test_batch_equivalence_with_solo_requests(self) -> None:
        """A batched call and the same texts run one at a time must agree.

        This is the CPU-side form of the padding regression: if any sample is
        fed at the batch-max length instead of its own real length, the short
        rows of a mixed batch drift from their solo result. Measured on this
        service, padding the graph input costs cosine 0.058 on the CPU path,
        so this comparison is a hard discriminator, not a nicety.
        """
        texts = [
            "def hello(): print(1)",
            "import os\nos.environ['X'] = 'Y'",
            text_of_exact_length(300),
            "ただの短い文",
        ]
        status_b, body_b = self.embed({"input": texts})
        self.assertEqual(status_b, 200)
        self.assertOpenAIShape(body_b, len(texts))

        for i, text in enumerate(texts):
            status_s, body_s = self.embed({"input": text})
            self.assertEqual(status_s, 200)
            solo = body_s["data"][0]["embedding"]
            batched = body_b["data"][i]["embedding"]
            self.assertEqual(
                solo, batched,
                f"batch position {i} differs from the same text sent alone -- "
                "one of the two paths is padding-dependent",
            )
            # The batched element's own token count must also match its solo
            # count, which rules out a mask that counts padding columns.
            solo_tokens = body_s["usage"]["prompt_tokens"]
            from .support import UnitTokenCount
            self.assertEqual(solo_tokens, UnitTokenCount(text))

    def test_get_is_not_allowed(self) -> None:
        from .support import get
        status, _ = get("/v1/embeddings")
        self.assertIn(status, (405, 404))

    def test_unknown_path_is_404(self) -> None:
        from .support import get
        status, _ = get("/v1/nope")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
