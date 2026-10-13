"""Mean-pooling correctness: that padding rows never contribute.

This is the section most at risk of being a decorative test, so it is built
around *discriminators that can fail* rather than restatements of the code:

1. ``hidden[i, real_len:]`` must be exactly zero. The service re-pads the
   per-sample outputs back to a rectangular tensor for pooling; if that pad
   were filled with anything else (garbage, last-row copies), every short row
   of a mixed batch would be wrong. Asserted at the backend level through
   ``last_hidden_state`` -- the one place the values are observable.

2. Batch-vs-solo identity: a short text's embedding must be *bit-identical*
   whether it is sent alone or as part of a batch with a 1024-token sibling.
   A pooling bug (dividing by the padded sequence length, or masking only the
   denominator) is invisible when every row has the same real length, so the
   batch is deliberately mixed by factor ~100.

3. The measured padding effect: feeding the graph a padded row instead of the
   sample's real length. On CPU this service gives cosine 0.058 vs the
   per-sample result -- far worse than the fp16 bug -- so this is the real
   signal, and it is checked through the module under test rather than
   reimplemented.

Explicitly NOT asserted: "mean_pooling multiplies by the mask" restated, or
"pooled has 768 columns". Both pass if the function were deleted.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from ncnn_backend import HIDDEN_SIZE, NcnnEmbeddingBackend  # noqa: E402
from .support import ServiceCase, post  # noqa: E402

PARAM = os.path.join(REPO, "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.param")
WEIGHTS = os.path.join(REPO, "unixcoder-onnx", "unixcoder.pnnx.fp32.ncnn.bin")

SHORT = "def hello(): print(1)"          # 9 tokens
LONG_TEXT = " foo" * 300                  # 302 tokens


def _encode(texts):
    from .support import tokenizer
    enc = tokenizer()(texts, padding=True, truncation=True, return_tensors="np")
    return enc["input_ids"], enc["attention_mask"]


class PadRowsAreZeroTest(unittest.TestCase):
    """The hidden-state-level invariant that makes output re-padding safe."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = NcnnEmbeddingBackend(PARAM, WEIGHTS, use_vulkan=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()
        cls.backend = None

    def test_padding_rows_are_identical_zero(self) -> None:
        ids, mask = _encode([SHORT, LONG_TEXT])
        hidden = self.backend.last_hidden_state(ids, mask)
        self.assertEqual(hidden.shape[0], 2)
        self.assertEqual(hidden.shape[2], HIDDEN_SIZE)
        for i, row_mask in enumerate(mask):
            n = int(row_mask.sum())
            with self.subTest(row=i, real_tokens=n):
                self.assertTrue(np.all(hidden[i, n:] == 0.0),
                                f"row {i}: {hidden.shape[1] - n} padding rows "
                                "are not zero -- mean pooling would include "
                                "them")

    def test_pad_rows_stay_zero_for_many_lengths(self) -> None:
        # Rows of strictly increasing length in one batch: the longest sets
        # the batch max, so every row *except* it has padding rows.
        counts = [4, 5, 6, 7, 9, 11, 13, 17, 23]
        texts = [" foo" * (n - 2) for n in counts]
        ids, mask = _encode(texts)
        hidden = self.backend.last_hidden_state(ids, mask)
        real_lens = mask.sum(axis=1).astype(int)
        self.assertEqual(list(real_lens), counts)
        self.assertEqual(hidden.shape[1], max(counts))
        for i, n in enumerate(real_lens):
            with self.subTest(i=i, real_tokens=int(n)):
                if int(n) == hidden.shape[1]:
                    continue  # the longest row has no padding
                self.assertLess(int(n), hidden.shape[1])
                self.assertTrue(np.all(hidden[i, int(n):] == 0.0),
                                f"row {i} (real {int(n)}, batch max "
                                f"{hidden.shape[1]}) has non-zero padding rows")

    def test_pooling_discriminator_is_the_denominator(self) -> None:
        """The real, non-vacuous mean-pooling assertion.

        Because the pad rows are exactly zero, masking the numerator is a
        no-op -- which means a pooling test that only checks "the mask is
        applied" would pass even if pooling were wrong. What actually
        discriminates is the *denominator*: the real token count, not the
        padded sequence length.

        This test runs both denominators over the real hidden states and
        asserts they differ, then that the service's own mean_pooling uses
        the right one (checked end-to-end in the HTTP module).
        """
        ids, mask = _encode([SHORT, LONG_TEXT])
        hidden = self.backend.last_hidden_state(ids, mask)
        m = mask.astype(bool)
        seq_len = hidden.shape[1]

        pooled_correct = (hidden * m[:, :, None]).sum(1) / m.sum(1, keepdims=True)
        pooled_naive = hidden.sum(1) / seq_len

        self.assertGreater(
            float(np.abs(pooled_correct - pooled_naive).max()), 0.0,
            "the two denominators agree -- this batch has no usable padding "
            "and the test cannot discriminate",
        )
        # And they must not be near-equal: a diff of 1e-6 would still let a
        # naive implementation pass a loose tolerance.
        self.assertGreater(
            float(np.linalg.norm(pooled_correct - pooled_naive)
                  / np.linalg.norm(pooled_correct)),
            0.01,
            "the naive and correct poolings are too close to discriminate",
        )
        # Sanity: the correct pooling really is the mean of the real rows.
        # Float32 accumulation order makes exact equality impossible here.
        n = int(m[0].sum())
        manual = hidden[0, :n].sum(0) / n
        self.assertTrue(np.allclose(pooled_correct[0], manual, rtol=1e-5, atol=1e-5),
                        "pooled row is not the mean of its real token rows")
        self.assertEqual(seq_len, int(mask[1].sum()),
                         "the batch is not mixed-length as intended")


class PoolingDiscriminatorTest(unittest.TestCase):
    """Pooling maths, checked against the observable outputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = NcnnEmbeddingBackend(PARAM, WEIGHTS, use_vulkan=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.close()
        cls.backend = None

    def test_short_row_survives_a_long_sibling_unchanged(self) -> None:
        """Mixed batch: a 9-token row next to a 302-token row.

        Both implementations are run through the real backend: per-sample
        length (the service's) vs the whole batch padded to the max. The
        service's row must be *bit-identical* to its solo result.
        """
        ids_s, mask_s = _encode([SHORT])
        ids_m, mask_m = _encode([SHORT, LONG_TEXT])
        short_solo = self.backend.last_hidden_state(ids_s, mask_s)
        mixed = self.backend.last_hidden_state(ids_m, mask_m)
        n = int(mask_s.sum(axis=1)[0])
        self.assertTrue(np.array_equal(short_solo[0], mixed[0, :n]),
                        "the short row's hidden states changed because it "
                        "was batched with a long row")

    def test_padded_graph_input_is_not_tolerated_on_cpu(self) -> None:
        """The documented 'CPU tolerates padding' claim, measured.

        README/ncnn_backend.py both state padding the batch graph input is
        *harmless on CPU*. Feeding the padded row here measures cosine 0.058
        against the per-sample result on this build -- so the claim does not
        hold for this CPU build, and per-sample feeding is load-bearing on
        both devices. Pinned so a regression to padded feeding cannot pass
        as "fine on CPU".

        If this ever flips to 'padded is fine on CPU', the assertion below is
        the thing to change, and this test becomes documentation of that.
        """
        ids, mask = _encode([SHORT, LONG_TEXT])
        n_real = int(mask[0].sum())

        service_run = self.backend.last_hidden_state(ids, mask)[0, :n_real]

        # Hand-build the padded forward: all 302 columns, mask marks the pads.
        ex = self.backend.net.create_extractor()
        import ncnn

        ids_row = ids[0].astype(np.int32)
        keep = (ids_row != 1).astype(np.float32)
        pos_row = (np.cumsum(keep) * keep + 1).astype(np.int32)
        tt = np.zeros_like(ids_row)
        mask_2d = np.where(keep[:, None] == 1.0, 0.0, -1.0e9).astype(np.float32)
        for name, arr, dtype in (
            (self.backend.input_names[0], ids_row, np.int32),
            (self.backend.input_names[1], mask_2d, np.float32),
            (self.backend.input_names[2], pos_row, np.int32),
            (self.backend.input_names[3], tt, np.int32),
        ):
            ex.input(name, ncnn.Mat(np.ascontiguousarray(arr, dtype=dtype)))
        out = ncnn.Mat()
        self.assertEqual(ex.extract(self.backend.output_name, out), 0,
                         "padded forward failed to run")
        padded_run = np.asarray(out.numpy()).reshape(-1, HIDDEN_SIZE)[:n_real]

        cos = _row_cos(padded_run, service_run)
        self.assertLess(
            cos, 0.9,
            f"padded graph input reproduces the per-sample result (cos "
            f"{cos:.3f}) on CPU -- if that is true on this build, this test "
            "should say so, not stay silent",
        )

    def test_batch_row_order_does_not_change_any_row(self) -> None:
        ids_a, mask_a = _encode([SHORT, LONG_TEXT])
        ids_b, mask_b = _encode([LONG_TEXT, SHORT])
        a = self.backend.last_hidden_state(ids_a, mask_a)
        b = self.backend.last_hidden_state(ids_b, mask_b)
        n_short = int(mask_a[0].sum())
        n_long = int(mask_a[1].sum())
        self.assertTrue(np.array_equal(a[0, :n_short], b[1, :n_short]))
        self.assertTrue(np.array_equal(a[1, :n_long], b[0, :n_long]))


def _row_cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class EndToEndPoolingTest(ServiceCase):
    """The same properties through the HTTP contract, where the bug would be
    consumer-visible."""

    def test_short_element_in_mixed_batch_is_its_solo_self(self) -> None:
        """The strongest CPU-side padding discriminator available over HTTP.

        Sending the short text alone and in a batch with a 1024-token sibling
        must return the *same embedding*, bit for bit. Any implementation
        that pooled over padded positions, or that divided by the padded
        sequence length, breaks here -- and equally, one that padded the
        graph input breaks here on this build.
        """
        sibling = " foo" * 1022          # 1024 tokens: the maximum
        status_s, solo = self.embed({"input": SHORT})
        status_b, batch = self.embed({"input": [SHORT, sibling]})
        self.assertEqual((status_s, status_b), (200, 200))
        self.assertOpenAIShape(batch, 2)
        self.assertEqual(
            solo["data"][0]["embedding"], batch["data"][0]["embedding"],
            "the 9-token element changed when batched with a 1024-token "
            "sibling -- padding is leaking into the first element",
        )
        # And the long sibling must itself be unchanged by the batch.
        status_l, long_solo = self.embed({"input": sibling})
        self.assertEqual(status_l, 200)
        self.assertEqual(
            long_solo["data"][0]["embedding"], batch["data"][1]["embedding"],
            "the 1024-token element changed when batched with a short one",
        )

    def test_very_different_lengths_in_one_batch(self) -> None:
        lengths = [3, 9, 64, 700, 1024]
        texts = [" foo" * (n - 2) for n in lengths]
        status_b, batch = self.embed({"input": texts})
        self.assertEqual(status_b, 200)
        self.assertOpenAIShape(batch, len(texts))
        for i, text in enumerate(texts):
            with self.subTest(i=i, tokens=lengths[i]):
                status_s, solo = self.embed({"input": text})
                self.assertEqual(status_s, 200)
                self.assertEqual(
                    solo["data"][0]["embedding"], batch["data"][i]["embedding"],
                    f"element {i} ({lengths[i]} tokens) differs from its solo "
                    "result -- one of the two paths handles padding wrongly",
                )

    def test_usage_does_not_count_padding_in_a_mixed_batch(self) -> None:
        lengths = [9, 302, 1024]
        texts = [" foo" * (n - 2) for n in lengths]
        status, body = self.embed({"input": texts})
        self.assertEqual(status, 200)
        self.assertEqual(
            body["usage"]["prompt_tokens"], sum(lengths),
            f"usage {body['usage']['prompt_tokens']} != {sum(lengths)} real "
            "tokens -- the padded columns were counted",
        )

    def test_order_of_a_mixed_batch_is_stable(self) -> None:
        lengths = [1024, 9, 302, 3]
        forward = [" foo" * (n - 2) for n in lengths]
        reverse = list(reversed(forward))
        _, f = self.embed({"input": forward})
        _, r = self.embed({"input": reverse})
        for i in range(len(forward)):
            self.assertEqual(
                f["data"][i]["embedding"],
                r["data"][len(forward) - 1 - i]["embedding"],
                "batch position changed the embedding -- padding is "
                "position-dependent",
            )


if __name__ == "__main__":
    unittest.main()
