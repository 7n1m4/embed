"""The MAX_INPUT_TOKENS guard and the surrounding input-validation errors.

What this pins:

* exactly at the limit (1024) -> 200
* one token over (1025) -> 413 naming the offending count
* ONE over-limit element rejects the WHOLE request (no partial results)
* the body names the offending element index and its token count
* the rejection happens BEFORE the backend is loaded/extractored -- proved by
  poisoning the backend cache, so an implementation that reaches the backend
  fails this suite instead of passing by accident.

Also the neighbouring error paths (400 on empty input, 405 on GET) because a
guard that is right at the boundary and wrong next door is still a bug.
"""
from __future__ import annotations

import unittest

from .support import ServiceCase, post, text_of_exact_length, UnitTokenCount

LIMIT = 1024


class TokenLimitGuardTest(ServiceCase):
    def test_exactly_at_limit_succeeds(self) -> None:
        text = text_of_exact_length(LIMIT)
        self.assertEqual(UnitTokenCount(text), LIMIT, "test fixture is wrong")
        status, body = self.embed({"input": text})
        self.assertEqual(status, 200)
        self.assertOpenAIShape(body, 1)
        self.assertEqual(body["usage"]["prompt_tokens"], LIMIT)

    def test_one_token_over_limit_is_413(self) -> None:
        text = text_of_exact_length(LIMIT + 1)
        self.assertEqual(UnitTokenCount(text), LIMIT + 1, "test fixture is wrong")
        status, body = self.embed({"input": text})
        self.assertEqual(status, 413)
        self.assertIn("detail", body)
        self.assertIn(str(LIMIT + 1), body["detail"])
        self.assertIn(str(LIMIT), body["detail"])
        self.assertNotIn("data", body, "a 413 must not return embeddings")
        self.assertNotEqual(body.get("object"), "list")

    def test_well_over_limit_is_413(self) -> None:
        text = text_of_exact_length(2163)
        status, body = self.embed({"input": text})
        self.assertEqual(status, 413, "the documented 2163-token regime must 413")
        self.assertIn("2163", body["detail"])

    def test_batch_with_one_over_limit_element_rejects_whole_request(self) -> None:
        texts = ["short and fine", "also fine", text_of_exact_length(LIMIT + 1)]
        status, body = self.embed({"input": texts})
        self.assertEqual(status, 413)
        self.assertIn("2", body["detail"], "the body must name the offending index")
        self.assertIn(str(LIMIT + 1), body["detail"])
        self.assertNotIn("data", body,
                         "a rejected batch must not return partial results")

    def test_limit_is_per_element_not_per_batch(self) -> None:
        """A batch whose TOTAL exceeds the limit but no element does must pass.

        This is what makes the guard a per-element bound rather than a budget.
        """
        size = 8
        texts = [text_of_exact_length(200) for _ in range(size)]
        status, body = self.embed({"input": texts})
        self.assertEqual(status, 200)
        self.assertOpenAIShape(body, size)
        self.assertEqual(body["usage"]["prompt_tokens"], 200 * size)

    def test_first_over_limit_element_is_reported(self) -> None:
        texts = [text_of_exact_length(LIMIT + 1), "fine"]
        status, body = self.embed({"input": texts})
        self.assertEqual(status, 413)
        self.assertIn("element 0", body["detail"])

    def test_largest_over_limit_element_is_reported(self) -> None:
        """With several over-limit elements, the biggest count is named."""
        texts = [
            "fine",
            text_of_exact_length(LIMIT + 5),      # 1029
            text_of_exact_length(LIMIT + 40),     # 1064
        ]
        status, body = self.embed({"input": texts})
        self.assertEqual(status, 413)
        detail = body["detail"]
        self.assertIn("element 2", detail)
        self.assertIn("1064", detail,
                      "the largest offending count must be named, not the "
                      "first or the smallest")
        # And it must not name the smaller one as the offender.
        self.assertNotIn("1029", detail)

    def test_rejection_never_reaches_the_backend(self) -> None:
        """Prove ordering: guard first, backend second.

        The service loads the model lazily on first inference. This test
        drives the real endpoint with the backend cache poisoned so that
        *any* backend call returns 503 -- and then asserts the over-limit
        request still answers 413, not 503. If the guard were moved after
        ``ensure_backend()``, or if an implementation tokenized-and-then-
        inferred before checking, the poison would surface instead.
        """
        import embedding_service as svc
        from .support import REPO

        import os
        import sys

        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        svc.set_device("cpu")

        from fastapi.testclient import TestClient

        client = TestClient(svc.app)
        # Warm the real model once so the measurement is not about load time.
        warm = client.post("/v1/embeddings",
                           json={"input": "def hello(): print(1)"})
        self.assertEqual(warm.status_code, 200)

        # Poison: the module's lazily-built backend cache.
        svc._backend = None
        svc._backend_error = "poisoned: backend unavailable"
        try:
            over = client.post("/v1/embeddings",
                               json={"input": text_of_exact_length(LIMIT + 1)})
            self.assertEqual(
                over.status_code, 413,
                "an over-limit request touched the backend: got "
                f"{over.status_code} {over.json().get('detail')}",
            )
            at_limit = client.post("/v1/embeddings",
                                   json={"input": text_of_exact_length(LIMIT)})
            self.assertEqual(
                at_limit.status_code, 503,
                "the at-limit request did not reach the backend, so this "
                "test is not actually proving guard-first ordering",
            )
        finally:
            svc._backend_error = None
            svc._backend = None
            restored = client.post("/v1/embeddings",
                                   json={"input": "def hello(): print(1)"})
            self.assertEqual(restored.status_code, 200,
                             "the poisoned backend was not restored")


class InputValidationTest(ServiceCase):
    def test_empty_list_is_400(self) -> None:
        status, body = self.embed({"input": []})
        self.assertEqual(status, 400)
        self.assertIn("detail", body)

    def test_empty_string_is_400(self) -> None:
        status, _ = self.embed({"input": ""})
        self.assertEqual(status, 400)

    def test_list_containing_empty_string_is_400(self) -> None:
        status, body = self.embed({"input": ["fine", ""]})
        self.assertEqual(status, 400,
                         "a batch with an empty element must fail whole, not "
                         "silently drop it")

    def test_list_containing_non_string_is_400_or_422(self) -> None:
        status, _ = self.embed({"input": ["fine", 5]})
        self.assertIn(status, (400, 422))

    def test_missing_input_is_rejected(self) -> None:
        status, _ = self.embed({"model": "unixcoder-base"})
        self.assertIn(status, (400, 422))

    def test_get_endpoint_is_405(self) -> None:
        from .support import get
        status, _ = get("/v1/embeddings")
        self.assertEqual(status, 405)

    def test_413_is_not_retried_as_transient_by_a_client(self) -> None:
        """Documented rationale: 413, not 503, so a client cannot retry it.

        The distinction is observable only through the status code, which is
        exactly what a memory-store client branches on.
        """
        status, body = self.embed({"input": text_of_exact_length(LIMIT + 1)})
        self.assertEqual(status, 413)
        self.assertNotIn(status, (500, 503))


if __name__ == "__main__":
    unittest.main()
