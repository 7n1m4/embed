"""Mock embedding client for anima/embed.

Exercises the OpenAI-compatible /v1/embeddings contract the way a coding
agent's memory client would: one single-string request, one batched request,
plus a shape/usage sanity check on each.

Exits 0 only if every request returns 200 with a well-formed response
(768-dim embeddings, integer usage counters).

Usage:
    python mock_client.py [--url http://localhost:8000]
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

EMBEDDING_DIM = 768

SINGLE_INPUT = "def hello(): print(1)"
BATCH_INPUT = [
    "class A:\n    def f(self):\n        return 1",
    "SELECT * FROM users WHERE id = 1",
    "print('hello world')",
]


def post_embeddings(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}")
        return json.loads(resp.read().decode("utf-8"))


def check_response(body: dict, expected_count: int, label: str) -> None:
    if body.get("object") != "list":
        raise RuntimeError(f"{label}: object={body.get('object')!r}, expected 'list'")
    data = body.get("data")
    if not isinstance(data, list) or len(data) != expected_count:
        raise RuntimeError(f"{label}: got {len(data) if data else 0} embeddings, expected {expected_count}")

    for i, item in enumerate(data):
        if item.get("object") != "embedding":
            raise RuntimeError(f"{label}: data[{i}].object={item.get('object')!r}, expected 'embedding'")
        embedding = item.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIM:
            raise RuntimeError(f"{label}: data[{i}].embedding is not {EMBEDDING_DIM} floats")
        if not all(isinstance(v, float) or isinstance(v, int) for v in embedding):
            raise RuntimeError(f"{label}: data[{i}].embedding contains a non-numeric value")
        if item.get("index") != i:
            raise RuntimeError(f"{label}: data[{i}].index={item.get('index')!r}")

    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError(f"{label}: missing usage object")
    for key in ("prompt_tokens", "total_tokens"):
        value = usage.get(key)
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(f"{label}: usage.{key}={value!r}, expected non-negative int")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock embedding client for anima/embed")
    parser.add_argument("--url", default="http://localhost:8000/v1/embeddings")
    parser.add_argument("--model", default="unixcoder-base")
    args = parser.parse_args()

    failures = 0

    for label, text in (("single", SINGLE_INPUT), ("batch", BATCH_INPUT)):
        try:
            expected = 1 if isinstance(text, str) else len(text)
            body = post_embeddings(args.url, {"input": text, "model": args.model})
            check_response(body, expected, label)
        except (urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"FAIL {label}: {exc}")
            failures += 1
            continue

        dim = len(body["data"][0]["embedding"])
        tokens = body["usage"]["prompt_tokens"]
        print(f"OK   {label}: {len(body['data'])} embedding(s), dim={dim}, prompt_tokens={tokens}")

    if failures:
        print(f"mock client failed: {failures} check(s) failed")
        return 1
    print("mock client passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
