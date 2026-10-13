# Vendored microsoft/unixcoder-base tokenizer (RoBERTa BPE)

These files are the *complete* tokenizer the service needs, copied verbatim
from the Hugging Face cache snapshot for the pinned model commit
`5604afdc964f6c53782a6813140ade5216b99006` (the repo's pinned `main` revision,
as reported by the hub API).

Why vendored at all
-------------------
The service used to do `AutoTokenizer.from_pretrained("microsoft/unixcoder-base")`
on every start. That is a network dependency at process start: if the Hub is
unreachable, rate-limiting, or the cached snapshot has been evicted, the service
crashes *before* it can bind port 8000 -- a total outage for the memory store
that depends on it. Upstream also does **not** publish `tokenizer.json`, so
transformers must fall back to the slow `vocab.json` + `merges.txt` pair; the
`.no_exist/*/tokenizer.json` 0-byte entries in the hub cache are that failed
lookup, not a real file.

With these files in the repository the tokenizer is a local artefact: no
network, no cache eviction, no hub outage can stop the service from starting.

Files
-----
| File                    | Size     | What it is                                            |
|-------------------------|----------|-------------------------------------------------------|
| `vocab.json`            | 937652   | 51416 token -> id entries (vocab_size 51416)          |
| `merges.txt`            | 444183   | 49625 BPE merges, header `#version: 0.2`              |
| `tokenizer_config.json` | 1113     | `tokenizer_class: RobertaTokenizer`, special tokens   |
| `special_tokens_map.json` | 772    | the 7 special tokens                                  |

Byte-for-byte checksums of the originals (blob path in
`~/.cache/huggingface/hub/models--microsoft--unixcoder-base/blobs/`):

    vocab.json              ee56f17aaab1a7f01e81a5f167a08fa9dbd971f7
    merges.txt              0b35bb83a0228e7d26c3cdf0600aa5f868da83fc

`tokenizer_file: null` and `special_tokens_map_file: null` in
`tokenizer_config.json` are intentional: they tell transformers to use the
slow `RobertaTokenizer` (vocab.json + merges.txt) rather than a fast
`tokenizer.json` that does not exist upstream for this model.
