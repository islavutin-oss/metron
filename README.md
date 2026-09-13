# metron

*μέτρον* — Greek, "a measure": the standard something is measured against,
not the measurement itself.

Load tester for OpenAI-compatible inference endpoints (vLLM, SGLang, MAX, or a
hosted service). Parses the streaming response for per-token timings and reports
TTFT, inter-token latency, TPOT and end-to-end latency as distributions.

## Architecture

Two processes. k6 controls request timing; the Node proxy measures token
timing.

```
   ┌─────────────────────────────────────────────────────────────────┐
   │  metron  —  spawns both, collects both, prints one table        │
   └───────┬─────────────────────────────────────────┬───────────────┘
           │ spawns                                  │ spawns
           ▼                                         ▼
 ┌───────────────────────┐                 ┌───────────────────────┐
 │  k6   THE CLOCK       │   plain POST    │  proxy  THE STOPWATCH │
 │                       │ ──────────────► │                       │
 │ constant-arrival-rate │ ◄────────────── │  node:http, no deps   │
 │ fires N req/s whether │   JSON: the     │  streams SSE from the │
 │ or not the last one   │   timings       │  server, timestamps   │
 │ came back — no wait   │                 │  every chunk          │
 └───────────────────────┘                 └───────────┬───────────┘
                                                       │ SSE, stream:true
                                                       ▼
                                            ┌───────────────────────┐
                                            │  the inference server │
                                            │  (vLLM, SGLang, MAX…) │
                                            └───────────────────────┘
```

k6's `http.post()` returns after the response completes and yields the body as a
single string; there is no per-chunk callback. It can therefore time request
submission and completion, but nothing in between, which is where TTFT and ITL
are defined. The proxy reads the SSE stream chunk by chunk and timestamps each
chunk.

```
  k6 ──── POST ────►  proxy ──── POST (stream:true) ────►  server
                        │
                        │  t0 = performance.now()
                        │       ↑ taken AFTER k6's request body has fully
                        │         arrived — so the k6→proxy hop is outside
                        │         TTFT, ITL and TPOT below
                        ▼
                     server queues, prefills, starts decoding
                        │
     ◄── SSE chunk, first one carrying real content ───────┤
                        │  t1     TTFT = t1 − t0
     ◄── SSE chunk ─────┤
                        │  t2     ITL₀ = t2 − t1
     ◄── SSE chunk ─────┤
                        │  t3     ITL₁ = t3 − t2
                       ...
     ◄── data: [DONE] ──┤
                        │  tN     E2E  = tN − t0   (see note)
                        ▼
  k6 ◄── one JSON reply: {ttft, itl[], e2e, completion_tokens}
```

k6 receives one HTTP response carrying the proxy's computed values and records
them as trends. Prompt and length selection are seeded: identical configuration
produces an identical request stream.

TTFT, ITL and TPOT are taken from the proxy's clock and are the engine's
numbers. The reported E2E is k6's own request duration — deliberately
client-side, so it carries what a caller waits for rather than what the server
spent. The two therefore do not reconcile exactly: E2E minus TTFT is a little
more than TPOT times the token count, and the difference is the local hop.

## Install

```bash
nvm install 20                                    # node 20+
# k6: download from https://github.com/grafana/k6/releases, put it on PATH

git clone https://github.com/islavutin-oss/metron && cd metron
python3 -m venv venv && source venv/bin/activate
pip install -e .                                  # ".[dev]" for tests
```

Runs from a clone; the k6 script and Node proxy are packaged beside the Python
sources. Both binaries are checked at startup rather than mid-run.

## Run

```bash
metron --url=http://localhost:8000 --route=/v1/chat/completions \
       --vus=3 --duration=30s
```

The model name is read from `/v1/models`. Add `--token=$TOKEN` for endpoints
requiring authentication.

| flag | default | meaning |
|---|---|---|
| `--url` | — | base URL of the server |
| `--route` | — | `/v1/chat/completions` or `/v1/completions` |
| `--vus` | — | virtual users; each waits for its reply before sending again |
| `--rps` | — | arrival rate; needs `--executor=constant-arrival-rate` |
| `--executor` | `constant-vus` | `constant-vus`, `constant-arrival-rate`, `ramping-arrival-rate` |
| `--max-vus` | 500 | VU ceiling an arrival rate may use |
| `--duration` | — | run length, e.g. `30s` |
| `--iterations` | — | stop after N requests instead of `--duration` |
| `--token` | — | bearer token |
| `--run-id` | generated | identifier recorded with the artifacts |
| `--output-dir` | `./artifacts` | where results are written |
| `--proxy-server-port` | first free from 3000 | streaming proxy port |
| `--extra-args` | — | metadata into `config.json`, e.g. `hardware=H200,server_solution=sglang` |

`--vus` and `--rps` take `<start>:<step>:<end>` to sweep in one command:
`--vus=1:2:10` runs 1, 2, 4, 6, 8, 10.

An arrival rate requires enough concurrent VUs to sustain it. Beyond
`--max-vus`, k6 drops iterations rather than exceed the ceiling and reports the
count; raise `--max-vus` or lower `--rps`.

## Prompts

Default corpus is [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (MIT).
Whole questions are concatenated in a seeded order so prompts do not share a
prefix, preventing prefix caching from serving one request from another.

Corpus content affects results. A synthetic 13-word vocabulary measured on an
RTX 4000 reported TTFT constant at 28 ms across 1–16 concurrent users, because
prefill was being skipped.

| flag | default | meaning |
|---|---|---|
| `--corpus` | `openai/gsm8k` | HuggingFace dataset id, or a local `.jsonl` |
| `--corpus-field` | `question` | field to read from each record |
| `--corpus-config` | `main` | HuggingFace config; ignored for a local file |
| `--corpus-split` | `train` | HuggingFace split |
| `--seed` | 42 | same seed, same request stream |
| `--tokenizer` | — | for accurate truncation; otherwise 1 token ≈ 4 chars |

Lengths come from `--input-tokens-distribution` / `--output-tokens-distribution`,
each taking `normal(mu, sigma)`, `uniform(a, b)`, `const(n)` or
`dataset(file.jsonl[, ctx])`.

| you want | use | prompt length |
|---|---|---|
| your text as raw material | `--corpus file.jsonl` | whatever the distribution asks for |
| your prompts exactly as written | `--input-tokens-distribution="dataset(file.jsonl)"` | whatever is in the file |

`.jsonl` is one object per line read from `content`; `.json` is one list read
from `prompt`. The optional second `dataset()` argument truncates to a context
length. Nothing is downloaded.

Set `temperature`, `logprobs` or `n` in `metron/configs/.env.openai`, or
override per run: `--additional-api-args "TEMPERATURE=0.1,MODEL_NAME=..."`.

## Output

`./artifacts/<run-uuid>/<rps_n|vus_n>/` per point in the sweep:

| file | contents |
|---|---|
| `summary.json` | aggregated metrics, p50 through p99.9 |
| `trace.json` | per-sample trace |
| `config.json` | workload configuration and run metadata |
| `proxy.log` | streaming proxy output |

```
ms             p50         p90         p99         max
TTFT         21.06       23.30       25.39       33.15
ITL           6.33        6.58       14.87       15.96
TPOT          6.59        6.62        6.64        6.64
E2E         436.88      439.96      443.54      449.64
```

Six conditions invalidate the reported distribution. Each is detected and
reported alongside the numbers.

| condition | signal | effect |
|---|---|---|
| schedule slip | k6 `dropped_iterations` > 0 | driver waited on the server; unsent requests are the slow ones (coordinated omission) — tail understated |
| excluded failures | non-2xx count > 0 | failures carry no latency; percentiles cover survivors only |
| insufficient samples | requests below the percentile's threshold | percentile withheld, not printed |
| content-free successes | `responses_without_content` > 0 | a 2xx that streamed nothing generated nothing; it contributes zero tokens, never the number that was requested |
| estimated token counts | `token_counts_estimated` > 0 | no usage block, so tokens were counted from content chunks; an engine packing several tokens per chunk undercounts throughput |
| truncated streams | `incomplete_streams` > 0 | the stream ended without `[DONE]` or a finish reason; a half-length answer looks faster than a complete one |

A non-2xx from the engine is reported as a non-2xx: the streaming proxy
forwards the upstream status rather than answering 200 on its behalf.

A percentile needs ~10 observations beyond it to be stable: p90 ~100 requests,
p99 ~1,000.

## Metrics

Latency in milliseconds, reported as distributions with the percentile appended:
`ITL (p50)`.

| metric | definition |
|---|---|
| TTFT | request submission → first token |
| ITL | between neighbouring SSE chunks of one request |
| ITL/token | ITL divided by the tokens in a chunk; printed when the engine packs several tokens per chunk and reports `usage` |
| TPOT | (end-to-end − TTFT) / generated tokens |
| E2E | request submission → last token |
| total generated tokens | across all requests in the run |
| output token throughput | tokens/second across all requests |

## Reproducing a published measurement

A run's provenance records the engine image digest, model revision, GPU and
driver, engine flags and sweep. Rebuild from those:

```bash
# 1. engine — base pinned by digest, not tag
docker build -f reproduce/Dockerfile.vllm \
  --build-arg BASE=<base-image@sha256:...> -t vllm-repro .

# 2. serve — weights from a mount, engine offline
docker run --rm --gpus all -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/cache/hf" \
  vllm-repro <model-id> \
  --max-model-len <recorded> --gpu-memory-utilization <recorded>

# 3. same sweep
metron --url=http://127.0.0.1:8000 --route=/v1/chat/completions \
  --executor=<recorded> --vus=<start:step:end> --duration=<recorded>s \
  --input-tokens-distribution="const(<recorded>)" \
  --output-tokens-distribution="const(<recorded>)" \
  --tokenizer=<model-id>
```

Compare the built image digest against the one in the provenance. Third-party
reproduction requires a public base image.

Measure on the machine that serves (localhost), as the published runs did. A run
measured across a network includes that path in every latency value.

## Development

```bash
pip install -e ".[dev]"
pre-commit run --all-files    # ruff + mypy; same as CI
python3 -m pytest tests/      # e2e tests need k6 and node
```

Two invariants govern review: the driver must never throttle its own sending or
suppress dropped-iteration reporting, and non-2xx responses must never
contribute to throughput or latency percentiles. `tests/test_repo_hygiene.py`
fails the build on secrets, names or non-loopback IPs in the tree.

Report vulnerabilities through a GitHub Security Advisory on this repository
rather than a public issue.

## License

Apache 2.0, Copyright (c) 2024 AXELTEC SOFTWARE LTD. See [LICENSE](LICENSE).

metron is the open-source extraction of an internal load tester first written
in 2024.

Third-party components, none vendored or redistributed here:

| component | licence | how it is used |
|---|---|---|
| [k6](https://github.com/grafana/k6) | AGPL-3.0 | invoked as a separate unmodified binary you install |
| `textSummary` from jslib.k6.io | Grafana's terms | version-pinned, fetched by k6 at run time |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | MIT | default prompts, downloaded at run time |

A container image that bundles k6 conveys an AGPL binary: keep it unmodified,
pin the version, ship its LICENSE and a source pointer. Do not patch or fork k6
without reading AGPL-3.0 section 13. None of this affects metron's own licence.
