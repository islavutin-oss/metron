# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0
"""The streaming proxy is where the timings are measured, so its edges are
where the numbers go quietly wrong. These drive the real proxy_server.js against
a mock upstream whose stream shape the test controls, and assert on what the
proxy reports back — the two bugs this covers each disguised a failure as a
plausible number:

* a role-only first SSE chunk (no content) must NOT count as a token or set
  TTFT — otherwise first-token latency reads early and there is a phantom
  inter-token interval;
* a stream that ends without [DONE] must still get a reply — otherwise the
  request hangs to timeout and a truncated response is recorded as a slow one.
"""

import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()


async def _stream(shape: str):
    if shape == "role_then_content":
        # OpenAI's first chunk is a role-only delta with no content field.
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        for _ in range(3):
            yield b'data: {"choices":[{"delta":{"content":"tok"}}]}\n\n'
        yield b"data: [DONE]\n\n"
    elif shape == "truncated":
        # Content, then the server drops the connection WITHOUT a [DONE].
        for _ in range(3):
            yield b'data: {"choices":[{"delta":{"content":"tok"}}]}\n\n'
        # generator returns -> stream closes with no [DONE]
    elif shape == "empty_stream":
        yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        yield b"data: [DONE]\n\n"
    elif shape == "packed":
        for _ in range(2):
            yield b'data: {"choices":[{"delta":{"content":"a b c d "}}]}\n\n'
        yield b'data: {"choices":[],"usage":{"completion_tokens":8}}\n\n'
        yield b"data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat(request: Request):
    shape = (await request.json()).get("model", "role_then_content")
    if shape == "server_error":
        return JSONResponse({"error": "induced"}, status_code=500)
    return StreamingResponse(_stream(shape), media_type="text/event-stream")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def proxy():
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    up_port, px_port = _free_port(), _free_port()
    threading.Thread(
        target=lambda: uvicorn.run(
            app, host="127.0.0.1", port=up_port, log_level="warning"
        ),
        daemon=True,
    ).start()

    env = {
        "PROXY_PORT": str(px_port),
        "URL": f"http://127.0.0.1:{up_port}/v1/chat/completions",
        "API_ROUTE": "/v1/chat/completions",
        "STREAM": "true",
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    proc = subprocess.Popen(
        [
            "node",
            str(Path(__file__).resolve().parents[1] / "metron" / "proxy_server.js"),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # wait for both to listen
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", px_port), 0.2).close()
            socket.create_connection(("127.0.0.1", up_port), 0.2).close()
            break
        except OSError:
            time.sleep(0.2)
    yield f"http://127.0.0.1:{px_port}/v1/chat/completions"
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _ask(url, shape):
    return requests.post(url, json={"model": shape, "stream": True}, timeout=15).json()


def _post(url, shape):
    return requests.post(url, json={"model": shape, "stream": True}, timeout=15)


def test_role_only_chunk_is_not_counted_as_a_token(proxy):
    """3 content tokens -> exactly 2 inter-token intervals. If the empty role
    chunk were counted, there would be 3 and TTFT would be set on it."""
    out = _ask(proxy, "role_then_content")
    assert out["ttft"] is not None
    assert len(out["itl"]) == 2, out  # 3 content tokens - 1
    # TTFT precedes the first inter-token gap, never exceeds e2e.
    assert 0 < out["ttft"] <= out["e2e"]


def test_stream_without_done_still_replies(proxy):
    """A truncated stream (no [DONE]) must not hang; the proxy answers with
    what it measured so the request completes instead of timing out."""
    out = _ask(proxy, "truncated")  # would raise on timeout if it hung
    assert "e2e" in out and out["e2e"] > 0
    assert len(out["itl"]) == 2  # still measured the 3 tokens seen


def test_upstream_error_status_is_propagated(proxy):
    """An engine that 500s must reach k6 as a non-2xx. The proxy used to answer
    200 whatever the upstream said, so failures were counted as successes and
    their tokens invented."""
    r = _post(proxy, "server_error")
    assert r.status_code == 500
    assert r.json()["upstream_status"] == 500


def test_successful_stream_reports_its_content_chunks(proxy):
    out = _ask(proxy, "role_then_content")
    assert out["content_chunks"] == 3
    assert out["complete"] is True
    assert out["upstream_status"] == 200


def test_content_free_success_reports_no_chunks(proxy):
    out = _ask(proxy, "empty_stream")
    assert out["content_chunks"] == 0
    assert out["ttft"] is None
    assert "completion_tokens" not in out


def test_truncated_stream_is_marked_incomplete(proxy):
    out = _ask(proxy, "truncated")
    assert out["complete"] is False
    assert out["content_chunks"] == 3


def test_packed_chunks_are_counted_separately_from_tokens(proxy):
    """Four tokens per SSE chunk: two chunks, eight tokens. Reporting chunks as
    tokens would understate throughput by the packing factor."""
    out = _ask(proxy, "packed")
    assert out["content_chunks"] == 2
    assert out["completion_tokens"] == 8
