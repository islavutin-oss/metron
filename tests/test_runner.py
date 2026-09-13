# Copyright (c) 2024 AXELTEC SOFTWARE LTD.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import threading
import time
import subprocess
from typing import AsyncGenerator

app = FastAPI()


async def stream_tokens(field: str = "delta") -> AsyncGenerator[bytes, None]:
    chunk = (
        b'data: {"choices":[{"delta":{"content":"tok"}}]}\n\n'
        if field == "delta"
        else b'data: {"choices":[{"text":"tok"}]}\n\n'
    )
    for _ in range(10):
        yield chunk
        await asyncio.sleep(0.1)

    yield b"data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(stream_tokens(), media_type="text/event-stream")

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": " ".join(["tok" for _ in range(10)]),
                }
            }
        ]
    }


@app.post("/v1/completions")
async def text_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(stream_tokens("text"), media_type="text/event-stream")

    return {"choices": [{"text": " ".join(["tok" for _ in range(10)])}]}


def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)


@pytest.fixture(scope="module")
def mock_server():
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)
    yield


@pytest.mark.parametrize(
    "route,stream",
    [
        ("/v1/chat/completions", True),
        ("/v1/chat/completions", False),
        ("/v1/completions", True),
        ("/v1/completions", False),
    ],
)
def test_metron(mock_server, route: str, stream: bool, tmp_path, monkeypatch):
    # Use monkeypatch to set the environment variable for the duration of the test
    monkeypatch.setenv("STREAM", "true" if stream else "false")

    cmd = [
        "python3",
        "-m",
        "metron",
        "--url=http://localhost:8000",
        f"--route={route}",
        "--vus=1",
        "--duration=5s",
        "--input-tokens-distribution=const(10)",
        "--output-tokens-distribution=const(10)",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    assert result.returncode == 0
