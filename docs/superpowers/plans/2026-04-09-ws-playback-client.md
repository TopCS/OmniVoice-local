# WebSocket Playback Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone remote Python playback client for `/ws/tts`, extend WebSocket regression tests for missing sample, unsupported format, and idle timeout, and document a real Docker plus `websocat` smoke test.

**Architecture:** Keep the server-side WebSocket protocol unchanged and add a thin standalone client under `examples/` with lazy imports for `websockets` and `pyaudio`. Add focused tests for the client helper logic and extend `tests/test_ws_handler.py` for protocol error coverage, while treating Docker plus `websocat` as a documented manual smoke path rather than a normal automated test.

**Tech Stack:** Python, FastAPI `TestClient`, pytest, asyncio, `websockets`, `pyaudio`, Docker Compose, `websocat`

---

## File Structure

- Create: `examples/ws_playback_client.py`
  Standalone remote client that validates `pcm`, sends a single `synthesize` request, consumes interleaved JSON and binary frames, and streams audio to PyAudio.

- Create: `tests/test_ws_playback_client.py`
  Focused tests for the new example client's pure and fakeable behaviors: request validation, metadata-plus-binary ordering, and server error handling.

- Modify: `tests/test_ws_handler.py`
  Extend the existing in-process WebSocket tests to lock in the current missing-sample behavior and cover unsupported format and idle timeout.

- Modify: `README.md`
  Replace the current print-only Python snippet with the real playback client workflow, add PyAudio install notes, and document the real Docker plus `websocat` smoke test.

### Task 1: Add Tested Remote Playback Client

**Files:**
- Create: `examples/ws_playback_client.py`
- Create: `tests/test_ws_playback_client.py`
- Test: `tests/test_ws_playback_client.py`

- [ ] **Step 1: Write the failing client tests**

```python
from pathlib import Path
import asyncio
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import ws_playback_client


class FakeWebSocket:
    def __init__(self, messages):
        self._messages = list(messages)

    async def recv(self):
        if not self._messages:
            raise RuntimeError("No more messages")
        return self._messages.pop(0)


class FakeStream:
    def __init__(self):
        self.writes = []

    def write(self, payload):
        self.writes.append(payload)


def test_build_synthesize_request_rejects_non_pcm():
    try:
        ws_playback_client.build_synthesize_request(
            text="ciao",
            sample="agent-voice",
            num_step=16,
            output_format="wav",
        )
    except ValueError as exc:
        assert "Only output_format='pcm' is supported" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-pcm output format")


def test_consume_stream_writes_binary_after_audio_chunk_metadata():
    websocket = FakeWebSocket(
        [
            json.dumps({"type": "audio_chunk", "chunk_index": 0, "sentence": "Ciao!"}),
            b"\x01\x02",
            json.dumps({"type": "done", "total_chunks": 1, "total_duration_ms": 10, "total_generation_time_ms": 1}),
        ]
    )
    stream = FakeStream()

    chunk_count = asyncio.run(ws_playback_client.consume_stream(websocket, stream))

    assert chunk_count == 1
    assert stream.writes == [b"\x01\x02"]


def test_consume_stream_raises_on_server_error_event():
    websocket = FakeWebSocket(
        [json.dumps({"type": "error", "code": "SESSION_ERROR", "message": "Sample 'missing' not found."})]
    )
    stream = FakeStream()

    try:
        asyncio.run(ws_playback_client.consume_stream(websocket, stream))
    except RuntimeError as exc:
        assert "SESSION_ERROR" in str(exc)
        assert "Sample 'missing' not found." in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for server error event")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_ws_playback_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ws_playback_client'`

- [ ] **Step 3: Write the minimal client implementation**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from typing import Any


def build_synthesize_request(
    *,
    text: str,
    sample: str | None,
    num_step: int,
    output_format: str,
) -> dict[str, Any]:
    normalized_format = output_format.lower()
    if normalized_format != "pcm":
        raise ValueError("Only output_format='pcm' is supported for live playback.")

    payload: dict[str, Any] = {
        "type": "synthesize",
        "text": text,
        "output_format": "pcm",
        "num_step": num_step,
    }
    if sample:
        payload["sample"] = sample
    return payload


async def consume_stream(websocket: Any, stream: Any) -> int:
    chunk_count = 0
    awaiting_binary = False

    while True:
        message = await websocket.recv()
        if isinstance(message, bytes):
            if not awaiting_binary:
                raise RuntimeError("Received binary payload before audio_chunk metadata.")
            stream.write(message)
            awaiting_binary = False
            continue

        event = json.loads(message)
        print(json.dumps(event, ensure_ascii=False))
        event_type = event.get("type")

        if event_type == "audio_chunk":
            if awaiting_binary:
                raise RuntimeError("Received a second audio_chunk before the previous binary payload.")
            chunk_count += 1
            awaiting_binary = True
            continue

        if event_type == "done":
            if awaiting_binary:
                raise RuntimeError("Received done before the binary payload for the last chunk.")
            return chunk_count

        if event_type == "error":
            raise RuntimeError(f"{event.get('code')}: {event.get('message')}")

        raise RuntimeError(f"Unexpected event type: {event_type}")


async def run_client(args: argparse.Namespace) -> int:
    websockets = importlib.import_module("websockets")
    pyaudio = importlib.import_module("pyaudio")

    payload = build_synthesize_request(
        text=args.text,
        sample=args.sample,
        num_step=args.num_step,
        output_format=args.output_format,
    )

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True,
    )

    try:
        async with websockets.connect(args.url) as websocket:
            await websocket.send(json.dumps(payload))
            return await consume_stream(websocket, stream)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play OmniVoice WS PCM output in real time.")
    parser.add_argument("--url", default="ws://localhost:8000/ws/tts")
    parser.add_argument("--text", required=True)
    parser.add_argument("--sample")
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--output-format", default="pcm")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunk_count = asyncio.run(run_client(args))
    print(f"played_chunks={chunk_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the client tests to verify they pass**

Run: `pytest tests/test_ws_playback_client.py -v`
Expected: PASS with `3 passed`

- [ ] **Step 5: Commit**

```bash
git add examples/ws_playback_client.py tests/test_ws_playback_client.py
git commit -m "feat: add remote websocket playback client example"
```

### Task 2: Extend WebSocket Error Regression Tests

**Files:**
- Modify: `tests/test_ws_handler.py:41-147`
- Test: `tests/test_ws_handler.py`

- [ ] **Step 1: Write the failing WebSocket error tests**

```python
def test_ws_set_voice_missing_sample_returns_session_error():
    client, _ = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50),
        voice_samples={},
    )

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "set_voice", "sample": "missing"})
        error = ws.receive_json()

    assert error == {
        "type": "error",
        "message": "Sample 'missing' not found.",
        "code": "SESSION_ERROR",
    }


def test_ws_synthesize_rejects_unsupported_output_format():
    client, _ = _make_client(WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50))

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json(
            {
                "type": "synthesize",
                "text": "Ciao mondo.",
                "sample": "agent-voice",
                "output_format": "mp3",
            }
        )
        error = ws.receive_json()

    assert error == {
        "type": "error",
        "message": "Unsupported output_format 'mp3'.",
        "code": "UNSUPPORTED_FORMAT",
    }


def test_ws_idle_timeout_emits_error_before_closing():
    client, _ = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50, inactivity_timeout_s=0)
    )

    with client.websocket_connect("/ws/tts") as ws:
        error = ws.receive_json()

    assert error == {
        "type": "error",
        "message": "Connection idle timeout.",
        "code": "IDLE_TIMEOUT",
    }
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_ws_handler.py::test_ws_set_voice_missing_sample_returns_session_error tests/test_ws_handler.py::test_ws_synthesize_rejects_unsupported_output_format tests/test_ws_handler.py::test_ws_idle_timeout_emits_error_before_closing -v`
Expected: FAIL because `_make_client()` does not yet accept `voice_samples`, and at least one of the new assertions is not yet wired into the helper-based setup.

- [ ] **Step 3: Make the minimal test harness change and keep server behavior intact**

```python
def _make_client(
    config: WSConfig | None = None,
    voice_samples: dict[str, dict[str, str]] | None = None,
):
    ws_handler.encode_audio_chunk = lambda audio, output_format, sample_rate=24000: b"x" * 4800
    model = FakeModel()
    app = FastAPI()
    app.include_router(
        create_ws_router(
            get_model=lambda: model,
            get_voice_samples=lambda: voice_samples
            or {"agent-voice": {"audio_path": "/tmp/agent.wav", "ref_text": "hello"}},
            model_lock=__import__("asyncio").Lock(),
            config=config or WSConfig(min_sentence_chars=1, buffer_timeout_ms=50),
            active_counter=lambda delta: None,
        )
    )
    return TestClient(app), model
```

Do not change `app/ws_handler.py` for this task unless one of the new tests reveals a real mismatch between the documented current behavior and the implementation.

- [ ] **Step 4: Run the full WebSocket handler test file**

Run: `pytest tests/test_ws_handler.py -v`
Expected: PASS with all existing streaming tests plus the 3 new error-path tests green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ws_handler.py
git commit -m "test: cover websocket error paths"
```

### Task 3: Update README For Remote Playback And Smoke Testing

**Files:**
- Modify: `README.md:99-151`
- Modify: `README.md:118-151`
- Modify: `README.md` near the WebSocket section to add remote playback and smoke-test notes

- [ ] **Step 1: Replace the print-only Python snippet with the real playback client workflow**

````markdown
### Remote Python playback client

The example client lives at `examples/ws_playback_client.py` and is meant to run on a remote machine with local audio output.

Install dependencies on the remote machine:

```bash
pip install websockets pyaudio
```

If PyAudio is missing system audio libraries:

- Debian/Ubuntu system package option: `sudo apt install python3-pyaudio`
- macOS: `brew install portaudio && pip install pyaudio`

Run the client:

```bash
python examples/ws_playback_client.py \
  --url ws://SERVER_HOST:8000/ws/tts \
  --text "Ciao! Questo stream arriva dal server remoto." \
  --sample agent-voice \
  --output-format pcm
```

The client only supports `pcm` for live playback. It prints JSON events and plays each binary PCM chunk as it arrives.
````

- [ ] **Step 2: Add a manual real-stack smoke test using Docker Compose and `websocat`**

````markdown
### End-to-end smoke test with Docker Compose and websocat

This is a real environment smoke test. It requires:

- Docker Compose
- model download and startup time
- GPU access when your deployment expects it
- at least one valid sample in `./samples`

Start the stack:

```bash
docker compose up -d --build
curl http://localhost:8000/health
```

Open a WebSocket session:

```bash
websocat ws://localhost:8000/ws/tts
```

Send:

```json
{"type":"synthesize","text":"Smoke test end to end.","sample":"agent-voice","output_format":"pcm"}
```

Expected result:

- one or more `audio_chunk` JSON events
- binary payload output from `websocat`
- a final `done` event

This smoke test is intentionally manual and complements, rather than replaces, the fast `pytest` coverage.
````

- [ ] **Step 3: Verify the example script still parses and the targeted tests still pass**

Run: `python -m py_compile examples/ws_playback_client.py && pytest tests/test_ws_playback_client.py tests/test_ws_handler.py -v`
Expected: no syntax errors, and both test files PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md examples/ws_playback_client.py tests/test_ws_playback_client.py tests/test_ws_handler.py
git commit -m "docs: add websocket playback usage and smoke test"
```

### Task 4: Run Final Verification, Including The Real Smoke Path

**Files:**
- Verify: `examples/ws_playback_client.py`
- Verify: `tests/test_ws_playback_client.py`
- Verify: `tests/test_ws_handler.py`
- Verify: `README.md`

- [ ] **Step 1: Run the full automated verification suite for the code changes**

Run: `pytest tests/test_ws_playback_client.py tests/test_ws_handler.py tests/test_audio_encoder.py tests/test_sentence_splitter.py -v`
Expected: PASS with zero failures.

- [ ] **Step 2: Run the real Docker smoke test against the local stack**

Run: `docker compose up -d --build`
Expected: the container starts and `/health` eventually responds successfully.

Then run: `curl http://localhost:8000/health`
Expected: HTTP 200 with model readiness information.

Then run: `websocat ws://localhost:8000/ws/tts`
Expected: an interactive session opens.

Send this JSON through the open `websocat` session:

```json
{"type":"synthesize","text":"Smoke test end to end.","sample":"agent-voice","output_format":"pcm"}
```

Expected: one or more `audio_chunk` metadata messages, binary audio output, and a final `done` event.

- [ ] **Step 3: Document any environment-specific blockers instead of hiding them**

If Docker, GPU access, model loading, or sample availability prevents the smoke test from completing, record the exact blocking command and output in the work summary. Do not claim the smoke test passed without direct evidence.

- [ ] **Step 4: Commit the final verified state**

```bash
git add README.md examples/ws_playback_client.py tests/test_ws_playback_client.py tests/test_ws_handler.py
git commit -m "feat: complete websocket playback client support"
```
