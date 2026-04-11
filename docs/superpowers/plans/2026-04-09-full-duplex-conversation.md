# Full Duplex Conversation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `/ws/conversation` path that supports microphone uplink, client-side Silero VAD driven barge-in, server-side ASR orchestration, and response-scoped assistant playback.

**Architecture:** Keep `/ws/tts` intact and introduce a separate conversation router plus a thin ASR adapter. The workstation example client captures mic audio and plays speaker audio concurrently, uses Silero VAD locally to emit `speech_start`/`speech_end`, and drops stale assistant output by `response_id` when barge-in occurs.

**Tech Stack:** FastAPI WebSocket, pytest, asyncio, faster-whisper, silero-vad, torch, pyaudio, websockets, numpy

---

## File Structure

- Create: `app/conversation_ws.py`
  New `/ws/conversation` protocol, per-session state, interruption logic, and assistant downlink events.

- Create: `app/asr.py`
  Thin ASR adapter around `faster_whisper.WhisperModel` with one method that accepts mono 16 kHz PCM bytes and returns final text.

- Create: `tests/test_conversation_ws.py`
  Fast protocol tests for `session_start`, `speech_start`, `input_audio_chunk`, `speech_end`, interruption, stale `response_id`, and terminal errors.

- Create: `examples/ws_duplex_client.py`
  Workstation duplex client with microphone capture, Silero VAD transitions, local playback stop on barge-in, and assistant audio handling.

- Create: `tests/test_ws_duplex_client.py`
  Targeted tests for VAD-driven state transitions, stale response dropping, and playback interruption.

- Modify: `app/server.py`
  Wire the new router and ASR adapter into the app lifecycle and expose environment configuration.

- Modify: `requirements.txt`
  Add server-side ASR dependency required by the new conversation path.

- Modify: `README.md`
  Document the new duplex endpoint, workstation dependencies, and manual barge-in smoke workflow.

### Task 1: Add Conversation Protocol Skeleton And Tests

**Files:**
- Create: `app/conversation_ws.py`
- Create: `tests/test_conversation_ws.py`
- Test: `tests/test_conversation_ws.py`

- [ ] **Step 1: Write the failing protocol tests**

```python
import asyncio
import json
import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
    ),
)

from conversation_ws import ConversationConfig, create_conversation_router


class FakeConversationService:
    def __init__(self):
        self.transcribe_calls = []
        self.respond_calls = []

    async def transcribe(self, audio_bytes, sample_rate):
        self.transcribe_calls.append((audio_bytes, sample_rate))
        return "ciao mondo"

    async def respond(self, transcript, sample):
        self.respond_calls.append((transcript, sample))
        return [
            ("resp-1", {"type": "assistant_text", "response_id": "resp-1", "text": "ciao"}),
            ("resp-1", {"type": "audio_chunk", "response_id": "resp-1", "chunk_index": 0, "is_last": True}, b"\x00\x01"),
            ("resp-1", {"type": "response_done", "response_id": "resp-1"}),
        ]


def _make_client(service=None):
    service = service or FakeConversationService()
    app = FastAPI()
    app.include_router(
        create_conversation_router(
            service=service,
            config=ConversationConfig(),
            active_counter=lambda delta: None,
        )
    )
    return TestClient(app), service


def test_ws_conversation_session_start_acknowledges_sample():
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "voce_1"})
        assert ws.receive_json() == {
            "type": "session_ready",
            "sample": "voce_1",
            "sample_rate": 16000,
        }


def test_ws_conversation_speech_end_triggers_transcript_and_response():
    client, service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "voce_1"})
        ws.receive_json()
        ws.send_json({"type": "speech_start"})
        ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "listening"}
        assert ws.receive_json() == {
            "type": "transcript_final",
            "text": "ciao mondo",
        }
        assert ws.receive_json() == {
            "type": "assistant_text",
            "response_id": "resp-1",
            "text": "ciao",
        }
        meta = ws.receive_json()
        audio = ws.receive_bytes()
        done = ws.receive_json()

    assert meta["type"] == "audio_chunk"
    assert meta["response_id"] == "resp-1"
    assert audio == b"\x00\x01"
    assert done == {"type": "response_done", "response_id": "resp-1"}
    assert service.transcribe_calls[0][1] == 16000


def test_ws_conversation_barge_in_emits_interrupted_and_uses_new_response_id():
    class InterruptService(FakeConversationService):
        async def respond(self, transcript, sample):
            if not self.respond_calls:
                self.respond_calls.append((transcript, sample))
                return [
                    ("resp-1", {"type": "assistant_text", "response_id": "resp-1", "text": "prima"}),
                ]
            self.respond_calls.append((transcript, sample))
            return [
                ("resp-2", {"type": "assistant_text", "response_id": "resp-2", "text": "seconda"}),
                ("resp-2", {"type": "response_done", "response_id": "resp-2"}),
            ]

    client, _service = _make_client(InterruptService())

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "voce_1"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        ws.send_bytes(b"\x00\x00" * 1600)
        ws.send_json({"type": "speech_end"})
        ws.receive_json()
        ws.receive_json()
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "interrupted", "response_id": "resp-1"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_conversation_ws.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'conversation_ws'`

- [ ] **Step 3: Write the minimal conversation router implementation**

```python
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


@dataclass
class ConversationConfig:
    enabled: bool = True
    input_sample_rate: int = 16000


@dataclass
class ConversationState:
    sample: str | None = None
    collecting: bool = False
    audio_buffer: bytearray = field(default_factory=bytearray)
    active_response_id: str | None = None


def create_conversation_router(*, service: Any, config: ConversationConfig, active_counter):
    router = APIRouter()
    response_counter = itertools.count(1)

    @router.websocket("/ws/conversation")
    async def ws_conversation(websocket: WebSocket) -> None:
        await websocket.accept()
        state = ConversationState()
        active_counter(1)
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    if not state.collecting:
                        await websocket.send_json({"type": "error", "code": "UNEXPECTED_AUDIO", "message": "Audio received before speech_start."})
                        continue
                    state.audio_buffer.extend(message["bytes"])
                    continue

                payload = message.get("text")
                event = {} if payload is None else __import__("json").loads(payload)
                event_type = event.get("type")

                if event_type == "session_start":
                    state.sample = event.get("sample")
                    await websocket.send_json({"type": "session_ready", "sample": state.sample, "sample_rate": config.input_sample_rate})
                    continue

                if event_type == "speech_start":
                    state.collecting = True
                    state.audio_buffer.clear()
                    if state.active_response_id is not None:
                        await websocket.send_json({"type": "interrupted", "response_id": state.active_response_id})
                        state.active_response_id = None
                    await websocket.send_json({"type": "listening"})
                    continue

                if event_type == "speech_end":
                    state.collecting = False
                    transcript = await service.transcribe(bytes(state.audio_buffer), config.input_sample_rate)
                    await websocket.send_json({"type": "transcript_final", "text": transcript})
                    for response_id, outbound, *maybe_audio in await service.respond(transcript, state.sample):
                        state.active_response_id = response_id
                        await websocket.send_json(outbound)
                        if maybe_audio:
                            await websocket.send_bytes(maybe_audio[0])
                    continue

                await websocket.send_json({"type": "error", "code": "INVALID_MESSAGE", "message": f"Unsupported message type '{event_type}'."})
        except WebSocketDisconnect:
            pass
        finally:
            active_counter(-1)
            await websocket.close()

    return router
```

- [ ] **Step 4: Run the conversation router tests to verify they pass**

Run: `pytest tests/test_conversation_ws.py -v`
Expected: PASS with the initial protocol tests green.

- [ ] **Step 5: Commit**

```bash
git add app/conversation_ws.py tests/test_conversation_ws.py
git commit -m "feat: add duplex conversation websocket skeleton"
```

### Task 2: Add ASR Adapter And Server Wiring

**Files:**
- Create: `app/asr.py`
- Modify: `app/server.py:52-57`
- Modify: `app/server.py:72-82`
- Modify: `app/server.py:429-500`
- Modify: `requirements.txt`
- Test: `tests/test_conversation_ws.py`

- [ ] **Step 1: Add a failing server integration test that uses the ASR adapter contract**

```python
def test_ws_conversation_speech_end_uses_asr_adapter_text():
    class AdapterBackedService:
        def __init__(self):
            self.seen = []

        async def transcribe(self, audio_bytes, sample_rate):
            self.seen.append((audio_bytes, sample_rate))
            return "trascritto"

        async def respond(self, transcript, sample):
            return [
                ("resp-1", {"type": "assistant_text", "response_id": "resp-1", "text": transcript}),
                ("resp-1", {"type": "response_done", "response_id": "resp-1"}),
            ]

    client, service = _make_client(AdapterBackedService())

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "voce_1"})
        ws.receive_json()
        ws.send_json({"type": "speech_start"})
        ws.receive_json()
        ws.send_bytes(b"\x01\x02" * 1600)
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "trascritto"}
        assert ws.receive_json() == {"type": "assistant_text", "response_id": "resp-1", "text": "trascritto"}
```

- [ ] **Step 2: Run the targeted test to verify it fails for the right reason**

Run: `pytest tests/test_conversation_ws.py::test_ws_conversation_speech_end_uses_asr_adapter_text -v`
Expected: FAIL before the adapter and server wiring exist.

- [ ] **Step 3: Add the minimal ASR adapter and wire it into the app**

```python
# app/asr.py
from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import dataclass


@dataclass
class ASRConfig:
    model_size: str = "small"
    device: str = "cuda"
    compute_type: str = "float16"


class FasterWhisperASR:
    def __init__(self, config: ASRConfig):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(config.model_size, device=config.device, compute_type=config.compute_type)

    def _pcm16_to_wav_bytes(self, audio_bytes: bytes, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        return buffer.getvalue()

    async def transcribe(self, audio_bytes: bytes, sample_rate: int) -> str:
        wav_bytes = self._pcm16_to_wav_bytes(audio_bytes, sample_rate)

        def _run() -> str:
            segments, _info = self._model.transcribe(io.BytesIO(wav_bytes), vad_filter=False, beam_size=5)
            return " ".join(segment.text.strip() for segment in segments).strip()

        return await asyncio.to_thread(_run)
```

```python
# requirements.txt
fastapi>=0.115.0,<1.0
uvicorn[standard]>=0.32.0,<1.0
python-multipart>=0.0.18,<1.0
faster-whisper>=1.1.0,<2.0
```

```python
# app/server.py additions
from asr import ASRConfig, FasterWhisperASR
from conversation_ws import ConversationConfig, create_conversation_router

WS_CONVERSATION_ENABLED = os.environ.get("OMNIVOICE_WS_CONVERSATION_ENABLED", "true").lower() in ("true", "1", "yes")
ASR_MODEL = os.environ.get("OMNIVOICE_ASR_MODEL", "small")
ASR_DEVICE = os.environ.get("OMNIVOICE_ASR_DEVICE", "cuda")
ASR_COMPUTE_TYPE = os.environ.get("OMNIVOICE_ASR_COMPUTE_TYPE", "float16")

conversation_asr = FasterWhisperASR(ASRConfig(model_size=ASR_MODEL, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE))

class ConversationService:
    def __init__(self, asr, get_model, model_lock):
        self._asr = asr
        self._get_model = get_model
        self._model_lock = model_lock

    async def transcribe(self, audio_bytes, sample_rate):
        return await self._asr.transcribe(audio_bytes, sample_rate)

    async def respond(self, transcript, sample):
        ws_message = {"text": transcript, "sample": sample, "output_format": "pcm"}
        state = WSSessionState(prompt_sample=sample)
        return []
```

For this task, it is acceptable to wire the ASR adapter and conversation router into `app.server` while leaving assistant response generation still stub-like. The full response streaming logic lands in Task 3.

- [ ] **Step 4: Run the conversation tests again**

Run: `pytest tests/test_conversation_ws.py -v`
Expected: PASS with the adapter-backed transcript flow green.

- [ ] **Step 5: Commit**

```bash
git add app/asr.py app/server.py requirements.txt tests/test_conversation_ws.py
git commit -m "feat: add asr adapter for duplex conversations"
```

### Task 3: Stream Assistant Responses With Response IDs And Interrupts

**Files:**
- Modify: `app/conversation_ws.py`
- Modify: `app/server.py`
- Modify: `tests/test_conversation_ws.py`
- Test: `tests/test_conversation_ws.py`

- [ ] **Step 1: Add failing tests for stale response suppression and full assistant streaming**

```python
def test_ws_conversation_stale_response_done_is_not_forwarded_after_interrupt():
    class StaleService(FakeConversationService):
        async def respond(self, transcript, sample):
            if transcript == "prima":
                return [
                    ("resp-1", {"type": "assistant_text", "response_id": "resp-1", "text": "prima"}),
                ]
            return [
                ("resp-2", {"type": "assistant_text", "response_id": "resp-2", "text": "seconda"}),
                ("resp-2", {"type": "response_done", "response_id": "resp-2"}),
            ]

    client, _service = _make_client(StaleService())

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "voce_1"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        ws.receive_json()
        ws.send_bytes(b"\x00\x01" * 1600)
        ws.send_json({"type": "speech_end"})
        ws.receive_json()
        ws.receive_json()
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        interrupted = ws.receive_json()

    assert interrupted == {"type": "interrupted", "response_id": "resp-1"}
```

- [ ] **Step 2: Run the targeted interrupt tests to verify they fail**

Run: `pytest tests/test_conversation_ws.py -k 'interrupt or stale_response' -v`
Expected: FAIL because response scoping and cancellation are not complete yet.

- [ ] **Step 3: Implement response-scoped assistant streaming on the server**

```python
# app/conversation_ws.py key changes
@dataclass
class ConversationState:
    sample: str | None = None
    collecting: bool = False
    audio_buffer: bytearray = field(default_factory=bytearray)
    active_response_id: str | None = None
    canceled_response_ids: set[str] = field(default_factory=set)


async def _forward_response(websocket: WebSocket, state: ConversationState, response_id: str, outbound: dict[str, Any], audio: bytes | None = None) -> None:
    if response_id in state.canceled_response_ids:
        return
    if outbound.get("response_id") != response_id:
        outbound = {**outbound, "response_id": response_id}
    await websocket.send_json(outbound)
    if audio is not None:
        await websocket.send_bytes(audio)


if event_type == "speech_start":
    state.collecting = True
    state.audio_buffer.clear()
    if state.active_response_id is not None:
        state.canceled_response_ids.add(state.active_response_id)
        await websocket.send_json({"type": "interrupted", "response_id": state.active_response_id})
        state.active_response_id = None
    await websocket.send_json({"type": "listening"})

if event_type == "speech_end":
    transcript = await service.transcribe(bytes(state.audio_buffer), config.input_sample_rate)
    await websocket.send_json({"type": "transcript_final", "text": transcript})
    async for response_id, outbound, audio_bytes in service.stream_response(transcript, state.sample):
        state.active_response_id = response_id
        await _forward_response(websocket, state, response_id, outbound, audio_bytes)
        if outbound.get("type") == "response_done" and state.active_response_id == response_id:
            state.active_response_id = None
```

```python
# app/server.py key changes inside ConversationService
import asyncio
import itertools

from audio_encoder import encode_audio_chunk


class ConversationService:
    def __init__(self, asr, get_model, get_voice_samples, model_lock, ws_config):
        self._asr = asr
        self._get_model = get_model
        self._get_voice_samples = get_voice_samples
        self._model_lock = model_lock
        self._ws_config = ws_config
        self._response_counter = itertools.count(1)

    async def transcribe(self, audio_bytes, sample_rate):
        return await self._asr.transcribe(audio_bytes, sample_rate)

    async def stream_response(self, transcript, sample):
        response_id = f"resp-{next(self._response_counter)}"
        assistant_text = transcript
        yield response_id, {"type": "assistant_text", "response_id": response_id, "text": assistant_text}, None

        message = {"text": assistant_text, "sample": sample, "output_format": "pcm", "num_step": self._ws_config.default_num_step}
        state = WSSessionState(prompt_sample=sample)
        await _ensure_voice_prompt(message, state, self._get_model, self._get_voice_samples, self._model_lock)
        sentences = SentenceSplitter(max_chars=self._ws_config.max_sentence_chars, min_chars=self._ws_config.min_sentence_chars).split_text(assistant_text)

        for sentence_index, sentence in enumerate(sentences):
            audio = await _generate_one(self._get_model, self._model_lock, {"text": sentence, "num_step": self._ws_config.default_num_step, "voice_clone_prompt": state.voice_clone_prompt})
            payload = encode_audio_chunk(audio, output_format="pcm", sample_rate=self._ws_config.sample_rate)
            yield response_id, {
                "type": "audio_chunk",
                "response_id": response_id,
                "chunk_index": sentence_index,
                "sentence": sentence,
                "is_last": sentence_index == len(sentences) - 1,
            }, payload

        yield response_id, {"type": "response_done", "response_id": response_id}, None
```

- [ ] **Step 4: Run the full conversation protocol test file**

Run: `pytest tests/test_conversation_ws.py -v`
Expected: PASS with interruption and response scoping covered.

- [ ] **Step 5: Commit**

```bash
git add app/conversation_ws.py app/server.py tests/test_conversation_ws.py
git commit -m "feat: stream duplex conversation responses"
```

### Task 4: Add Duplex Workstation Client With Silero VAD

**Files:**
- Create: `examples/ws_duplex_client.py`
- Create: `tests/test_ws_duplex_client.py`
- Test: `tests/test_ws_duplex_client.py`

- [ ] **Step 1: Write the failing client-side duplex tests**

```python
import asyncio
import importlib.util
import json
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "ws_duplex_client.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ws_duplex_client", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_client_stops_playback_immediately_on_speech_start():
    client = _load_module()

    class FakePlayback:
        def __init__(self):
            self.stopped = 0

        def stop_output(self):
            self.stopped += 1

    playback = FakePlayback()
    state = client.ClientState(playback=playback)
    client.handle_local_speech_start(state)

    assert playback.stopped == 1
    assert state.user_speaking is True


def test_client_drops_stale_audio_for_old_response_id():
    client = _load_module()

    class FakePlayback:
        def __init__(self):
            self.chunks = []

        def write(self, payload):
            self.chunks.append(payload)

    playback = FakePlayback()
    state = client.ClientState(playback=playback, active_response_id="resp-2")

    accepted = client.handle_audio_chunk_metadata(state, {"type": "audio_chunk", "response_id": "resp-1"})

    assert accepted is False
    assert playback.chunks == []
```

- [ ] **Step 2: Run the client tests to verify they fail**

Run: `pytest tests/test_ws_duplex_client.py -v`
Expected: FAIL because the duplex client module does not exist yet.

- [ ] **Step 3: Implement the minimal duplex workstation client**

```python
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass


@dataclass
class ClientState:
    playback: object
    active_response_id: str | None = None
    user_speaking: bool = False
    awaiting_audio_for_response: str | None = None


def handle_local_speech_start(state: ClientState) -> None:
    state.user_speaking = True
    state.awaiting_audio_for_response = None
    if hasattr(state.playback, "stop_output"):
        state.playback.stop_output()


def handle_audio_chunk_metadata(state: ClientState, event: dict[str, object]) -> bool:
    response_id = event.get("response_id")
    if state.active_response_id is not None and response_id != state.active_response_id:
        return False
    state.active_response_id = response_id if isinstance(response_id, str) else state.active_response_id
    state.awaiting_audio_for_response = state.active_response_id
    return True


async def duplex_loop(websocket, microphone, vad_iterator, playback, sample: str):
    state = ClientState(playback=playback)
    await websocket.send(json.dumps({"type": "session_start", "sample": sample}))

    async def uplink():
        while True:
            chunk = microphone.read_chunk()
            speech_event = vad_iterator(chunk)
            if speech_event == "start":
                handle_local_speech_start(state)
                await websocket.send(json.dumps({"type": "speech_start"}))
            if state.user_speaking:
                await websocket.send(chunk)
            if speech_event == "end":
                state.user_speaking = False
                await websocket.send(json.dumps({"type": "speech_end"}))

    async def downlink():
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                if state.awaiting_audio_for_response == state.active_response_id:
                    playback.write(message)
                continue
            event = json.loads(message)
            if event.get("type") == "interrupted":
                state.active_response_id = None
                state.awaiting_audio_for_response = None
            elif event.get("type") == "audio_chunk":
                if not handle_audio_chunk_metadata(state, event):
                    state.awaiting_audio_for_response = None
            elif event.get("type") == "response_done":
                if event.get("response_id") == state.active_response_id:
                    state.awaiting_audio_for_response = None

    await asyncio.gather(uplink(), downlink())
```

Use lazy imports in `main()` for `websockets`, `pyaudio`, `torch`, and `silero_vad`. Configure microphone capture at 16 kHz mono PCM for uplink and speaker playback at 24 kHz mono PCM for assistant output.

- [ ] **Step 4: Run the duplex client tests to verify they pass**

Run: `pytest tests/test_ws_duplex_client.py -v`
Expected: PASS with local interruption and stale-response dropping covered.

- [ ] **Step 5: Commit**

```bash
git add examples/ws_duplex_client.py tests/test_ws_duplex_client.py
git commit -m "feat: add duplex workstation client"
```

### Task 5: Document Duplex Workflow And Run Final Verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_conversation_ws.py`
- Test: `tests/test_ws_duplex_client.py`

- [ ] **Step 1: Add README documentation for duplex setup and barge-in smoke testing**

```markdown
### Full duplex conversation experiment

The experimental duplex client lives in `examples/ws_duplex_client.py`.

Workstation dependencies:

```bash
python -m pip install websockets pyaudio silero-vad torch
```

Server dependency:

```bash
python -m pip install faster-whisper
```

Run the duplex client from the workstation:

```bash
python examples/ws_duplex_client.py \
  --url ws://SERVER_HOST:8000/ws/conversation \
  --sample voce_1
```

Manual smoke flow:

1. start speaking and wait for an assistant response
2. while the assistant is still talking, start speaking again
3. confirm local playback stops immediately
4. confirm the old response is interrupted and a new one starts
```

- [ ] **Step 2: Run the final automated verification commands**

Run: `pytest tests/test_conversation_ws.py tests/test_ws_duplex_client.py tests/test_ws_handler.py tests/test_ws_playback_client.py tests/test_audio_encoder.py tests/test_sentence_splitter.py -v`
Expected: PASS with all old and new tests green.

- [ ] **Step 3: Run syntax verification on both client examples**

Run: `python -m py_compile examples/ws_playback_client.py examples/ws_duplex_client.py`
Expected: no output and exit code 0.

- [ ] **Step 4: Record the manual smoke-test commands without over-claiming**

Run these manually in a real environment with Docker, GPU, microphone, speakers, and the workstation pointed at the server:

```bash
docker compose up -d --build
curl http://localhost:8000/health
python examples/ws_duplex_client.py --url ws://SERVER_HOST:8000/ws/conversation --sample voce_1
```

Expected: assistant audio starts, user barge-in interrupts playback immediately, and a new response is produced without stale audio leaking through.

- [ ] **Step 5: Commit**

```bash
git add README.md requirements.txt app/asr.py app/conversation_ws.py app/server.py tests/test_conversation_ws.py examples/ws_duplex_client.py tests/test_ws_duplex_client.py
git commit -m "feat: add experimental duplex conversation flow"
```
