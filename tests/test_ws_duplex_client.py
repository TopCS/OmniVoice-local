from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples" / "ws_duplex_client.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ws_duplex_client", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakePlayback:
    def __init__(self):
        self.interrupt_calls = 0
        self.chunks = []

    def interrupt(self) -> None:
        self.interrupt_calls += 1

    def enqueue(self, chunk: bytes) -> None:
        self.chunks.append(chunk)


class FakeStream:
    def __init__(self):
        self.stop_calls = 0
        self.close_calls = 0
        self.start_calls = 0
        self.writes = []
        self.write_started = threading.Event()
        self.write_release = threading.Event()
        self.block_writes = False

    def stop_stream(self) -> None:
        self.stop_calls += 1

    def start_stream(self) -> None:
        self.start_calls += 1

    def write(self, data: bytes) -> None:
        self.writes.append(data)
        self.write_started.set()
        if self.block_writes:
            self.write_release.wait(timeout=1)

    def close(self) -> None:
        self.close_calls += 1


class FakePyAudioInstance:
    def __init__(self):
        self.open_calls = []
        self.terminated = False
        self.streams = []

    def open(self, **kwargs):
        self.open_calls.append(kwargs)
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def terminate(self) -> None:
        self.terminated = True


def test_duplex_state_interrupts_local_playback_on_speech_start():
    client = _load_module()
    state = client.DuplexState()
    playback = FakePlayback()

    assert state.start_local_speech(playback) == {"type": "speech_start"}
    assert playback.interrupt_calls == 1

    assert state.start_local_speech(playback) is None
    assert playback.interrupt_calls == 1


def test_duplex_state_drops_stale_audio_after_local_interrupt():
    client = _load_module()
    state = client.DuplexState()
    playback = FakePlayback()

    assert state.handle_event(
        {"type": "assistant_text", "response_id": 1}, playback
    ) == {
        "type": "assistant_text",
        "response_id": 1,
    }
    assert state.handle_event({"type": "audio_chunk", "response_id": 1}, playback) == {
        "type": "audio_chunk",
        "response_id": 1,
    }

    assert state.start_local_speech(playback) == {"type": "speech_start"}
    assert state.handle_audio_bytes(b"old-response", playback) is False
    assert playback.chunks == []
    assert state.end_local_speech() == {"type": "speech_end"}

    assert (
        state.handle_event({"type": "audio_chunk", "response_id": 1}, playback) is None
    )
    assert state.handle_event({"type": "audio_chunk", "response_id": 2}, playback) == {
        "type": "audio_chunk",
        "response_id": 2,
    }
    assert state.handle_audio_bytes(b"new-response", playback) is True
    assert playback.chunks == [b"new-response"]


def test_duplex_state_tracks_session_and_response_lifecycle():
    client = _load_module()
    state = client.DuplexState()
    playback = FakePlayback()

    assert state.handle_event(
        {"type": "session_started", "sample": "agent", "sample_rate": 16000},
        playback,
    ) == {"type": "session_started", "sample": "agent", "sample_rate": 16000}
    assert state.session_started is True
    assert state.sample == "agent"
    assert state.sample_rate == 16000

    assert state.handle_event({"type": "listening"}, playback) == {"type": "listening"}
    assert state.server_listening is True

    assert state.handle_event(
        {"type": "transcript_final", "text": "hello"}, playback
    ) == {
        "type": "transcript_final",
        "text": "hello",
    }
    assert state.server_listening is False

    assert state.handle_event(
        {"type": "assistant_text", "text": "hi", "response_id": 3}, playback
    ) == {"type": "assistant_text", "text": "hi", "response_id": 3}
    assert state.active_response_id == 3

    assert state.handle_event({"type": "interrupted", "response_id": 3}, playback) == {
        "type": "interrupted",
        "response_id": 3,
    }
    assert state.active_response_id is None

    assert (
        state.handle_event({"type": "response_done", "response_id": 3}, playback)
        is None
    )


def test_open_audio_streams_and_main_use_expected_runtime_configuration(monkeypatch):
    client = _load_module()
    pyaudio_instance = FakePyAudioInstance()
    pyaudio_module = types.SimpleNamespace(
        paInt16="pcm16",
        PyAudio=lambda: pyaudio_instance,
    )

    audio_api, input_stream, output_stream = client.open_audio_streams(pyaudio_module)

    assert pyaudio_instance.open_calls == [
        {
            "format": "pcm16",
            "channels": 1,
            "rate": 16000,
            "input": True,
            "frames_per_buffer": 512,
        },
        {"format": "pcm16", "channels": 1, "rate": 24000, "output": True},
    ]

    input_stream.stop_stream()
    input_stream.close()
    output_stream.stop_stream()
    output_stream.close()
    audio_api.terminate()
    assert pyaudio_instance.terminated is True

    captured = {}

    async def fake_run_client(
        url,
        sample,
        websockets_module,
        pyaudio_module,
        torch_module,
        silero_vad_module,
    ):
        captured.update(
            {
                "url": url,
                "sample": sample,
                "websockets": websockets_module,
                "pyaudio": pyaudio_module,
                "torch": torch_module,
                "silero_vad": silero_vad_module,
            }
        )

    monkeypatch.setattr(client, "run_client", fake_run_client)
    monkeypatch.setitem(sys.modules, "websockets", types.ModuleType("websockets"))
    monkeypatch.setitem(sys.modules, "pyaudio", types.ModuleType("pyaudio"))
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "silero_vad", types.ModuleType("silero_vad"))

    assert (
        client.main(["--url", "ws://example/ws/conversation", "--sample", "agent"]) == 0
    )
    assert captured == {
        "url": "ws://example/ws/conversation",
        "sample": "agent",
        "websockets": sys.modules["websockets"],
        "pyaudio": sys.modules["pyaudio"],
        "torch": sys.modules["torch"],
        "silero_vad": sys.modules["silero_vad"],
    }


class FakeWebSocket:
    def __init__(self, messages):
        self._messages = iter(messages)

    async def recv(self):
        return next(self._messages)


@pytest.mark.asyncio
async def test_playback_controller_interrupt_drops_buffered_audio():
    client = _load_module()
    stream = FakeStream()
    stream.block_writes = True
    playback = client.PlaybackController(stream)

    task = asyncio.create_task(playback.run())
    playback.enqueue(b"first")
    await asyncio.to_thread(stream.write_started.wait, 1)
    playback.enqueue(b"second")

    playback.interrupt()
    playback.enqueue(b"third")
    stream.write_release.set()

    for _ in range(20):
        if stream.writes == [b"first", b"third"]:
            break
        await asyncio.sleep(0.01)
    await playback.close()
    await task

    assert stream.writes == [b"first", b"third"]
    assert stream.stop_calls == 0
    assert stream.start_calls == 0


@pytest.mark.asyncio
async def test_playback_controller_close_drops_buffered_audio():
    client = _load_module()
    stream = FakeStream()
    stream.block_writes = True
    playback = client.PlaybackController(stream)

    task = asyncio.create_task(playback.run())
    playback.enqueue(b"first")
    await asyncio.to_thread(stream.write_started.wait, 1)
    playback.enqueue(b"second")

    await playback.close()
    stream.write_release.set()
    await task

    assert stream.writes == [b"first"]


@pytest.mark.asyncio
async def test_receive_loop_raises_on_binary_frame_without_metadata():
    client = _load_module()
    state = client.DuplexState()
    playback = FakePlayback()
    websocket = FakeWebSocket([b"orphan-audio"])

    with pytest.raises(
        RuntimeError, match="Received binary frame without audio_chunk metadata"
    ):
        await client.receive_loop(
            websocket, state, playback, event_sink=lambda event: None
        )


@pytest.mark.asyncio
async def test_receive_loop_raises_when_metadata_is_not_followed_by_binary():
    client = _load_module()
    state = client.DuplexState(active_response_id=1, last_response_id=1)
    playback = FakePlayback()
    websocket = FakeWebSocket(
        [
            '{"type": "audio_chunk", "response_id": 1}',
            '{"type": "response_done", "response_id": 1}',
        ]
    )

    with pytest.raises(RuntimeError, match="Received response_done before audio bytes"):
        await client.receive_loop(
            websocket, state, playback, event_sink=lambda event: None
        )


@pytest.mark.asyncio
async def test_receive_loop_drops_stale_binary_after_local_barge_in():
    client = _load_module()
    state = client.DuplexState(active_response_id=1, last_response_id=1)
    playback = FakePlayback()
    websocket = FakeWebSocket(
        [
            '{"type": "audio_chunk", "response_id": 1}',
            b"stale-audio",
            '{"type": "error", "message": "stop", "code": "STOP"}',
        ]
    )

    def event_sink(event):
        if event["type"] == "audio_chunk":
            assert state.start_local_speech(playback) == {"type": "speech_start"}

    with pytest.raises(RuntimeError, match="stop"):
        await client.receive_loop(websocket, state, playback, event_sink=event_sink)

    assert playback.chunks == []


@pytest.mark.asyncio
async def test_receive_loop_allows_interrupted_after_audio_metadata_without_binary():
    client = _load_module()
    state = client.DuplexState(active_response_id=1, last_response_id=1)
    playback = FakePlayback()
    events = []
    websocket = FakeWebSocket(
        [
            '{"type": "audio_chunk", "response_id": 1}',
            '{"type": "interrupted", "response_id": 1}',
            '{"type": "error", "message": "stop", "code": "STOP"}',
        ]
    )

    with pytest.raises(RuntimeError, match="stop"):
        await client.receive_loop(websocket, state, playback, event_sink=events.append)

    assert events == [
        {"type": "audio_chunk", "response_id": 1},
        {"type": "interrupted", "response_id": 1},
        {"type": "error", "message": "stop", "code": "STOP"},
    ]
