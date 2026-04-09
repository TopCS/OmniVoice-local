from pathlib import Path
import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local test fallback
    torch = None
    sys.modules["torch"] = types.SimpleNamespace(Tensor=object)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import ws_handler
from ws_handler import WSConfig, create_ws_router


class FakeAudio:
    def __init__(self, frames: int = 2400):
        self.shape = (1, frames)


class FakeModel:
    def __init__(self):
        self.generate_calls = []
        self.prompt_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        if hasattr(torch, "zeros"):
            return [torch.zeros((1, 2400), dtype=torch.float32)]
        return [FakeAudio(2400)]

    def create_voice_clone_prompt(self, **kwargs):
        self.prompt_calls.append(kwargs)
        return {"cached": True, **kwargs}


def _make_client(config: WSConfig | None = None):
    ws_handler.encode_audio_chunk = lambda audio, output_format, sample_rate=24000: (
        b"x" * 4800
    )
    model = FakeModel()
    app = FastAPI()
    app.include_router(
        create_ws_router(
            get_model=lambda: model,
            get_voice_samples=lambda: {
                "agent-voice": {"audio_path": "/tmp/agent.wav", "ref_text": "hello"}
            },
            model_lock=__import__("asyncio").Lock(),
            config=config or WSConfig(min_sentence_chars=1, buffer_timeout_ms=50),
            active_counter=lambda delta: None,
        )
    )
    return TestClient(app), model


def test_ws_synthesize_sends_metadata_then_binary_for_each_chunk():
    client, model = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50)
    )

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json(
            {
                "type": "synthesize",
                "text": "Ciao! Come va?",
                "output_format": "pcm",
                "num_step": 16,
            }
        )

        meta1 = ws.receive_json()
        audio1 = ws.receive_bytes()
        meta2 = ws.receive_json()
        audio2 = ws.receive_bytes()
        done = ws.receive_json()

    assert meta1["type"] == "audio_chunk"
    assert meta1["chunk_index"] == 0
    assert meta1["sentence"] == "Ciao!"
    assert meta1["is_last"] is False
    assert meta1["duration_ms"] == 100
    assert isinstance(audio1, bytes) and len(audio1) == 4800

    assert meta2["type"] == "audio_chunk"
    assert meta2["chunk_index"] == 1
    assert meta2["sentence"] == "Come va?"
    assert meta2["is_last"] is True
    assert isinstance(audio2, bytes) and len(audio2) == 4800

    assert done == {
        "type": "done",
        "total_chunks": 2,
        "total_duration_ms": 200,
        "total_generation_time_ms": done["total_generation_time_ms"],
    }
    assert len(model.generate_calls) == 2
    assert all(call["num_step"] == 16 for call in model.generate_calls)


def test_ws_synthesize_duplicate_sentence_text_marks_only_final_chunk_as_last():
    client, _model = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50)
    )

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json(
            {
                "type": "synthesize",
                "text": "Echo. Echo.",
                "output_format": "pcm",
            }
        )

        meta1 = ws.receive_json()
        ws.receive_bytes()
        meta2 = ws.receive_json()
        ws.receive_bytes()
        ws.receive_json()

    assert meta1["sentence"] == "Echo."
    assert meta1["is_last"] is False
    assert meta2["sentence"] == "Echo."
    assert meta2["is_last"] is True


def test_ws_text_chunk_and_flush_streaming_mode():
    client, model = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50)
    )

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "text_chunk", "text": "Hello."})
        meta1 = ws.receive_json()
        audio1 = ws.receive_bytes()

        ws.send_json({"type": "text_chunk", "text": "Pending tail without punctuation"})
        ws.send_json({"type": "text_flush"})
        meta2 = ws.receive_json()
        audio2 = ws.receive_bytes()
        done = ws.receive_json()

    assert meta1["type"] == "audio_chunk"
    assert meta1["sentence"] == "Hello."
    assert isinstance(audio1, bytes) and len(audio1) == 4800

    assert meta2["type"] == "audio_chunk"
    assert meta2["sentence"] == "Pending tail without punctuation"
    assert isinstance(audio2, bytes) and len(audio2) == 4800
    assert done["type"] == "done"
    assert done["total_chunks"] == 2
    assert len(model.generate_calls) == 2
    assert all(call["num_step"] == 16 for call in model.generate_calls)


def test_ws_set_voice_caches_prompt_for_following_chunks():
    client, model = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50)
    )

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "set_voice", "sample": "agent-voice"})
        ack = ws.receive_json()

        ws.send_json({"type": "synthesize", "text": "Voice clone test."})
        meta = ws.receive_json()
        audio = ws.receive_bytes()
        done = ws.receive_json()

    assert ack == {"type": "voice_set", "sample": "agent-voice"}
    assert meta["type"] == "audio_chunk"
    assert isinstance(audio, bytes) and len(audio) == 4800
    assert done["type"] == "done"

    assert len(model.prompt_calls) == 1
    assert len(model.generate_calls) == 1
    assert model.generate_calls[0]["voice_clone_prompt"]["cached"] is True


def test_ws_set_voice_missing_sample_sends_session_error_then_closes():
    client, _model = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50)
    )

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "set_voice", "sample": "missing-sample"})

        assert ws.receive_json() == {
            "type": "error",
            "message": "Sample 'missing-sample' not found.",
            "code": "SESSION_ERROR",
        }

        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ws_synthesize_invalid_output_format_returns_error_without_done():
    client, model = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50)
    )

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json(
            {
                "type": "synthesize",
                "text": "Ciao!",
                "output_format": "mp3",
            }
        )

        assert ws.receive_json() == {
            "type": "error",
            "message": "Unsupported output_format 'mp3'.",
            "code": "UNSUPPORTED_FORMAT",
        }

        assert model.generate_calls == []

        ws.send_json(
            {
                "type": "synthesize",
                "text": "Ciao!",
                "output_format": "pcm",
            }
        )

        meta = ws.receive_json()
        audio = ws.receive_bytes()
        done = ws.receive_json()

    assert meta["type"] == "audio_chunk"
    assert meta["sentence"] == "Ciao!"
    assert isinstance(audio, bytes) and len(audio) == 4800
    assert done["type"] == "done"
    assert len(model.generate_calls) == 1


def test_ws_idle_timeout_sends_error_then_closes_socket():
    client, _model = _make_client(
        WSConfig(
            min_sentence_chars=1,
            max_sentence_chars=150,
            buffer_timeout_ms=50,
            inactivity_timeout_s=0.1,
        )
    )

    with client.websocket_connect("/ws/tts") as ws:
        assert ws.receive_json() == {
            "type": "error",
            "message": "Connection idle timeout.",
            "code": "IDLE_TIMEOUT",
        }

        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_ws_text_flush_invalid_output_format_returns_error_without_done():
    client, model = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50)
    )

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "text_chunk", "text": "Ciao"})
        ws.send_json({"type": "text_flush", "output_format": "mp3"})

        assert ws.receive_json() == {
            "type": "error",
            "message": "Unsupported output_format 'mp3'.",
            "code": "UNSUPPORTED_FORMAT",
        }

        assert model.generate_calls == []

        ws.send_json({"type": "text_chunk", "text": "Hello"})
        ws.send_json({"type": "text_flush", "output_format": "pcm"})

        meta = ws.receive_json()
        audio = ws.receive_bytes()
        done = ws.receive_json()

    assert meta["type"] == "audio_chunk"
    assert meta["sentence"] == "Hello"
    assert isinstance(audio, bytes) and len(audio) == 4800
    assert done["type"] == "done"
    assert len(model.generate_calls) == 1


def test_ws_text_flush_empty_buffer_still_emits_done():
    client, model = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50)
    )

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json({"type": "text_flush"})
        done = ws.receive_json()

    assert done == {
        "type": "done",
        "total_chunks": 0,
        "total_duration_ms": 0,
        "total_generation_time_ms": 0,
    }
    assert model.generate_calls == []
