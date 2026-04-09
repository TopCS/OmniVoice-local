import importlib.util
from pathlib import Path
import asyncio
import base64
import sys
import threading
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest
from starlette.websockets import WebSocketState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from conversation_ws import (
    ConversationSessionState,
    _forward_service_response,
    _interrupt_active_response,
    create_conversation_router,
)


APP_DIR = Path(__file__).resolve().parents[1] / "app"


def _load_app_module(module_name: str, filename: str):
    module_path = APP_DIR / filename
    assert module_path.is_file(), f"Missing module file: {module_path}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeConversationService:
    def __init__(self):
        self.transcribe_calls = []
        self.response_started = []
        self.response_released = {}

    async def transcribe(self, audio_bytes: bytes, sample_rate: int) -> str:
        self.transcribe_calls.append((audio_bytes, sample_rate))
        return "hello there"

    async def respond(
        self, transcript: str, response_id: int, *, sample: str | None = None
    ):
        self.response_started.append((transcript, response_id, sample))
        yield {"type": "assistant_text", "text": f"answering {transcript}"}

        gate = self.response_released.setdefault(
            response_id, __import__("asyncio").Event()
        )
        await gate.wait()

        yield {
            "type": "audio_chunk",
            "response_id": response_id,
            "payload": b"audio-bytes",
        }
        yield {"type": "response_done", "response_id": response_id}


def _make_client(*, max_input_bytes: int = 1024 * 1024):
    service = FakeConversationService()
    app = FastAPI()
    app.include_router(
        create_conversation_router(service=service, max_input_bytes=max_input_bytes)
    )
    return TestClient(app), service


def _make_client_with_service(service, *, max_input_bytes: int = 1024 * 1024):
    app = FastAPI()
    app.include_router(
        create_conversation_router(service=service, max_input_bytes=max_input_bytes)
    )
    return TestClient(app)


def test_faster_whisper_asr_adapter_transcribes_pcm_bytes(monkeypatch):
    model_calls = {}

    class FakeWhisperModel:
        def __init__(self, model_name, **kwargs):
            model_calls["init"] = (model_name, kwargs)

        def transcribe(self, audio, **kwargs):
            model_calls["audio"] = audio
            model_calls["transcribe_kwargs"] = kwargs

            class Segment:
                def __init__(self, text):
                    self.text = text

            return iter([Segment(" hello"), Segment(" world")]), object()

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=FakeWhisperModel),
    )

    asr = _load_app_module("asr_under_test", "asr.py")
    adapter = asr.FasterWhisperASR(
        model_name="small",
        device="cuda:1",
        compute_type="int8_float16",
    )

    transcript = adapter.transcribe(
        np.array([0, 32767, -32768], dtype=np.int16).tobytes()
    )

    assert transcript == "hello world"
    assert model_calls["init"] == (
        "small",
        {"device": "cuda", "device_index": 1, "compute_type": "int8_float16"},
    )
    assert model_calls["transcribe_kwargs"] == {"beam_size": 1}
    assert model_calls["audio"].tolist() == pytest.approx([0.0, 32767 / 32768, -1.0])


@pytest.mark.asyncio
async def test_server_conversation_service_requires_16khz_audio(monkeypatch):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            return f"heard {len(pcm_bytes)} bytes"

    class FakeOllamaClient:
        def chat(self, **kwargs):
            raise AssertionError(f"chat should not be called: {kwargs}")

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    service = server.ConversationService(
        FakeASR(), assistant_client=FakeOllamaClient(), assistant_model="gemma4"
    )

    with pytest.raises(RuntimeError, match="16 kHz PCM"):
        await service.transcribe(b"abc", 8000)


@pytest.mark.asyncio
async def test_server_conversation_service_streams_response_scoped_text_and_audio(
    monkeypatch,
):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    class FakeAudio:
        shape = (1, 2400)

    class FakeModel:
        def __init__(self):
            self.prompt_calls = []

        def create_voice_clone_prompt(self, **kwargs):
            self.prompt_calls.append(kwargs)
            return {"cached": True, **kwargs}

    class FakeOllamaClient:
        def __init__(self):
            self.chat_calls = []

        async def chat(self, **kwargs):
            self.chat_calls.append(kwargs)
            return {
                "message": {
                    "content": "Assistant first sentence. Assistant second sentence?"
                }
            }

    generated_kwargs = []
    assistant_client = FakeOllamaClient()
    server = _load_server_module(monkeypatch, conversation_enabled=True)
    server.model = FakeModel()
    server.voice_samples = {
        "agent-voice": {"audio_path": "/tmp/agent.wav", "ref_text": "hello"}
    }

    async def fake_generate_one(get_model, model_lock, gen_kwargs):
        generated_kwargs.append(gen_kwargs)
        assert get_model() is server.model
        assert model_lock is server.model_lock
        return FakeAudio()

    monkeypatch.setattr(server, "_generate_one", fake_generate_one)
    monkeypatch.setattr(
        server,
        "encode_audio_chunk",
        lambda audio, output_format, sample_rate=24000: (
            f"{output_format}:{sample_rate}:{audio.shape[-1]}".encode("ascii")
        ),
    )

    service = server.ConversationService(
        FakeASR(),
        assistant_client=assistant_client,
        assistant_model="local-assistant",
    )

    events = [
        event
        async for event in service.respond(
            "Caller transcript that should not be echoed.",
            7,
            sample="agent-voice",
        )
    ]

    assert events == [
        {
            "type": "assistant_text",
            "text": "Assistant first sentence.",
            "response_id": 7,
        },
        {
            "type": "audio_chunk",
            "response_id": 7,
            "chunk_index": 0,
            "sentence": "Assistant first sentence.",
            "duration_ms": 100,
            "generation_time_ms": events[1]["generation_time_ms"],
            "is_last": False,
            "payload": b"pcm:24000:2400",
        },
        {
            "type": "assistant_text",
            "text": "Assistant second sentence?",
            "response_id": 7,
        },
        {
            "type": "audio_chunk",
            "response_id": 7,
            "chunk_index": 1,
            "sentence": "Assistant second sentence?",
            "duration_ms": 100,
            "generation_time_ms": events[3]["generation_time_ms"],
            "is_last": True,
            "payload": b"pcm:24000:2400",
        },
        {"type": "response_done", "response_id": 7},
    ]
    assert generated_kwargs == [
        {
            "text": "Assistant first sentence.",
            "num_step": server.WS_DEFAULT_NUM_STEP,
            "voice_clone_prompt": {
                "cached": True,
                "ref_audio": "/tmp/agent.wav",
                "ref_text": "hello",
            },
        },
        {
            "text": "Assistant second sentence?",
            "num_step": server.WS_DEFAULT_NUM_STEP,
            "voice_clone_prompt": {
                "cached": True,
                "ref_audio": "/tmp/agent.wav",
                "ref_text": "hello",
            },
        },
    ]
    assert assistant_client.chat_calls == [
        {
            "model": "local-assistant",
            "messages": [
                {
                    "role": "user",
                    "content": "Caller transcript that should not be echoed.",
                }
            ],
            "stream": False,
        }
    ]
    assert server.model.prompt_calls == [
        {"ref_audio": "/tmp/agent.wav", "ref_text": "hello"}
    ]


def test_server_registers_conversation_router_when_enabled(monkeypatch):
    server = _load_server_module(monkeypatch, conversation_enabled=True)

    route_paths = {route.path for route in server.app.routes}

    assert "/ws/conversation" in route_paths
    assert server.CONVERSATION_WS_ENABLED is True
    assert server.ASR_MODEL == "tiny.en"
    assert server.ASR_DEVICE == "cpu"
    assert server.ASR_COMPUTE_TYPE == "int8"


def test_server_skips_conversation_router_when_disabled(monkeypatch):
    server = _load_server_module(
        monkeypatch, conversation_enabled=False, broken_asr_import=True
    )

    route_paths = {route.path for route in server.app.routes}

    assert "/ws/conversation" not in route_paths
    assert server.CONVERSATION_WS_ENABLED is False


def test_server_conversation_defaults_ollama_model_to_gemma4(monkeypatch):
    server = _load_server_module(monkeypatch, conversation_enabled=True)

    assert server.OLLAMA_MODEL == "gemma4"


def test_server_conversation_service_uses_configured_ollama_host_and_model(
    monkeypatch,
):
    client_init = []
    asr_init = []

    class FakeAsyncClient:
        def __init__(self, host):
            client_init.append(host)

    class FakeASR:
        def __init__(self, model_name, device, compute_type):
            asr_init.append((model_name, device, compute_type))

    monkeypatch.setenv("OMNIVOICE_OLLAMA_HOST", "http://ollama.internal:11434")
    monkeypatch.setenv("OMNIVOICE_OLLAMA_MODEL", "custom-gemma")
    monkeypatch.setitem(
        sys.modules, "ollama", types.SimpleNamespace(AsyncClient=FakeAsyncClient)
    )

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    monkeypatch.setitem(
        sys.modules, "asr", types.SimpleNamespace(FasterWhisperASR=FakeASR)
    )

    service = server._create_conversation_service()

    assert client_init == ["http://ollama.internal:11434"]
    assert asr_init == [("tiny.en", "cpu", "int8")]
    assert service._assistant_model == "custom-gemma"


def _load_server_module(
    monkeypatch, *, conversation_enabled: bool, broken_asr_import: bool = False
):
    monkeypatch.setenv("OMNIVOICE_WS_ENABLED", "false")
    monkeypatch.setenv(
        "OMNIVOICE_CONVERSATION_WS_ENABLED",
        "true" if conversation_enabled else "false",
    )
    monkeypatch.setenv("OMNIVOICE_ASR_MODEL", "tiny.en")
    monkeypatch.setenv("OMNIVOICE_ASR_DEVICE", "cpu")
    monkeypatch.setenv("OMNIVOICE_ASR_COMPUTE_TYPE", "int8")

    fake_torch = types.SimpleNamespace(
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
        Tensor=object,
    )
    fake_torchaudio = types.SimpleNamespace(save=lambda *args, **kwargs: None)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    if broken_asr_import:
        broken_asr = types.ModuleType("asr")

        def _broken_getattr(name):
            raise AssertionError(f"Unexpected ASR import: {name}")

        broken_asr.__getattr__ = _broken_getattr
        monkeypatch.setitem(sys.modules, "asr", broken_asr)
    else:
        fake_asr = types.SimpleNamespace(FasterWhisperASR=object)
        monkeypatch.setitem(sys.modules, "asr", fake_asr)
    sys.modules.pop("audio_encoder", None)
    sys.modules.pop("ws_handler", None)
    sys.modules.pop("server_under_test", None)

    return _load_app_module("server_under_test", "server.py")


def test_session_start_acknowledges_sample():
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": 16000}
        )

        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
        }


def test_session_start_rejects_unknown_sample_before_audio_collection():
    class ValidatingConversationService(FakeConversationService):
        def __init__(self):
            super().__init__()
            self.validated_samples = []

        def validate_sample(self, sample: str | None) -> None:
            self.validated_samples.append(sample)
            if sample == "missing-voice":
                raise RuntimeError("Sample 'missing-voice' not found.")

    service = ValidatingConversationService()
    client = _make_client_with_service(service)

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "missing-voice"})

        assert ws.receive_json() == {
            "type": "error",
            "message": "Sample 'missing-voice' not found.",
            "code": "INVALID_MESSAGE",
        }

        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
        }

    assert service.validated_samples == ["missing-voice", "agent-voice"]


@pytest.mark.asyncio
async def test_samples_reload_clears_conversation_voice_prompt_cache(monkeypatch):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    service = server.ConversationService(
        FakeASR(), assistant_client=object(), assistant_model="gemma4"
    )
    service._voice_prompt_cache[("agent-voice", "hello")] = {"cached": True}
    server.conversation_service = service
    server.voice_samples = {
        "agent-voice": {"audio_path": "/tmp/old.wav", "ref_text": "hello"}
    }

    def fake_scan_samples(_directory):
        return {"agent-voice": {"audio_path": "/tmp/new.wav", "ref_text": "updated"}}

    monkeypatch.setattr(server, "scan_samples", fake_scan_samples)

    result = await server.reload_samples(None)

    assert result == {"status": "ok", "count": 1}
    assert server.voice_samples == {
        "agent-voice": {"audio_path": "/tmp/new.wav", "ref_text": "updated"}
    }
    assert service._voice_prompt_cache == {}


def test_health_counts_conversation_websocket_connections(monkeypatch):
    server = _load_server_module(monkeypatch, conversation_enabled=True)
    server.model = object()
    server.active_ws_connections = 0
    client = TestClient(server.app)

    with client.websocket_connect("/ws/conversation"):
        assert client.get("/health").status_code == 200
        assert client.get("/health").json()["active_ws_connections"] == 1

    assert client.get("/health").json()["active_ws_connections"] == 0


def test_session_start_invalid_sample_rate_returns_recoverable_error():
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": "bad"}
        )

        assert ws.receive_json() == {
            "type": "error",
            "message": "Field 'sample_rate' must be an integer.",
            "code": "INVALID_MESSAGE",
        }

        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": 16000}
        )

        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
        }


@pytest.mark.parametrize("sample_rate", [0, -16000])
def test_session_start_rejects_non_positive_sample_rate(sample_rate: int):
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {
                "type": "session_start",
                "sample": "agent-voice",
                "sample_rate": sample_rate,
            }
        )

        assert ws.receive_json() == {
            "type": "error",
            "message": "Field 'sample_rate' must be a positive integer.",
            "code": "INVALID_MESSAGE",
        }

        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": 16000}
        )
        assert ws.receive_json() == {
            "type": "session_started",
            "sample": "agent-voice",
            "sample_rate": 16000,
        }


@pytest.mark.parametrize("sample_rate", [8000, 44100])
def test_session_start_rejects_unsupported_positive_sample_rate(sample_rate: int):
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {
                "type": "session_start",
                "sample": "agent-voice",
                "sample_rate": sample_rate,
            }
        )

        assert ws.receive_json() == {
            "type": "error",
            "message": "Field 'sample_rate' must be 16000 for conversation ASR.",
            "code": "INVALID_MESSAGE",
        }


def test_audio_input_limit_returns_controlled_error_and_resets_collection():
    client, service = _make_client(max_input_bytes=5)

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}

        ws.send_bytes(b"abc")
        ws.send_bytes(b"def")

        assert ws.receive_json() == {
            "type": "error",
            "message": "Input audio exceeded the maximum buffered size.",
            "code": "INPUT_AUDIO_TOO_LARGE",
        }

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}

    assert service.transcribe_calls == []


def test_speech_end_triggers_transcript_and_assistant_response():
    client, service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": 16000}
        )
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}

        ws.send_bytes(b"abc")
        ws.send_bytes(b"def")
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }

        service.response_released[1].set()
        assert ws.receive_json() == {"type": "audio_chunk", "response_id": 1}
        assert ws.receive_bytes() == b"audio-bytes"
        assert ws.receive_json() == {"type": "response_done", "response_id": 1}

    assert service.transcribe_calls == [(b"abcdef", 16000)]


def test_barge_in_speech_start_emits_interrupted_for_active_response():
    client, service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json(
            {"type": "session_start", "sample": "agent-voice", "sample_rate": 16000}
        )
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        ws.receive_json()
        ws.send_bytes(b"abc")
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }

        ws.send_json({"type": "speech_start"})

        assert ws.receive_json() == {"type": "interrupted", "response_id": 1}
        assert ws.receive_json() == {"type": "listening"}

        service.response_released[1].set()


def test_duplicate_speech_start_returns_error_and_keeps_buffered_audio():
    client, service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}

        ws.send_bytes(b"abc")
        ws.send_json({"type": "speech_start"})

        assert ws.receive_json() == {
            "type": "error",
            "message": "Received speech_start while already collecting audio.",
            "code": "INVALID_MESSAGE",
        }

        ws.send_bytes(b"def")
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }

        service.response_released[1].set()
        assert ws.receive_json() == {"type": "audio_chunk", "response_id": 1}
        assert ws.receive_bytes() == b"audio-bytes"
        assert ws.receive_json() == {"type": "response_done", "response_id": 1}

    assert service.transcribe_calls == [(b"abcdef", 16000)]


def test_input_audio_chunk_message_appends_audio_payload():
    client, service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}

        ws.send_json(
            {
                "type": "input_audio_chunk",
                "audio": base64.b64encode(b"abc").decode("ascii"),
            }
        )
        ws.send_bytes(b"def")
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }

        service.response_released[1].set()
        assert ws.receive_json() == {"type": "audio_chunk", "response_id": 1}
        assert ws.receive_bytes() == b"audio-bytes"
        assert ws.receive_json() == {"type": "response_done", "response_id": 1}

    assert service.transcribe_calls == [(b"abcdef", 16000)]


def test_unexpected_audio_before_speech_start_returns_error():
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_bytes(b"abc")

        assert ws.receive_json() == {
            "type": "error",
            "message": "Audio received outside active speech input.",
            "code": "UNEXPECTED_AUDIO",
        }


def test_speech_end_without_active_speech_returns_error():
    client, _service = _make_client()

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {
            "type": "error",
            "message": "Received speech_end without speech_start.",
            "code": "INVALID_MESSAGE",
        }


def test_transcribe_failure_returns_controlled_error_and_session_continues():
    class FailingTranscribeService(FakeConversationService):
        async def transcribe(self, audio_bytes: bytes, sample_rate: int) -> str:
            self.transcribe_calls.append((audio_bytes, sample_rate))
            raise RuntimeError("transcribe failed")

    service = FailingTranscribeService()
    client = _make_client_with_service(service)

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}
        ws.send_bytes(b"abc")
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {
            "type": "error",
            "message": "transcribe failed",
            "code": "TRANSCRIBE_ERROR",
        }

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}


@pytest.mark.asyncio
async def test_response_error_fallback_ignores_disconnect_time_send_failures():
    class DisconnectingWebSocket:
        application_state = WebSocketState.CONNECTED

        async def send_json(self, payload):
            raise RuntimeError("socket closed")

        async def send_bytes(self, payload):
            raise AssertionError("send_bytes should not be called")

    class FailingConversationService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            raise RuntimeError("response failed")
            yield

    state = ConversationSessionState(active_response_id=1)
    state.response_task = asyncio.current_task()

    await _forward_service_response(
        DisconnectingWebSocket(),
        state,
        FailingConversationService(),
        "hello there",
        1,
    )

    assert state.active_response_id is None
    assert state.response_task is None


@pytest.mark.asyncio
async def test_forward_service_response_stops_forwarding_stale_events():
    class CollectingWebSocket:
        def __init__(self, state):
            self.application_state = WebSocketState.CONNECTED
            self.json_payloads = []
            self.binary_payloads = []
            self._state = state

        async def send_json(self, payload):
            self.json_payloads.append(payload)
            if payload["type"] == "assistant_text":
                self._state.active_response_id = 2

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    class LateEventService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            yield {"type": "assistant_text", "text": transcript}
            yield {"type": "audio_chunk", "payload": b"late-audio"}
            yield {"type": "response_done"}

    state = ConversationSessionState(active_response_id=1)
    state.response_task = asyncio.current_task()
    websocket = CollectingWebSocket(state)

    await _forward_service_response(websocket, state, LateEventService(), "hello", 1)

    assert websocket.json_payloads == [
        {"type": "assistant_text", "text": "hello", "response_id": 1}
    ]
    assert websocket.binary_payloads == []
    assert state.active_response_id == 2
    assert state.response_task == asyncio.current_task()


@pytest.mark.asyncio
async def test_forward_service_response_normalizes_response_ids_from_service_events():
    class CollectingWebSocket:
        application_state = WebSocketState.CONNECTED

        def __init__(self):
            self.json_payloads = []
            self.binary_payloads = []

        async def send_json(self, payload):
            if payload["type"] == "interrupted":
                assert generation_finished.is_set() is True
            self.json_payloads.append(payload)

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    class MismatchedResponseIdService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            yield {"type": "assistant_text", "text": transcript, "response_id": 999}
            yield {
                "type": "audio_chunk",
                "response_id": -1,
                "payload": b"audio",
            }
            yield {"type": "response_done", "response_id": 0}

    state = ConversationSessionState(active_response_id=7)
    state.response_task = asyncio.current_task()
    websocket = CollectingWebSocket()

    await _forward_service_response(
        websocket,
        state,
        MismatchedResponseIdService(),
        "hello",
        7,
    )

    assert websocket.json_payloads == [
        {"type": "assistant_text", "text": "hello", "response_id": 7},
        {"type": "audio_chunk", "response_id": 7},
        {"type": "response_done", "response_id": 7},
    ]
    assert websocket.binary_payloads == [b"audio"]
    assert state.active_response_id is None
    assert state.response_task is None


@pytest.mark.asyncio
async def test_forward_service_response_suppresses_error_for_stale_response_failure():
    class CollectingWebSocket:
        def __init__(self, state):
            self.application_state = WebSocketState.CONNECTED
            self.json_payloads = []
            self.binary_payloads = []
            self._state = state
            self._send_count = 0

        async def send_json(self, payload):
            self.json_payloads.append(payload)
            self._send_count += 1
            if payload["type"] == "assistant_text":
                self._state.active_response_id = 2
            if self._send_count == 1:
                raise RuntimeError("socket send failed after cancellation")

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    class StaleFailingService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            yield {"type": "assistant_text", "text": transcript}

    state = ConversationSessionState(active_response_id=1)
    state.response_task = asyncio.current_task()
    websocket = CollectingWebSocket(state)

    await _forward_service_response(websocket, state, StaleFailingService(), "hello", 1)

    assert websocket.json_payloads == [
        {"type": "assistant_text", "text": "hello", "response_id": 1}
    ]
    assert websocket.binary_payloads == []
    assert state.active_response_id == 2
    assert state.response_task == asyncio.current_task()


@pytest.mark.asyncio
async def test_forward_service_response_preserves_audio_pairing_during_stale_race():
    class CollectingWebSocket:
        def __init__(self, state):
            self.application_state = WebSocketState.CONNECTED
            self.json_payloads = []
            self.binary_payloads = []
            self._state = state

        async def send_json(self, payload):
            self.json_payloads.append(payload)
            if payload["type"] == "audio_chunk":
                self._state.active_response_id = 2

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    class LateAudioService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            yield {"type": "audio_chunk", "payload": b"audio"}
            yield {"type": "response_done"}

    state = ConversationSessionState(active_response_id=1)
    state.response_task = asyncio.current_task()
    websocket = CollectingWebSocket(state)

    await _forward_service_response(websocket, state, LateAudioService(), "hello", 1)

    assert websocket.json_payloads == [{"type": "audio_chunk", "response_id": 1}]
    assert websocket.binary_payloads == [b"audio"]
    assert state.active_response_id == 2
    assert state.response_task == asyncio.current_task()


@pytest.mark.asyncio
async def test_interrupt_waits_for_response_task_before_emitting_interrupted():
    class CollectingWebSocket:
        application_state = WebSocketState.CONNECTED

        def __init__(self, lifecycle):
            self.events = []
            self._lifecycle = lifecycle

        async def send_json(self, payload):
            self.events.append((payload, self._lifecycle.copy()))

    lifecycle = []
    websocket = CollectingWebSocket(lifecycle)
    state = ConversationSessionState(active_response_id=3)
    started = asyncio.Event()

    async def response_task_body():
        try:
            started.set()
            await asyncio.Future()
        finally:
            lifecycle.append("task-finally")

    task = asyncio.create_task(response_task_body())
    state.response_task = task
    state.background_response_tasks.add(task)
    await started.wait()

    await _interrupt_active_response(websocket, state)

    assert task.done() is True
    assert websocket.events == [
        ({"type": "interrupted", "response_id": 3}, ["task-finally"])
    ]


@pytest.mark.asyncio
async def test_interrupt_waits_for_blocking_generation_before_emitting_interrupted(
    monkeypatch,
):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    class FakeAudio:
        shape = (1, 2400)

    class FakeOllamaClient:
        async def chat(self, **kwargs):
            del kwargs
            return {"message": {"content": "This sentence has enough words."}}

    class CollectingWebSocket:
        application_state = WebSocketState.CONNECTED

        def __init__(self):
            self.json_payloads = []
            self.binary_payloads = []

        async def send_json(self, payload):
            self.json_payloads.append(payload)

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    generation_started = asyncio.Event()
    release_generation = threading.Event()
    generation_finished = threading.Event()
    server = _load_server_module(monkeypatch, conversation_enabled=True)
    server.model = object()

    async def fake_generate_one(get_model, model_lock, gen_kwargs):
        del get_model, model_lock, gen_kwargs
        generation_started.set()
        await asyncio.to_thread(release_generation.wait)
        generation_finished.set()
        return FakeAudio()

    monkeypatch.setattr(server, "_generate_one", fake_generate_one)
    monkeypatch.setattr(server, "encode_audio_chunk", lambda *args, **kwargs: b"pcm")

    service = server.ConversationService(
        FakeASR(), assistant_client=FakeOllamaClient(), assistant_model="gemma4"
    )
    websocket = CollectingWebSocket()
    state = ConversationSessionState(active_response_id=1)

    response_task = asyncio.create_task(
        _forward_service_response(
            websocket,
            state,
            service,
            "This sentence has enough words.",
            1,
        )
    )
    state.response_task = response_task
    state.background_response_tasks.add(response_task)

    await generation_started.wait()

    interrupt_task = asyncio.create_task(_interrupt_active_response(websocket, state))
    await asyncio.sleep(0)

    assert interrupt_task.done() is False
    assert websocket.json_payloads == [
        {
            "type": "assistant_text",
            "text": "This sentence has enough words.",
            "response_id": 1,
        }
    ]

    release_generation.set()

    await interrupt_task
    with pytest.raises(asyncio.CancelledError):
        await response_task

    assert websocket.json_payloads == [
        {
            "type": "assistant_text",
            "text": "This sentence has enough words.",
            "response_id": 1,
        },
        {"type": "interrupted", "response_id": 1},
    ]
    assert websocket.binary_payloads == []


@pytest.mark.asyncio
async def test_interrupt_cancels_inflight_async_ollama_request_before_emitting_interrupted(
    monkeypatch,
):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    class FakeOllamaClient:
        def __init__(self):
            self.chat_calls = []
            self.cancelled = asyncio.Event()
            self.started = asyncio.Event()

        async def chat(self, **kwargs):
            self.chat_calls.append(kwargs)
            self.started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    class CollectingWebSocket:
        application_state = WebSocketState.CONNECTED

        def __init__(self):
            self.json_payloads = []
            self.binary_payloads = []

        async def send_json(self, payload):
            self.json_payloads.append(payload)

        async def send_bytes(self, payload):
            self.binary_payloads.append(payload)

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    assistant_client = FakeOllamaClient()
    service = server.ConversationService(
        FakeASR(), assistant_client=assistant_client, assistant_model="gemma4"
    )
    websocket = CollectingWebSocket()
    state = ConversationSessionState(active_response_id=1)

    response_task = asyncio.create_task(
        _forward_service_response(
            websocket,
            state,
            service,
            "This sentence has enough words.",
            1,
        )
    )
    state.response_task = response_task
    state.background_response_tasks.add(response_task)

    await assistant_client.started.wait()

    await _interrupt_active_response(websocket, state)
    with pytest.raises(asyncio.CancelledError):
        await response_task

    assert assistant_client.chat_calls == [
        {
            "model": "gemma4",
            "messages": [
                {
                    "role": "user",
                    "content": "This sentence has enough words.",
                }
            ],
            "stream": False,
        }
    ]
    assert assistant_client.cancelled.is_set() is True
    assert websocket.json_payloads == [{"type": "interrupted", "response_id": 1}]
    assert websocket.binary_payloads == []


def test_response_task_failure_returns_controlled_error_event():
    class FailingConversationService(FakeConversationService):
        async def respond(
            self, transcript: str, response_id: int, *, sample: str | None = None
        ):
            self.response_started.append((transcript, response_id, sample))
            yield {"type": "assistant_text", "text": f"answering {transcript}"}
            raise RuntimeError("response failed")

    service = FailingConversationService()
    client = _make_client_with_service(service)

    with client.websocket_connect("/ws/conversation") as ws:
        ws.send_json({"type": "session_start", "sample": "agent-voice"})
        ws.receive_json()

        ws.send_json({"type": "speech_start"})
        ws.receive_json()
        ws.send_bytes(b"abc")
        ws.send_json({"type": "speech_end"})

        assert ws.receive_json() == {"type": "transcript_final", "text": "hello there"}
        assert ws.receive_json() == {
            "type": "assistant_text",
            "text": "answering hello there",
            "response_id": 1,
        }
        assert ws.receive_json() == {
            "type": "error",
            "message": "response failed",
            "code": "RESPONSE_ERROR",
            "response_id": 1,
        }

        ws.send_json({"type": "speech_start"})
        assert ws.receive_json() == {"type": "listening"}
