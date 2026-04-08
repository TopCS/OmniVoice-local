"""WebSocket handler for sentence-level streaming OmniVoice TTS."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from audio_encoder import encode_audio_chunk
from sentence_splitter import SentenceSplitter

log = logging.getLogger("omnivoice-ws")


@dataclass
class WSConfig:
    """Configuration for `/ws/tts` behavior."""

    enabled: bool = True
    default_num_step: int = 16
    max_sentence_chars: int = 150
    min_sentence_chars: int = 20
    buffer_timeout_ms: int = 2000
    inactivity_timeout_s: int = 60
    sample_rate: int = 24000


@dataclass
class WSSessionState:
    """Per-connection state for incremental streaming synthesis."""

    text_buffer: str = ""
    chunk_index: int = 0
    total_duration_ms: int = 0
    total_generation_time_ms: int = 0
    prompt_sample: Optional[str] = None
    prompt_text: Optional[str] = None
    voice_clone_prompt: Any = None
    opened_at: float = field(default_factory=time.monotonic)


def create_ws_router(
    *,
    get_model: Callable[[], Any],
    get_voice_samples: Callable[[], dict[str, dict[str, Any]]],
    model_lock: asyncio.Lock,
    config: WSConfig,
    active_counter: Callable[[int], None],
) -> APIRouter:
    """Create router exposing `/ws/tts` endpoint."""

    router = APIRouter()

    @router.websocket("/ws/tts")
    async def ws_tts(websocket: WebSocket) -> None:
        await websocket.accept()
        if not config.enabled:
            await _send_error(websocket, "WebSocket endpoint disabled.", "WS_DISABLED")
            await websocket.close(code=1008)
            return

        splitter = SentenceSplitter(max_chars=config.max_sentence_chars, min_chars=config.min_sentence_chars)
        state = WSSessionState()
        active_counter(1)
        log.info("WS connected from %s", websocket.client)

        try:
            while True:
                timeout_s = (
                    max(config.buffer_timeout_ms / 1000.0, 0.1)
                    if state.text_buffer
                    else float(config.inactivity_timeout_s)
                )
                msg = await _recv_json(websocket, timeout_s=timeout_s, idle_timeout_s=config.inactivity_timeout_s)
                if msg is None:
                    break
                if msg.get("type") == "__buffer_timeout__":
                    pending = state.text_buffer.strip()
                    state.text_buffer = ""
                    if pending:
                        await _synthesize_sentences(
                            websocket,
                            splitter.split_text(pending),
                            {"type": "text_chunk"},
                            state,
                            get_model,
                            model_lock,
                            config,
                        )
                    continue
                await _handle_message(websocket, msg, state, splitter, get_model, get_voice_samples, model_lock, config)
        except WebSocketDisconnect:
            log.info("WS disconnected by client %s", websocket.client)
        except Exception as exc:  # noqa: BLE001
            log.exception("WS session error: %s", exc)
            await _send_error(websocket, str(exc), "SESSION_ERROR")
        finally:
            active_counter(-1)
            elapsed = int((time.monotonic() - state.opened_at) * 1000)
            log.info(
                "WS closed client=%s chunks=%d audio_ms=%d gen_ms=%d session_ms=%d",
                websocket.client,
                state.chunk_index,
                state.total_duration_ms,
                state.total_generation_time_ms,
                elapsed,
            )
            await websocket.close()

    return router


async def _handle_message(
    websocket: WebSocket,
    message: dict[str, Any],
    state: WSSessionState,
    splitter: SentenceSplitter,
    get_model: Callable[[], Any],
    get_voice_samples: Callable[[], dict[str, dict[str, Any]]],
    model_lock: asyncio.Lock,
    config: WSConfig,
) -> None:
    msg_type = message.get("type")
    if msg_type == "synthesize":
        text = str(message.get("text", "")).strip()
        if not text:
            await _send_error(websocket, "Field 'text' must not be empty.", "INVALID_TEXT")
            return
        await _ensure_voice_prompt(message, state, get_model, get_voice_samples, model_lock)
        sentences = splitter.split_text(text)
        await _synthesize_sentences(websocket, sentences, message, state, get_model, model_lock, config)
        await _send_done(websocket, state)
        return

    if msg_type == "text_chunk":
        chunk = str(message.get("text", ""))
        if not chunk:
            return
        await _ensure_voice_prompt(message, state, get_model, get_voice_samples, model_lock)
        state.text_buffer += chunk
        complete, remainder = splitter.extract_complete_sentences(state.text_buffer)
        state.text_buffer = remainder
        if complete:
            await _synthesize_sentences(websocket, splitter._post_process(complete), message, state, get_model, model_lock, config)
        return

    if msg_type == "text_flush":
        pending = state.text_buffer.strip()
        state.text_buffer = ""
        if pending:
            await _synthesize_sentences(websocket, splitter.split_text(pending), message, state, get_model, model_lock, config)
        await _send_done(websocket, state)
        return

    if msg_type == "set_voice":
        await _ensure_voice_prompt(message, state, get_model, get_voice_samples, model_lock, force=True)
        await websocket.send_json({"type": "voice_set", "sample": state.prompt_sample})
        return

    await _send_error(websocket, f"Unsupported message type '{msg_type}'.", "INVALID_MESSAGE")


async def _synthesize_sentences(
    websocket: WebSocket,
    sentences: list[str],
    message: dict[str, Any],
    state: WSSessionState,
    get_model: Callable[[], Any],
    model_lock: asyncio.Lock,
    config: WSConfig,
) -> None:
    output_format = str(message.get("output_format") or "pcm").lower()
    if output_format not in {"pcm", "wav"}:
        await _send_error(websocket, f"Unsupported output_format '{output_format}'.", "UNSUPPORTED_FORMAT")
        return

    for sentence in sentences:
        gen_kwargs = {
            "text": sentence,
            "num_step": int(message.get("num_step") or config.default_num_step),
        }
        if message.get("speed") is not None:
            gen_kwargs["speed"] = float(message["speed"])
        if state.voice_clone_prompt is not None:
            gen_kwargs["voice_clone_prompt"] = state.voice_clone_prompt
        elif message.get("instruct"):
            gen_kwargs["instruct"] = str(message["instruct"])

        started = time.monotonic()
        audio = await _generate_one(get_model, model_lock, gen_kwargs)
        generation_ms = int((time.monotonic() - started) * 1000)
        duration_ms = int(audio.shape[-1] * 1000 / config.sample_rate)

        payload = encode_audio_chunk(audio, output_format=output_format, sample_rate=config.sample_rate)
        is_last = sentence == sentences[-1]
        await websocket.send_json(
            {
                "type": "audio_chunk",
                "chunk_index": state.chunk_index,
                "sentence": sentence,
                "duration_ms": duration_ms,
                "generation_time_ms": generation_ms,
                "is_last": is_last,
            }
        )
        await websocket.send_bytes(payload)

        state.chunk_index += 1
        state.total_duration_ms += duration_ms
        state.total_generation_time_ms += generation_ms


async def _generate_one(get_model: Callable[[], Any], model_lock: asyncio.Lock, gen_kwargs: dict[str, Any]):
    model = get_model()
    if model is None:
        raise RuntimeError("Model not loaded yet.")

    async with model_lock:
        audio_tensors = await asyncio.to_thread(model.generate, **gen_kwargs)
    if not audio_tensors:
        raise RuntimeError("Model returned empty audio.")
    return audio_tensors[0]


async def _ensure_voice_prompt(
    message: dict[str, Any],
    state: WSSessionState,
    get_model: Callable[[], Any],
    get_voice_samples: Callable[[], dict[str, dict[str, Any]]],
    model_lock: asyncio.Lock,
    force: bool = False,
) -> None:
    sample = message.get("sample")
    if not sample:
        if force:
            raise RuntimeError("Field 'sample' is required.")
        return

    sample_name = str(sample)
    if not force and state.prompt_sample == sample_name and state.voice_clone_prompt is not None:
        return

    samples = get_voice_samples()
    if sample_name not in samples:
        raise RuntimeError(f"Sample '{sample_name}' not found.")

    sample_info = samples[sample_name]
    ref_text = message.get("ref_text") if message.get("ref_text") is not None else sample_info.get("ref_text")

    model = get_model()
    if model is None:
        raise RuntimeError("Model not loaded yet.")

    kwargs = {"ref_audio": sample_info["audio_path"]}
    if ref_text is not None:
        kwargs["ref_text"] = ref_text

    async with model_lock:
        state.voice_clone_prompt = await asyncio.to_thread(model.create_voice_clone_prompt, **kwargs)
    state.prompt_sample = sample_name
    state.prompt_text = ref_text


async def _recv_json(websocket: WebSocket, timeout_s: float, idle_timeout_s: int) -> Optional[dict[str, Any]]:
    try:
        payload = await asyncio.wait_for(websocket.receive_json(), timeout=timeout_s)
    except asyncio.TimeoutError:
        if timeout_s < idle_timeout_s:
            return {"type": "__buffer_timeout__"}
        await websocket.send_json({"type": "error", "message": "Connection idle timeout.", "code": "IDLE_TIMEOUT"})
        await websocket.close(code=1001)
        return None
    return payload if isinstance(payload, dict) else {"type": None}


async def _send_error(websocket: WebSocket, message: str, code: str) -> None:
    await websocket.send_json({"type": "error", "message": message, "code": code})


async def _send_done(websocket: WebSocket, state: WSSessionState) -> None:
    await websocket.send_json(
        {
            "type": "done",
            "total_chunks": state.chunk_index,
            "total_duration_ms": state.total_duration_ms,
            "total_generation_time_ms": state.total_generation_time_ms,
        }
    )
