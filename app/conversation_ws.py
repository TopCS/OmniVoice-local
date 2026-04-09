"""Minimal conversation WebSocket protocol skeleton."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

SUPPORTED_SAMPLE_RATE = 16000
DEFAULT_MAX_INPUT_BYTES = 1024 * 1024


@dataclass
class ConversationSessionState:
    started: bool = False
    sample: str | None = None
    sample_rate: int = SUPPORTED_SAMPLE_RATE
    collecting: bool = False
    audio_chunks: list[bytes] = field(default_factory=list)
    buffered_audio_bytes: int = 0
    next_response_id: int = 1
    active_response_id: int | None = None
    response_task: asyncio.Task[None] | None = None
    cancelled_response_ids: set[int] = field(default_factory=set)
    background_response_tasks: set[asyncio.Task[None]] = field(default_factory=set)


def create_conversation_router(
    *,
    service: Any,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    active_counter: Callable[[int], None] | None = None,
) -> APIRouter:
    """Create router exposing the `/ws/conversation` skeleton endpoint."""

    router = APIRouter()
    active_counter = active_counter or (lambda delta: None)

    @router.websocket("/ws/conversation")
    async def ws_conversation(websocket: WebSocket) -> None:
        await websocket.accept()
        state = ConversationSessionState()
        active_counter(1)

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                await _handle_incoming(
                    websocket,
                    message,
                    state,
                    service,
                    max_input_bytes=max_input_bytes,
                )
        except WebSocketDisconnect:
            pass
        finally:
            active_counter(-1)
            await _cancel_active_response(state)
            if websocket.application_state != WebSocketState.DISCONNECTED:
                await websocket.close()

    return router


async def _handle_incoming(
    websocket: WebSocket,
    message: dict[str, Any],
    state: ConversationSessionState,
    service: Any,
    *,
    max_input_bytes: int,
) -> None:
    audio = message.get("bytes")
    if audio is not None:
        await _handle_audio_frame(
            websocket, audio, state, max_input_bytes=max_input_bytes
        )
        return

    raw_text = message.get("text")
    if raw_text is None:
        await _send_error(
            websocket, "Unsupported websocket payload.", "INVALID_MESSAGE"
        )
        return

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        await _send_error(websocket, "Message must be valid JSON.", "INVALID_MESSAGE")
        return

    if not isinstance(payload, dict):
        await _send_error(
            websocket, "Message must be a JSON object.", "INVALID_MESSAGE"
        )
        return

    await _handle_event(
        websocket, payload, state, service, max_input_bytes=max_input_bytes
    )


async def _handle_event(
    websocket: WebSocket,
    message: dict[str, Any],
    state: ConversationSessionState,
    service: Any,
    *,
    max_input_bytes: int,
) -> None:
    msg_type = message.get("type")
    if msg_type == "session_start":
        sample = str(message["sample"]) if message.get("sample") else None
        if message.get("sample_rate") is not None:
            try:
                sample_rate = int(message["sample_rate"])
            except (TypeError, ValueError):
                await _send_error(
                    websocket,
                    "Field 'sample_rate' must be an integer.",
                    "INVALID_MESSAGE",
                )
                return
            if sample_rate <= 0:
                await _send_error(
                    websocket,
                    "Field 'sample_rate' must be a positive integer.",
                    "INVALID_MESSAGE",
                )
                return
            if sample_rate != SUPPORTED_SAMPLE_RATE:
                await _send_error(
                    websocket,
                    f"Field 'sample_rate' must be {SUPPORTED_SAMPLE_RATE} for conversation ASR.",
                    "INVALID_MESSAGE",
                )
                return
        else:
            sample_rate = state.sample_rate

        try:
            await _validate_sample(service, sample)
        except Exception as exc:  # noqa: BLE001
            await _send_error(websocket, str(exc), "INVALID_MESSAGE")
            return

        state.started = True
        state.sample = sample
        state.sample_rate = sample_rate
        await websocket.send_json(
            {
                "type": "session_started",
                "sample": state.sample,
                "sample_rate": state.sample_rate,
            }
        )
        return

    if not state.started:
        await _send_error(
            websocket,
            "Send session_start before conversation events.",
            "INVALID_MESSAGE",
        )
        return

    if msg_type == "speech_start":
        if state.collecting:
            await _send_error(
                websocket,
                "Received speech_start while already collecting audio.",
                "INVALID_MESSAGE",
            )
            return

        await _interrupt_active_response(websocket, state)
        state.collecting = True
        state.audio_chunks.clear()
        state.buffered_audio_bytes = 0
        await websocket.send_json({"type": "listening"})
        return

    if msg_type == "speech_end":
        if not state.collecting:
            await _send_error(
                websocket,
                "Received speech_end without speech_start.",
                "INVALID_MESSAGE",
            )
            return

        state.collecting = False
        audio_bytes = b"".join(state.audio_chunks)
        state.audio_chunks.clear()
        state.buffered_audio_bytes = 0

        try:
            transcript = await service.transcribe(audio_bytes, state.sample_rate)
        except Exception as exc:  # noqa: BLE001
            await _send_error(websocket, str(exc), "TRANSCRIBE_ERROR")
            return

        if isinstance(transcript, dict):
            await websocket.send_json(transcript)
        else:
            await websocket.send_json({"type": "transcript_final", "text": transcript})

        response_id = state.next_response_id
        state.next_response_id += 1
        state.active_response_id = response_id
        state.response_task = asyncio.create_task(
            _forward_service_response(
                websocket, state, service, transcript, response_id
            )
        )
        state.background_response_tasks.add(state.response_task)
        state.response_task.add_done_callback(
            lambda task: _cleanup_response_task(state, task)
        )
        return

    if msg_type == "input_audio_chunk":
        encoded_audio = message.get("audio")
        if not isinstance(encoded_audio, str):
            await _send_error(
                websocket,
                "Field 'audio' must be a base64 string.",
                "INVALID_MESSAGE",
            )
            return

        try:
            audio = base64.b64decode(encoded_audio.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error):
            await _send_error(
                websocket,
                "Field 'audio' must be valid base64.",
                "INVALID_MESSAGE",
            )
            return

        await _handle_audio_frame(
            websocket, audio, state, max_input_bytes=max_input_bytes
        )
        return

    if msg_type is None:
        await _send_error(
            websocket, "Use binary frames for audio input.", "INVALID_MESSAGE"
        )
        return

    await _send_error(
        websocket,
        f"Unsupported message type '{msg_type}'.",
        "INVALID_MESSAGE",
    )


async def _handle_audio_frame(
    websocket: WebSocket,
    audio: bytes,
    state: ConversationSessionState,
    *,
    max_input_bytes: int,
) -> None:
    if not state.collecting:
        await _send_error(
            websocket,
            "Audio received outside active speech input.",
            "UNEXPECTED_AUDIO",
        )
        return

    next_size = state.buffered_audio_bytes + len(audio)
    if next_size > max_input_bytes:
        state.collecting = False
        state.audio_chunks.clear()
        state.buffered_audio_bytes = 0
        await _send_error(
            websocket,
            "Input audio exceeded the maximum buffered size.",
            "INPUT_AUDIO_TOO_LARGE",
        )
        return

    state.audio_chunks.append(audio)
    state.buffered_audio_bytes = next_size


async def _forward_service_response(
    websocket: WebSocket,
    state: ConversationSessionState,
    service: Any,
    transcript: Any,
    response_id: int,
) -> None:
    try:
        async for event in _iterate_response_events(
            _call_service_respond(service, transcript, response_id, state.sample)
        ):
            if _response_is_stale(state, response_id):
                break
            payload = _normalize_response_event(event, response_id)
            binary = payload.pop("payload", None)
            await websocket.send_json(payload)
            if binary is not None:
                await websocket.send_bytes(binary)
            if _response_is_stale(state, response_id):
                break
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        if (
            not _response_is_stale(state, response_id)
            and websocket.application_state != WebSocketState.DISCONNECTED
        ):
            with contextlib.suppress(Exception):
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": str(exc),
                        "code": "RESPONSE_ERROR",
                        "response_id": response_id,
                    }
                )
    finally:
        state.cancelled_response_ids.discard(response_id)
        if state.active_response_id == response_id:
            state.active_response_id = None
            state.response_task = None


async def _iterate_response_events(result: Any):
    if hasattr(result, "__aiter__"):
        async for item in result:
            yield item
        return

    if asyncio.iscoroutine(result):
        result = await result

    for item in result:
        yield item


async def _interrupt_active_response(
    websocket: WebSocket, state: ConversationSessionState
) -> None:
    if state.response_task is None or state.response_task.done():
        return

    response_id = state.active_response_id
    task = state.response_task
    if response_id is not None:
        state.cancelled_response_ids.add(response_id)
    state.active_response_id = None
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    if state.response_task is task:
        state.response_task = None

    if response_id is not None:
        await websocket.send_json({"type": "interrupted", "response_id": response_id})


async def _cancel_active_response(state: ConversationSessionState) -> None:
    tasks = [task for task in state.background_response_tasks if not task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks)


def _response_is_stale(state: ConversationSessionState, response_id: int) -> bool:
    return (
        response_id in state.cancelled_response_ids
        or state.active_response_id != response_id
    )


def _cleanup_response_task(
    state: ConversationSessionState, task: asyncio.Task[None]
) -> None:
    state.background_response_tasks.discard(task)
    with contextlib.suppress(asyncio.CancelledError):
        task.exception()


def _call_service_respond(
    service: Any, transcript: Any, response_id: int, sample: str | None
):
    respond = service.respond
    parameters = inspect.signature(respond).parameters
    if "sample" in parameters:
        return respond(transcript, response_id, sample=sample)
    return respond(transcript, response_id)


async def _validate_sample(service: Any, sample: str | None) -> None:
    validate = getattr(service, "validate_sample", None)
    if validate is None:
        return

    result = validate(sample)
    if asyncio.iscoroutine(result):
        await result


def _normalize_response_event(event: Any, response_id: int) -> dict[str, Any]:
    payload = dict(event)
    if payload.get("type") in {"assistant_text", "audio_chunk", "response_done"}:
        payload["response_id"] = response_id
    else:
        payload.setdefault("response_id", response_id)
    return payload


async def _send_error(websocket: WebSocket, message: str, code: str) -> None:
    await websocket.send_json({"type": "error", "message": message, "code": code})
