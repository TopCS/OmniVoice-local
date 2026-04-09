from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections import deque
from dataclasses import dataclass


UPLINK_SAMPLE_RATE = 16000
PLAYBACK_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
UPLINK_CHUNK_SIZE = 512
PLAYBACK_FRAME_BYTES = 960


def build_session_start(
    *, sample: str | None = None, sample_rate: int = UPLINK_SAMPLE_RATE
) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "session_start",
        "sample_rate": sample_rate,
    }
    if sample is not None:
        event["sample"] = sample
    return event


@dataclass
class DuplexState:
    session_started: bool = False
    sample: str | None = None
    sample_rate: int = UPLINK_SAMPLE_RATE
    server_listening: bool = False
    local_speech_active: bool = False
    last_response_id: int | None = None
    active_response_id: int | None = None
    pending_audio_response_id: int | None = None

    def start_local_speech(self, playback) -> dict[str, object] | None:
        if self.local_speech_active:
            return None

        self.local_speech_active = True
        self.server_listening = False
        self.active_response_id = None
        playback.interrupt()
        return {"type": "speech_start"}

    def end_local_speech(self) -> dict[str, object] | None:
        if not self.local_speech_active:
            return None

        self.local_speech_active = False
        return {"type": "speech_end"}

    def handle_event(
        self, event: dict[str, object], playback
    ) -> dict[str, object] | None:
        del playback

        event_type = event.get("type")
        if event_type == "audio_chunk":
            if self.pending_audio_response_id is not None:
                raise RuntimeError("Received audio_chunk metadata before audio bytes")
        elif self.pending_audio_response_id is not None:
            if (
                event_type == "interrupted"
                and event.get("response_id") == self.pending_audio_response_id
            ):
                self.pending_audio_response_id = None
            else:
                event_name = event_type if isinstance(event_type, str) else "event"
                raise RuntimeError(f"Received {event_name} before audio bytes")

        if event_type == "session_started":
            self.session_started = True
            self.sample = str(event["sample"]) if event.get("sample") else None
            self.sample_rate = int(event.get("sample_rate", UPLINK_SAMPLE_RATE))
            return event

        if event_type == "listening":
            self.server_listening = True
            return event

        if event_type == "transcript_final":
            self.server_listening = False
            return event

        if event_type in {
            "assistant_text",
            "audio_chunk",
            "response_done",
            "interrupted",
        }:
            response_id = event.get("response_id")
            if not isinstance(response_id, int):
                return None
            if not self._accept_response_event(event_type, response_id):
                if event_type == "audio_chunk":
                    self.pending_audio_response_id = None
                return None

            if event_type == "audio_chunk":
                self.pending_audio_response_id = response_id
            elif event_type in {"response_done", "interrupted"}:
                self.pending_audio_response_id = None
                self.active_response_id = None
            return event

        return event

    def handle_audio_bytes(self, audio: bytes, playback) -> bool:
        response_id = self.pending_audio_response_id
        self.pending_audio_response_id = None
        if response_id is None:
            raise RuntimeError("Received binary frame without audio_chunk metadata")
        if self.local_speech_active or self.active_response_id != response_id:
            return False

        playback.enqueue(audio)
        return True

    def _accept_response_event(self, event_type: str, response_id: int) -> bool:
        if self.last_response_id is None or response_id > self.last_response_id:
            self.last_response_id = response_id
            if event_type != "interrupted":
                self.active_response_id = response_id
            return True

        if response_id < self.last_response_id:
            return False

        if event_type == "interrupted":
            return True

        return self.active_response_id == response_id


class PlaybackController:
    def __init__(self, stream):
        self.stream = stream
        self._generation = 0
        self._closed = False
        self._pending: deque[tuple[int, bytes]] = deque()
        self._signals: asyncio.Queue[None] = asyncio.Queue()

    def enqueue(self, audio: bytes) -> None:
        if self._closed:
            return
        generation = self._generation
        for offset in range(0, len(audio), PLAYBACK_FRAME_BYTES):
            self._pending.append(
                (generation, audio[offset : offset + PLAYBACK_FRAME_BYTES])
            )
        self._signal()

    def interrupt(self) -> None:
        if self._closed:
            return
        self._generation += 1
        self._pending.clear()
        self._signal()

    async def run(self) -> None:
        while True:
            if not self._pending:
                if self._closed:
                    return
                await self._signals.get()
                continue

            generation, audio = self._pending.popleft()
            if generation != self._generation:
                continue
            await asyncio.to_thread(self.stream.write, audio)
            if self._closed:
                return

    async def close(self) -> None:
        self._closed = True
        self._generation += 1
        self._pending.clear()
        self._signal()

    def _signal(self) -> None:
        if self._signals.empty():
            self._signals.put_nowait(None)


def open_audio_streams(pyaudio_module):
    audio_api = pyaudio_module.PyAudio()
    input_stream = audio_api.open(
        format=pyaudio_module.paInt16,
        channels=AUDIO_CHANNELS,
        rate=UPLINK_SAMPLE_RATE,
        input=True,
        frames_per_buffer=UPLINK_CHUNK_SIZE,
    )
    output_stream = audio_api.open(
        format=pyaudio_module.paInt16,
        channels=AUDIO_CHANNELS,
        rate=PLAYBACK_SAMPLE_RATE,
        output=True,
    )
    return audio_api, input_stream, output_stream


def pcm16_bytes_to_float_tensor(audio: bytes, torch_module):
    audio_int16 = torch_module.frombuffer(audio, dtype=torch_module.int16)
    return audio_int16.to(dtype=torch_module.float32) / 32768.0


async def microphone_loop(
    websocket,
    state: DuplexState,
    input_stream,
    playback: PlaybackController,
    torch_module,
    silero_vad_module,
) -> None:
    model = silero_vad_module.load_silero_vad()
    vad = silero_vad_module.VADIterator(model, sampling_rate=UPLINK_SAMPLE_RATE)

    try:
        while True:
            chunk = await asyncio.to_thread(
                input_stream.read,
                UPLINK_CHUNK_SIZE,
                exception_on_overflow=False,
            )
            speech_event = (
                vad(
                    pcm16_bytes_to_float_tensor(chunk, torch_module),
                    return_seconds=True,
                )
                or {}
            )

            if "start" in speech_event:
                control = state.start_local_speech(playback)
                if control is not None:
                    await websocket.send(json.dumps(control))

            if state.local_speech_active:
                await websocket.send(chunk)

            if "end" in speech_event:
                control = state.end_local_speech()
                if control is not None:
                    await websocket.send(json.dumps(control))
    finally:
        vad.reset_states()


async def receive_loop(
    websocket,
    state: DuplexState,
    playback: PlaybackController,
    *,
    event_sink=None,
) -> None:
    if event_sink is None:
        event_sink = lambda event: print(json.dumps(event), flush=True)

    while True:
        message = await websocket.recv()
        if isinstance(message, bytes):
            state.handle_audio_bytes(message, playback)
            continue

        event = json.loads(message)
        accepted = state.handle_event(event, playback)
        if accepted is None:
            continue

        event_sink(accepted)
        if accepted.get("type") == "error":
            raise RuntimeError(accepted.get("message") or "Server returned an error")


async def run_client(
    url: str,
    sample: str | None,
    websockets_module,
    pyaudio_module,
    torch_module,
    silero_vad_module,
) -> None:
    async with websockets_module.connect(url) as websocket:
        await websocket.send(json.dumps(build_session_start(sample=sample)))

        audio_api = None
        input_stream = None
        output_stream = None
        playback = None
        playback_task = None
        receiver_task = None
        microphone_task = None
        try:
            audio_api, input_stream, output_stream = open_audio_streams(pyaudio_module)
            playback = PlaybackController(output_stream)
            state = DuplexState(sample=sample)

            playback_task = asyncio.create_task(playback.run())
            receiver_task = asyncio.create_task(
                receive_loop(websocket, state, playback)
            )
            microphone_task = asyncio.create_task(
                microphone_loop(
                    websocket,
                    state,
                    input_stream,
                    playback,
                    torch_module,
                    silero_vad_module,
                )
            )
            await asyncio.gather(receiver_task, microphone_task)
        finally:
            for task in (receiver_task, microphone_task):
                if task is not None:
                    task.cancel()
            for task in (receiver_task, microphone_task):
                if task is not None:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

            if playback is not None:
                await playback.close()
            if playback_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await playback_task

            for stream in (input_stream, output_stream):
                if stream is None:
                    continue
                stop_stream = getattr(stream, "stop_stream", None)
                if callable(stop_stream):
                    stop_stream()
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
            if audio_api is not None:
                audio_api.terminate()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream duplex conversation audio with local Silero VAD."
    )
    parser.add_argument("--url", default="ws://localhost:8000/ws/conversation")
    parser.add_argument("--sample")
    args = parser.parse_args(argv)

    try:
        import pyaudio
    except ModuleNotFoundError:
        print(
            "Missing dependency: install pyaudio to enable duplex audio.",
            file=sys.stderr,
        )
        return 1

    try:
        import silero_vad
    except ModuleNotFoundError:
        print(
            "Missing dependency: install silero-vad to enable local VAD.",
            file=sys.stderr,
        )
        return 1

    try:
        import torch
    except ModuleNotFoundError:
        print("Missing dependency: install torch to run Silero VAD.", file=sys.stderr)
        return 1

    try:
        import websockets
    except ModuleNotFoundError:
        print(
            "Missing dependency: install websockets to connect to the server.",
            file=sys.stderr,
        )
        return 1

    try:
        asyncio.run(
            run_client(args.url, args.sample, websockets, pyaudio, torch, silero_vad)
        )
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
