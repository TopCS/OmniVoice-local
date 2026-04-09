"""Thin ASR adapter around faster-whisper."""

from __future__ import annotations

from typing import Any

import numpy as np


def _resolve_device(device: str) -> dict[str, Any]:
    if device.startswith("cuda:"):
        _, _, index = device.partition(":")
        if index.isdigit():
            return {"device": "cuda", "device_index": int(index)}
    return {"device": device}


class FasterWhisperASR:
    """Transcribe mono 16 kHz PCM audio with faster-whisper."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "default",
    ) -> None:
        from faster_whisper import WhisperModel

        model_kwargs = {
            **_resolve_device(device),
            "compute_type": compute_type,
        }
        self._model = WhisperModel(model_name, **model_kwargs)

    def transcribe(self, pcm_bytes: bytes) -> str:
        if not pcm_bytes:
            return ""
        if len(pcm_bytes) % 2 != 0:
            raise ValueError("PCM audio must contain 16-bit samples.")

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        audio /= 32768.0

        segments, _info = self._model.transcribe(audio, beam_size=1)
        return "".join(segment.text for segment in segments).strip()
