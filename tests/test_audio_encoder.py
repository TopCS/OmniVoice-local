from pathlib import Path
import sys
import struct
import wave

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from audio_encoder import encode_audio_chunk, tensor_to_pcm16


def test_tensor_to_pcm16_length_and_type():
    audio = torch.tensor([[0.0, 0.5, -0.5, 1.0, -1.0]], dtype=torch.float32)
    pcm = tensor_to_pcm16(audio)
    assert isinstance(pcm, bytes)
    assert len(pcm) == 10


def test_encode_pcm_matches_expected_samples():
    audio = torch.tensor([[0.0, 1.0, -1.0]], dtype=torch.float32)
    pcm = encode_audio_chunk(audio, output_format="pcm")
    values = struct.unpack("<hhh", pcm)
    assert list(values) == [0, 32767, -32767]


def test_encode_wav_is_valid_chunk_file(tmp_path):
    audio = torch.zeros((1, 2400), dtype=torch.float32)
    wav_bytes = encode_audio_chunk(audio, output_format="wav", sample_rate=24000)
    wav_path = tmp_path / "chunk.wav"
    wav_path.write_bytes(wav_bytes)

    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 24000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 2400
