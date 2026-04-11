# Conversation Assistant Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `/ws/conversation` so the assistant behaves like a concise phone agent, keeps short-term memory, uses sticky ASR language bias, sanitizes output for TTS, and logs per-turn latency.

**Architecture:** Introduce a small assistant-backend abstraction around the current Ollama integration, keep three turns of conversational memory in websocket session state, add a strong system prompt plus sanitization layer before TTS, and emit structured timing logs for ASR, LLM, and TTS. Keep Ollama as the default provider and preserve the existing duplex protocol and response-scoped streaming semantics.

**Tech Stack:** FastAPI WebSocket, asyncio, Ollama async client, faster-whisper, pytest, OmniVoice TTS streaming, Python logging

---

## File Structure

- Create: `app/assistant_backends.py`
  Provider-agnostic assistant backend interface plus the default Ollama implementation and prompt helpers.

- Create: `app/text_sanitizer.py`
  Pure text sanitization helpers that strip markdown/JSON-like formatting and normalize text for TTS.

- Modify: `app/server.py`
  Extend `ConversationService` with memory, system prompt composition, sticky ASR language bias, assistant backend wiring, and structured latency logs.

- Modify: `app/conversation_ws.py`
  Extend session state and `session_start` handling for language override and per-session conversation identifiers.

- Modify: `app/asr.py`
  Allow `language=` hints to be passed to `faster-whisper` while preserving current defaults.

- Modify: `tests/test_conversation_ws.py`
  Add regression coverage for memory, prompt inclusion, sticky language bias, sanitization, and logging.

- Modify: `README.md`
  Document the new assistant-governance behavior and new environment variables.

### Task 1: Add Assistant Backend And Phone-Agent Prompt Policy

**Files:**
- Create: `app/assistant_backends.py`
- Modify: `app/server.py`
- Modify: `tests/test_conversation_ws.py`
- Test: `tests/test_conversation_ws.py`

- [ ] **Step 1: Write the failing tests for provider abstraction and system prompt injection**

```python
@pytest.mark.asyncio
async def test_server_conversation_service_includes_phone_agent_system_prompt(monkeypatch):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes) -> str:
            raise AssertionError("transcribe should not be called")

    class FakeAssistantBackend:
        def __init__(self):
            self.calls = []

        async def generate(self, *, messages, model, session_id):
            self.calls.append(
                {
                    "messages": messages,
                    "model": model,
                    "session_id": session_id,
                }
            )
            return "Certo, dimmi pure."

    class FakeAudio:
        shape = (1, 2400)

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    server.model = types.SimpleNamespace(create_voice_clone_prompt=lambda **kwargs: kwargs)
    server.voice_samples = {}
    monkeypatch.setattr(server, "_generate_one", lambda get_model, model_lock, gen_kwargs: FakeAudio())
    monkeypatch.setattr(server, "encode_audio_chunk", lambda audio, output_format, sample_rate=24000: b"pcm")

    backend = FakeAssistantBackend()
    service = server.ConversationService(
        FakeASR(),
        assistant_backend=backend,
        assistant_model="gemma4",
    )

    events = [event async for event in service.respond("Vorrei un aiuto.", 1)]

    assert events[0] == {"type": "assistant_text", "text": "Certo, dimmi pure.", "response_id": 1}
    messages = backend.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "phone-like spoken conversation" in messages[0]["content"]
    assert "plain text only" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Vorrei un aiuto."}
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest tests/test_conversation_ws.py::test_server_conversation_service_includes_phone_agent_system_prompt -v`
Expected: FAIL because `ConversationService` does not yet accept an assistant backend or prepend a system prompt.

- [ ] **Step 3: Add the minimal assistant backend abstraction and prompt builder**

```python
# app/assistant_backends.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


PHONE_AGENT_SYSTEM_PROMPT = (
    "You are a voice agent for a phone-like spoken conversation. "
    "Your tone is calm, direct, and reassuring. "
    "Reply in plain text only. Do not use markdown, JSON, bullet lists, emojis, code blocks, or labels like Answer: or Response:. "
    "Keep replies short and natural for text-to-speech. "
    "If you need clarification, ask for it in one short sentence."
)


class AssistantBackend(Protocol):
    async def generate(
        self,
        *,
        messages: Sequence[dict[str, str]],
        model: str,
        session_id: str,
    ) -> str: ...


@dataclass
class OllamaAssistantBackend:
    client: object

    async def generate(self, *, messages, model: str, session_id: str) -> str:
        response = await self.client.chat(model=model, messages=list(messages), stream=False)
        if isinstance(response, dict):
            return str(response.get("message", {}).get("content") or "")
        message = getattr(response, "message", None)
        return str(getattr(message, "content", "") or "")
```

```python
# app/server.py key changes
from assistant_backends import AssistantBackend, OllamaAssistantBackend, PHONE_AGENT_SYSTEM_PROMPT


class ConversationService:
    def __init__(self, asr_adapter: Any, *, assistant_backend: AssistantBackend, assistant_model: str):
        self._asr_adapter = asr_adapter
        self._assistant_backend = assistant_backend
        self._assistant_model = assistant_model
        ...

    async def _generate_assistant_text(self, transcript: str, *, session_id: str) -> str:
        messages = [
            {"role": "system", "content": PHONE_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        return await self._assistant_backend.generate(
            messages=messages,
            model=self._assistant_model,
            session_id=session_id,
        )
```

```python
# app/server.py in _create_conversation_service
from ollama import AsyncClient

return ConversationService(
    FasterWhisperASR(model_name=ASR_MODEL, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE),
    assistant_backend=OllamaAssistantBackend(AsyncClient(host=OLLAMA_HOST)),
    assistant_model=OLLAMA_MODEL,
)
```

- [ ] **Step 4: Run the conversation tests again**

Run: `pytest tests/test_conversation_ws.py -v`
Expected: PASS with the new prompt-injection test and all existing duplex tests green.

- [ ] **Step 5: Commit**

```bash
git add app/assistant_backends.py app/server.py tests/test_conversation_ws.py
git commit -m "feat: add assistant backend prompt policy"
```

### Task 2: Add Short-Term Memory And Sticky ASR Language Bias

**Files:**
- Modify: `app/conversation_ws.py`
- Modify: `app/server.py`
- Modify: `app/asr.py`
- Modify: `tests/test_conversation_ws.py`
- Test: `tests/test_conversation_ws.py`

- [ ] **Step 1: Write the failing tests for memory trimming and language bias**

```python
@pytest.mark.asyncio
async def test_server_conversation_service_trims_history_to_last_three_turns(monkeypatch):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes, *, language: str | None = None):
            raise AssertionError("transcribe should not be called")

    class FakeAssistantBackend:
        def __init__(self):
            self.messages = None

        async def generate(self, *, messages, model, session_id):
            self.messages = messages
            return "Va bene."

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    backend = FakeAssistantBackend()
    service = server.ConversationService(FakeASR(), assistant_backend=backend, assistant_model="gemma4")
    history = [
        {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4"},
    ]

    await service._generate_assistant_text("u5", session_id="session-1", history=history)

    assert backend.messages[-7:] == [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "a4"},
        {"role": "user", "content": "u5"},
    ]


def test_faster_whisper_asr_adapter_passes_language_hint(monkeypatch):
    seen = {}

    class FakeWhisperModel:
        def __init__(self, model_name, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)
            return iter([]), types.SimpleNamespace(language="it")

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    asr = _load_app_module("asr_under_test_language", "asr.py")
    adapter = asr.FasterWhisperASR(model_name="small")

    adapter.transcribe(np.array([0, 1], dtype=np.int16).tobytes(), language="it")

    assert seen["language"] == "it"
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_conversation_ws.py::test_server_conversation_service_trims_history_to_last_three_turns tests/test_conversation_ws.py::test_faster_whisper_asr_adapter_passes_language_hint -v`
Expected: FAIL because history and language hints are not yet implemented.

- [ ] **Step 3: Implement short-term memory and sticky language bias**

```python
# app/asr.py key changes
class FasterWhisperASR:
    ...
    def transcribe(self, pcm_bytes: bytes, *, language: str | None = None) -> tuple[str, str | None]:
        ...
        kwargs = {"beam_size": 1}
        if language:
            kwargs["language"] = language
        segments, info = self._model.transcribe(audio, **kwargs)
        transcript = "".join(segment.text for segment in segments).strip()
        detected_language = getattr(info, "language", None)
        return transcript, detected_language
```

```python
# app/conversation_ws.py key changes
@dataclass
class ConversationSessionState:
    ...
    session_id: str = field(default_factory=lambda: str(__import__("uuid").uuid4()))
    language_hint: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)
```

```python
# app/server.py key changes
class ConversationService:
    async def transcribe(self, audio_bytes: bytes, sample_rate: int, *, language_hint: str | None = None) -> tuple[str, str | None]:
        if sample_rate != 16000:
            raise RuntimeError("Conversation ASR currently expects mono 16 kHz PCM audio.")
        return await asyncio.to_thread(self._asr_adapter.transcribe, audio_bytes, language=language_hint)

    async def _generate_assistant_text(self, transcript: str, *, session_id: str, history: list[dict[str, str]]) -> str:
        trimmed_history = history[-6:]
        messages = [{"role": "system", "content": PHONE_AGENT_SYSTEM_PROMPT}, *trimmed_history, {"role": "user", "content": transcript}]
        return await self._assistant_backend.generate(messages=messages, model=self._assistant_model, session_id=session_id)
```

In `app.conversation_ws`, after `speech_end`, store the detected language back into `state.language_hint` unless an explicit override is present, and append completed user/assistant turns to `state.history`, trimming to the last three turns.

- [ ] **Step 4: Run the full conversation test file**

Run: `pytest tests/test_conversation_ws.py -v`
Expected: PASS with memory trimming and sticky language bias covered.

- [ ] **Step 5: Commit**

```bash
git add app/asr.py app/conversation_ws.py app/server.py tests/test_conversation_ws.py
git commit -m "feat: add memory and sticky asr language bias"
```

### Task 3: Sanitize Assistant Output For TTS

**Files:**
- Create: `app/text_sanitizer.py`
- Modify: `app/server.py`
- Modify: `tests/test_conversation_ws.py`
- Test: `tests/test_conversation_ws.py`

- [ ] **Step 1: Write the failing sanitization tests**

```python
def test_sanitize_assistant_text_removes_markdown_json_and_noise():
    sanitizer = _load_app_module("text_sanitizer_under_test", "text_sanitizer.py")

    cleaned = sanitizer.sanitize_assistant_text(
        """
        ## Risposta
        {"answer": "Certo!"}
        - punto uno
        - punto due
        ```json
        {"k": 1}
        ```
        """
    )

    assert cleaned == "Certo! punto uno punto due"
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest tests/test_conversation_ws.py::test_sanitize_assistant_text_removes_markdown_json_and_noise -v`
Expected: FAIL because the sanitizer module does not exist yet.

- [ ] **Step 3: Add the minimal sanitizer and use it before sentence splitting**

```python
# app/text_sanitizer.py
from __future__ import annotations

import json
import re


def sanitize_assistant_text(text: str) -> str:
    value = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    value = re.sub(r"^#+\s*", "", value, flags=re.MULTILINE)
    value = re.sub(r"^[-*+]\s+", "", value, flags=re.MULTILINE)
    value = value.strip()

    if value.startswith("{") and value.endswith("}"):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                for key in ("answer", "response", "text", "content"):
                    if isinstance(parsed.get(key), str):
                        value = parsed[key]
                        break
        except json.JSONDecodeError:
            pass

    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"\s{2,}", " ", value)
    value = re.sub(r"[*_`#]", "", value)
    return value.strip()
```

```python
# app/server.py key change
from text_sanitizer import sanitize_assistant_text

...
response_text = sanitize_assistant_text((await self._generate_assistant_text(...)).strip())
```

- [ ] **Step 4: Run the sanitization and conversation tests**

Run: `pytest tests/test_conversation_ws.py -v`
Expected: PASS with sanitization coverage added and existing conversation streaming still green.

- [ ] **Step 5: Commit**

```bash
git add app/text_sanitizer.py app/server.py tests/test_conversation_ws.py
git commit -m "feat: sanitize assistant text for tts"
```

### Task 4: Add Structured Latency Logging And README Updates

**Files:**
- Modify: `app/server.py`
- Modify: `README.md`
- Modify: `tests/test_conversation_ws.py`
- Test: `tests/test_conversation_ws.py`

- [ ] **Step 1: Write the failing test for per-turn latency logging**

```python
@pytest.mark.asyncio
async def test_server_conversation_logs_turn_latency_fields(monkeypatch, caplog):
    class FakeASR:
        def transcribe(self, pcm_bytes: bytes, *, language: str | None = None):
            return "ciao", "it"

    class FakeAssistantBackend:
        async def generate(self, *, messages, model, session_id):
            return "Va bene."

    class FakeAudio:
        shape = (1, 2400)

    server = _load_server_module(monkeypatch, conversation_enabled=True)
    server.model = types.SimpleNamespace(create_voice_clone_prompt=lambda **kwargs: kwargs)
    monkeypatch.setattr(server, "_generate_one", lambda get_model, model_lock, gen_kwargs: FakeAudio())
    monkeypatch.setattr(server, "encode_audio_chunk", lambda audio, output_format, sample_rate=24000: b"pcm")

    service = server.ConversationService(FakeASR(), assistant_backend=FakeAssistantBackend(), assistant_model="gemma4")
    list([event async for event in service.respond("ciao", 1, session_id="session-1", history=[], language_hint="it")])

    assert any(
        "asr_ms=" in record.message and "llm_ms=" in record.message and "tts_total_ms=" in record.message
        for record in caplog.records
    )
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest tests/test_conversation_ws.py::test_server_conversation_logs_turn_latency_fields -v`
Expected: FAIL because the conversation service does not log per-turn timings yet.

- [ ] **Step 3: Add structured latency logging and README documentation**

```python
# app/server.py key change sketch
log.info(
    "conversation_turn session_id=%s response_id=%s detected_language=%s language_hint=%s assistant_backend=%s assistant_model=%s asr_ms=%d llm_ms=%d tts_first_chunk_ms=%d tts_total_ms=%d turn_total_ms=%d",
    session_id,
    response_id,
    detected_language,
    language_hint,
    type(self._assistant_backend).__name__,
    self._assistant_model,
    asr_ms,
    llm_ms,
    tts_first_chunk_ms,
    tts_total_ms,
    turn_total_ms,
)
```

```markdown
### Conversation assistant governance

`/ws/conversation` now keeps the last three turns of short-term memory, uses a phone-agent system prompt, sanitizes assistant output before TTS, and logs per-turn latency fields for ASR, LLM, and TTS.

Relevant environment variables:

- `OMNIVOICE_OLLAMA_HOST`
- `OMNIVOICE_OLLAMA_MODEL`
- `OMNIVOICE_ASR_MODEL`
- `OMNIVOICE_ASR_DEVICE`
- `OMNIVOICE_ASR_COMPUTE_TYPE`
```

- [ ] **Step 4: Run final verification for the governance slice**

Run: `pytest tests/test_conversation_ws.py tests/test_ws_duplex_client.py tests/test_ws_handler.py tests/test_ws_playback_client.py tests/test_audio_encoder.py tests/test_sentence_splitter.py -v`
Expected: PASS with all current duplex and playback tests green.

Run: `python -m py_compile app/server.py app/asr.py app/conversation_ws.py app/assistant_backends.py app/text_sanitizer.py examples/ws_duplex_client.py`
Expected: exit code 0 with no output.

- [ ] **Step 5: Commit**

```bash
git add app/server.py app/asr.py app/conversation_ws.py app/assistant_backends.py app/text_sanitizer.py tests/test_conversation_ws.py README.md
git commit -m "feat: govern duplex assistant responses"
```
