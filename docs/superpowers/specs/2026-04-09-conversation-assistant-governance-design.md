# Conversation Assistant Governance Design

## Goal

Improve the duplex conversation stack so the assistant behaves like a short, calm, direct, reassuring phone agent, avoids noisy formatting, keeps short-term conversational memory, uses sticky language bias for ASR, and emits per-turn latency logs for ASR, LLM, and TTS.

## Current State

The repository already has an experimental `/ws/conversation` path backed by:

- `faster-whisper` for ASR
- Ollama for assistant text generation
- OmniVoice streaming TTS for assistant audio

The current implementation still has three gaps for real conversational use:

1. assistant behavior is under-governed because there is no strong system prompt
2. assistant output is not sanitized before TTS
3. the conversation path is effectively single-turn in its semantic context because only the latest user transcript is sent to Ollama

Additionally, ASR currently relies on plain autodetect each turn, and the server does not yet emit focused per-turn latency telemetry for diagnosing where the interaction feels slow.

## Scope

Included:

- A provider-agnostic assistant backend interface
- Ollama as the default assistant backend
- Short-term conversation memory stored on the server
- A strong phone-agent system prompt
- Output sanitization before TTS
- Sticky language bias for ASR across turns, with optional explicit client override
- Per-turn latency logging for ASR, LLM, and TTS

Excluded from this slice:

- replacing Ollama as the default provider
- deep multilingual routing or automatic translation
- long-term memory or persistence across websocket sessions
- a UI for editing prompts or inspecting metrics

## Recommended Approach

Add a small assistant abstraction layer and keep memory/state server-side.

The server should build the final `messages` array for the assistant model, prepend a strong system prompt, keep only the last three turns of history, sanitize assistant output before sentence splitting and TTS, and log per-turn timings. Ollama remains the default backend, but the code structure should allow an OpenAI backend later without reworking the conversation protocol.

## Alternatives Considered

### 1. Ollama-only inline prompt edits

This would patch behavior quickly, but it would couple prompt policy, provider integration, and session memory too tightly inside `ConversationService`.

### 2. Switch default provider to OpenAI now

This might improve prompt adherence immediately, but it increases cost and external dependency surface before measuring what stronger governance can already fix on Ollama.

### 3. Keep everything stateless and rely on the prompt only

This is the smallest code change, but it leaves obvious conversational continuity problems and makes interruptions feel disconnected from prior turns.

## Architecture

### Assistant backend abstraction

Introduce an `AssistantBackend` interface with one responsibility: given a prepared `messages` array and backend settings, return assistant plain text.

Initial implementations:

- `OllamaAssistantBackend` as the default
- `OpenAIAssistantBackend` as a future-compatible optional provider

The rest of the conversation stack should not care which backend produced the text.

### Session memory

Add short-term semantic memory to the conversation websocket session state.

Store up to three turns of:

- user transcript
- assistant final text

The server should trim history aggressively and only include the last three turns in the prompt window. This keeps the system responsive and avoids uncontrolled context growth.

### ASR language bias

Use sticky language bias rather than hard language lock.

Behavior:

- first turn uses normal detection
- later turns pass the previously detected language as the `language=` hint to `faster-whisper`
- if the client explicitly overrides language, that override wins

This keeps the system stable across turns without making language switching impossible.

### Output sanitization layer

Before sentence splitting and TTS, assistant text should pass through a sanitization step that removes formatting artifacts likely to sound bad when spoken.

This layer should normalize:

- code fences
- markdown bullets/headings
- obvious JSON wrappers
- repeated punctuation noise
- excessive whitespace

The sanitization step should preserve plain conversational meaning while forcing the output into TTS-friendly plain text.

## Assistant Prompt Policy

The server should inject a system prompt that enforces all of the following:

- the assistant is a voice agent for a phone-like spoken conversation
- tone is calm, direct, and reassuring
- answers are short
- output must be plain text only
- no markdown
- no JSON
- no lists or bullets
- no emojis
- no code blocks
- no prefixed labels like `Answer:` or `Response:`
- if clarification is needed, ask for it in one short sentence
- avoid generating symbols or formatting that would sound unnatural when read aloud

The policy should be enforced both by the prompt and by sanitization, not by prompt alone.

## Protocol Changes

### Client language override

Allow an optional language hint in `session_start`, such as:

```json
{"type":"session_start","sample":"voce_1","language":"it"}
```

If present, the server should use it as the ASR language hint for the session until changed.

If absent, the server should bias future ASR turns using the most recently detected language.

### No protocol change for assistant provider

The client should not need to know whether Ollama or another assistant backend generated the response. Provider selection remains server-side.

## Observability

For each completed user turn, emit structured server logs containing at least:

- `conversation_id` or equivalent session identifier
- `response_id`
- `detected_language`
- `language_hint`
- `assistant_backend`
- `assistant_model`
- `asr_ms`
- `llm_ms`
- `tts_first_chunk_ms`
- `tts_total_ms`
- `turn_total_ms`

These logs are intended for operator diagnosis, not end-user protocol exposure.

## Testing Strategy

### Unit and protocol tests

Add tests for:

- system prompt inclusion in assistant backend calls
- memory trimming to the last three turns
- sanitization of markdown, JSON-like wrappers, and noisy formatting
- language sticky bias after first detected turn
- explicit client language override in `session_start`
- latency logging fields emitted for a completed turn

### Regression focus

Ensure the new governance layer does not break:

- interruption semantics
- response scoping by `response_id`
- existing TTS chunk streaming

### Manual validation

In a real duplex run, verify:

1. the user speaks in Italian for the first turn
2. the assistant responds briefly in plain text style suitable for speech
3. the next turn remains biased to Italian without explicit reconfiguration
4. the assistant does not emit markdown or JSON artifacts in the spoken output
5. logs make it obvious whether latency is dominated by ASR, LLM, or TTS

## Risks And Mitigations

### Prompt still not obeyed perfectly

Risk:

- Ollama may still leak formatting or overly long answers

Mitigation:

- combine prompt constraints with output sanitization
- keep provider abstraction so OpenAI can be enabled later if needed

### Sticky language over-corrects real language changes

Risk:

- the server may bias too strongly toward the previous language

Mitigation:

- allow explicit client override
- treat the bias as a hint, not a hard lock

### Memory introduces drift

Risk:

- extra history can make answers less concise

Mitigation:

- keep only three turns
- keep the system prompt strong and concise

## Success Criteria

- The assistant uses a calm, direct, reassuring phone-agent tone.
- Assistant output is plain text suitable for TTS, without markdown/JSON formatting artifacts.
- The server maintains three turns of short-term memory per websocket session.
- ASR uses sticky language bias after the first turn, unless the client explicitly overrides it.
- Server logs expose enough timing detail to identify whether ASR, LLM, or TTS dominates latency.
- Ollama remains the default provider while the architecture can support OpenAI later.
