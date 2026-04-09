# Full Duplex Conversation With Client-Side VAD Design

## Goal

Add a real-time conversational path that supports barge-in: the user can start speaking while the assistant is still talking, local playback stops immediately, and the system pivots to the new user turn with minimal latency.

## Current State

The repository already supports sentence-streaming TTS over `/ws/tts`. That endpoint accepts text-oriented messages such as `synthesize`, `text_chunk`, `text_flush`, and `set_voice`, then returns `audio_chunk` metadata plus binary audio, followed by `done`.

It does not currently accept microphone audio, perform ASR, manage conversational turn-taking, or expose cancellation primitives for an in-flight assistant response.

## Scope

Included:

- A new conversation-focused WebSocket protocol separate from `/ws/tts`.
- A duplex client flow where the workstation captures microphone input and plays assistant audio.
- Silero VAD on the workstation to detect local speech start/end and reduce useless uplink audio.
- Server-side orchestration for ASR, LLM turn handling, TTS, and cancellation of an in-flight assistant response.
- Explicit `response_id`-based interruption semantics so stale audio can be ignored safely.

Excluded from the first phase:

- Perfect acoustic echo cancellation.
- Always-on raw audio upload with server-side VAD only.
- Production-grade multi-party call behavior.
- Rich UI work; the first milestone can remain CLI or minimally script-driven.
- Over-optimized streaming ASR in the very first slice.

## Recommended Approach

Use a hybrid architecture:

- client-side Silero VAD on the workstation
- server-side ASR, conversation state, LLM orchestration, and TTS
- a new `/ws/conversation` endpoint that owns full-duplex session semantics

This keeps interruption latency low because the workstation can stop speaker playback the moment it hears new user speech, while the server remains the source of truth for response generation and cancellation.

## Alternatives Considered

### 1. Full server-side VAD and ingest

The workstation would stream raw microphone audio continuously and the server would decide everything.

Pros:

- thinner client
- simpler client deployment

Cons:

- higher bandwidth
- more server load
- slower perceived barge-in because interrupt detection waits for server round-trip

### 2. Client-side VAD and client-side ASR

The workstation would send text to the server after local recognition.

Pros:

- lower server cost
- simpler server control plane

Cons:

- much heavier client footprint
- harder to standardize across machines
- more moving parts outside the server deployment

### 3. Keep extending `/ws/tts`

Pros:

- fewer endpoints

Cons:

- conflates text-to-speech streaming with full conversational audio uplink
- makes the existing TTS contract harder to reason about
- higher regression risk for current clients

## Architecture

### Workstation duplex client

Responsibilities:

- capture microphone frames continuously
- run Silero VAD locally
- emit `speech_start` and `speech_end`
- send microphone audio frames only while speech is active or within a small hangover window
- play assistant PCM output as it arrives
- stop local playback immediately when the user starts speaking again
- ignore stale audio belonging to superseded assistant responses

The client is responsible for perceived interrupt responsiveness.

### New `/ws/conversation` endpoint

Responsibilities:

- accept session configuration
- receive audio uplink and control events
- run ASR when an utterance is ready
- manage conversation turn state
- start assistant generation
- stream assistant text and TTS audio back down
- cancel in-flight assistant responses when a new user turn starts

This endpoint should be treated as a separate protocol from `/ws/tts`, not as a loose extension of it.

### Shared synthesis layer

The current sentence-chunking and PCM streaming behavior under `/ws/tts` should be reused underneath the new conversation endpoint where practical. The key addition is cancellation awareness and response scoping.

### Session state

Per conversation session, the server needs to track:

- current utterance buffer and ingest status
- current assistant `response_id`
- canceled `response_id` values or an equivalent generation token
- short conversation history
- selected voice sample or other TTS configuration
- whether the assistant is currently speaking

## Protocol Design

### Client to server messages

- `session_start`
  includes conversation settings such as chosen sample, language hints, and TTS options
- `input_audio_chunk`
  carries PCM microphone audio frames
- `speech_start`
  tells the server the user has started a new utterance and should preempt assistant output
- `speech_end`
  tells the server the current utterance boundary is closed and ASR can finalize
- `interrupt`
  optional explicit cancellation message if the client wants to force a stop independently of VAD transitions

### Server to client messages

- `listening`
  confirms the server is ingesting user speech for the current utterance
- `transcript_partial`
  optional in later phases
- `transcript_final`
  final recognized text for the current user turn
- `assistant_text`
  partial or final assistant text for the active response
- `audio_chunk`
  assistant audio metadata paired with a binary payload, tagged with `response_id`
- `response_done`
  marks completion of one assistant response
- `interrupted`
  confirms a previous response was canceled
- `error`
  terminal or recoverable protocol/runtime error

## Barge-In Rules

### Local interruption

When Silero VAD detects user speech start:

1. the client stops local playback immediately
2. the client emits `speech_start` or `interrupt`
3. the server marks the current assistant response canceled
4. any remaining audio for the canceled response is ignored by the client

This local-first stop is essential to making the system feel conversational.

### Response scoping

Every assistant response must carry a `response_id`.

The client keeps track of the active response and drops any late-arriving `audio_chunk`, `assistant_text`, or `response_done` events whose `response_id` is no longer current.

This prevents race conditions where canceled audio leaks into the next turn.

## Rollout Plan

### Phase 1: Barge-in with utterance-final ASR

Target first:

- Silero VAD locally
- audio uplink over `/ws/conversation`
- ASR finalized on `speech_end`
- LLM response generation after final transcript
- TTS streaming back down
- immediate local playback stop plus server cancellation on barge-in

This is the recommended first milestone because it already feels conversational without requiring fully streaming ASR.

### Phase 2: Partial transcripts and lower latency

Add:

- optional `transcript_partial`
- earlier assistant text visibility
- tighter interruption timing and buffering heuristics

### Phase 3: Higher-fidelity duplex behavior

Possible later additions:

- echo control or AEC
- playback ducking
- smarter turn-state heuristics
- more robust reconnect/session recovery

## Testing Strategy

### Protocol tests

Add fast server tests for:

- `speech_start` interrupting an in-flight response
- stale `response_id` output being ignored or canceled correctly
- invalid message ordering and malformed audio chunks
- recoverable versus terminal error paths

### Client tests

Add targeted tests for:

- VAD transition handling
- playback interruption on local speech start
- dropping stale `response_id` audio
- correct client message sequencing around `speech_start`, `input_audio_chunk`, and `speech_end`

### Manual smoke tests

Run a real workstation test where:

1. the assistant begins speaking
2. the user interrupts mid-playback
3. local playback stops immediately
4. the new utterance is recognized
5. a new assistant response starts without leaking stale audio from the previous response

## Risks And Mitigations

### VAD false positives

Risk:

- breathing, room noise, or speaker leakage can trigger interruptions too aggressively

Mitigation:

- make VAD thresholds configurable
- keep the first release workstation-local and operator-tunable

### No AEC in first phase

Risk:

- assistant playback may re-enter the mic path and cause accidental barge-in

Mitigation:

- start with headphones or controlled audio setups
- treat AEC as a later phase, not a prerequisite for first duplex experiments

### Cancellation races

Risk:

- canceled response audio arrives after a new turn has already started

Mitigation:

- require `response_id` on all assistant downlink events
- have the client drop stale response output deterministically

## Success Criteria

- A workstation can capture mic audio, detect local speech with Silero VAD, and stream assistant playback simultaneously.
- If the user starts speaking while assistant audio is playing, local playback stops immediately.
- The server cancels the old response and starts a new turn cleanly.
- Assistant output is tagged so stale audio from old responses is ignored.
- The first milestone works without requiring perfect echo cancellation.
