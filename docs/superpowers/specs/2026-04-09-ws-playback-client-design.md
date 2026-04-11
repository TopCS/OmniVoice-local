# WebSocket Playback Client Design

## Goal

Add a minimal remote Python client with real-time PCM playback via PyAudio, document a real end-to-end smoke test against the existing `docker compose` stack using `websocat`, and extend WebSocket tests to cover key error paths.

## Context

The repository already exposes a sentence-streaming WebSocket endpoint at `/ws/tts` and has unit tests in `tests/test_ws_handler.py`. The current README includes a minimal Python receive loop, but it only prints events and byte lengths. The requested client is for a separate remote machine that can reach the server and has local audio output available.

## Scope

Included:

- A standalone Python example client for remote playback.
- README updates for remote client usage and real end-to-end smoke testing.
- New WebSocket tests for:
  - missing sample
  - invalid `output_format`
  - idle timeout

Excluded:

- Reconnect logic, buffering strategies, or transport resiliency features.
- Support for client-side playback of `wav` frames.
- A fully automated integration test that boots Docker and requires GPU/model assets.

## Recommended Approach

Use a standalone example script under `examples/` that depends on `websockets` and `pyaudio`, keep the existing server code mostly unchanged, and treat the Docker-based path as a documented smoke test rather than a normal automated test suite target.

This keeps client code separate from the server package, matches the real deployment model where the client runs on another machine, and avoids adding brittle infrastructure-coupled tests to `pytest`.

## Alternatives Considered

### 1. Put the client under `app/`

This would reduce some message-schema duplication, but it would mix a remote helper utility into the server package and create unnecessary coupling.

### 2. Add only shell examples with `websocat`

This would reduce code, but it would not satisfy the requirement for a minimal Python client with actual audio playback.

### 3. Automate the full Docker smoke test in `pytest`

This would look attractive on paper, but it would be flaky and environment-dependent because it relies on Docker, GPU access, model loading time, and real voice samples.

## File Plan

### `examples/ws_playback_client.py`

Responsibility: standalone remote playback client.

Expected behavior:

- Accept server URL and request parameters from CLI.
- Open a WebSocket connection to `/ws/tts`.
- Send a `synthesize` message with `text`, optional `sample`, optional `num_step`, and `output_format="pcm"`.
- Receive JSON control frames and binary audio frames.
- Stream binary PCM data directly to a PyAudio output stream configured for mono 16-bit 24000 Hz playback.
- Print structured events for `audio_chunk`, `done`, and `error`.
- Exit non-zero on protocol or server errors.

Intentional limits:

- Playback support is only for `pcm`.
- If the user requests any format other than `pcm`, the client should reject it locally with a clear message.
- No reconnect or retry behavior.

### `tests/test_ws_handler.py`

Responsibility: fast protocol-level tests of the WebSocket router.

Add coverage for:

- `set_voice` with a nonexistent sample returns the current error response shape, which today is an `error` event with the missing-sample message routed through code `SESSION_ERROR`, followed by connection close.
- `synthesize` with invalid `output_format` returns an `error` event with code `UNSUPPORTED_FORMAT`.
- Opening a connection and waiting past `inactivity_timeout_s` returns an `error` event with code `IDLE_TIMEOUT` and closes the socket.

These tests should continue to use the existing FastAPI `TestClient` setup and fake model to stay cheap and deterministic.

### `README.md`

Responsibility: operational documentation.

Changes:

- Replace or expand the current minimal Python client example with the new remote playback workflow.
- Document PyAudio installation notes for Linux/macOS where needed.
- Add a real smoke-test section using `docker compose` and `websocat`.
- Explicitly state that the smoke test requires a functioning runtime environment: Docker, model download, GPU access when needed, and at least one real voice sample in `./samples`.

### Optional helper script

Avoid adding a smoke-test shell script unless the README commands become repetitive enough to justify it. The default plan is README-only documentation.

## Protocol Behavior

### Client request flow

1. The remote client connects to the server URL.
2. It sends a single `synthesize` request.
3. The request includes `text`, optional `sample`, optional `num_step`, and `output_format="pcm"`.
4. The server emits one JSON `audio_chunk` message followed by one binary audio payload per synthesized sentence.
5. The client writes each binary payload to the open PyAudio stream immediately.
6. The server emits `done` after the synthesis request completes.
7. The client closes the socket and audio stream cleanly.

### Client-side validation

The client should validate the requested output format before sending anything. Only `pcm` is allowed for playback mode.

This avoids implying support for `wav` playback while keeping the example minimal.

### Server-side errors covered by tests

- Missing sample:
  a `set_voice` request naming a sample that is not present should lock in the current behavior from `app/ws_handler.py`: an `error` event whose message includes `Sample '<name>' not found.` and whose code is `SESSION_ERROR`, followed by socket close.
- Invalid output format:
  a `synthesize` request with `output_format` outside `{"pcm", "wav"}` should emit code `UNSUPPORTED_FORMAT`.
- Idle timeout:
  a connection with no traffic beyond `inactivity_timeout_s` should emit code `IDLE_TIMEOUT` and then close.

The design intentionally tests the current server behavior rather than redefining the protocol.

## Testing Strategy

### Unit and protocol tests

Extend `tests/test_ws_handler.py` first in TDD style:

- write each failing test
- verify the failure is for the intended reason
- implement the smallest server or documentation change needed
- rerun the targeted tests

These tests are the main regression protection.

### Manual remote client verification

From a remote machine with working speakers:

- install Python dependencies for the example client
- point it at the deployed WebSocket endpoint
- request speech using a known sample or supported auto-voice path
- confirm that audio is heard continuously as chunks arrive

This verifies the real playback path that cannot be exercised on the server host in the target environment.

### Real end-to-end smoke test

Treat this as an operator workflow, not a standard unit test:

1. Start the server with `docker compose up -d --build`.
2. Wait for `/health` to report ready.
3. Ensure at least one valid sample exists in `./samples`.
4. Connect with `websocat` to `/ws/tts`.
5. Send a `synthesize` message using a real sample and `output_format="pcm"`.
6. Verify that the session yields at least one `audio_chunk` event, one binary frame, and a final `done` event.

This covers the real stack without pretending it belongs in the normal fast test suite.

## Risks And Mitigations

### PyAudio installation friction

PyAudio can require system packages such as PortAudio. Mitigation: keep installation notes concise in the README and keep the example script dependency surface small.

### Protocol assumptions in the client

The client assumes the existing server ordering of JSON metadata followed by a binary frame. Mitigation: document that expectation explicitly and fail loudly if the stream shape is unexpected.

### Environment-dependent smoke test

The Docker smoke test depends on runtime assets and infrastructure. Mitigation: document it as a smoke test, not as a mandatory lightweight local test.

## Success Criteria

- A remote machine can run the example client and hear streamed audio playback from `/ws/tts`.
- The README clearly documents both remote playback usage and the real Docker plus `websocat` smoke test.
- WebSocket test coverage includes missing sample, unsupported format, and idle timeout.
- The changes stay minimal and preserve the existing WebSocket protocol.
