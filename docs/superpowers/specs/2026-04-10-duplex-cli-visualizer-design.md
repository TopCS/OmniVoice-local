# Duplex CLI Visualizer Design

## Goal

Add a terminal-side voice visualizer to the duplex workstation client so assistant playback feels more alive without changing the audio path.

## Current State

`examples/ws_duplex_client.py` already:

- captures microphone audio at 16 kHz
- streams turns to `/ws/conversation`
- receives assistant PCM audio at 24 kHz
- plays that audio locally through PyAudio

The client currently prints protocol events but has no visual playback indicator.

## Scope

Included:

- a terminal visualizer integrated into `examples/ws_duplex_client.py`
- default-on behavior when running in an interactive terminal
- automatic disable when stdout is not a TTY
- tests for level calculation, rendering, and non-TTY fallback

Excluded:

- real DSP or EQ filtering
- FFT/spectrum analysis
- multi-line dashboards
- any change to the actual audio stream

## Recommended Approach

Add a small VU-meter-style visualizer driven by the same PCM frames already queued for playback.

This keeps the implementation minimal: no extra dependencies, no extra audio copies beyond simple level calculation, and no change to the actual playback signal. The effect will look like an equalizer to the user, but the code remains a lightweight terminal meter.

## Alternatives Considered

### 1. True multi-band equalizer visualization

This would look flashier, but it requires FFT logic, more CPU, and more complex terminal drawing. It is unnecessary for the first iteration.

### 2. Optional `--visualizer` flag only

This is safer for script users, but the request here is for a more delightful default UX. We can still auto-disable in non-TTY mode, which covers the main downside.

### 3. Full curses-like terminal UI

This would provide richer state display, but it would complicate the client substantially and make debugging harder.

## Architecture

### `PlaybackController`

Keep `PlaybackController` responsible for audio playback and extend it to report the latest normalized frame level to a visualizer object.

It should not modify audio samples or delay playback. The visualizer observes playback; it does not participate in audio processing.

### `TerminalVisualizer`

Add a small helper in `examples/ws_duplex_client.py` with responsibilities:

- keep the latest state label such as `idle`, `listening`, `assistant`, `interrupted`
- receive normalized playback levels
- draw one terminal line using carriage-return updates
- clear or finish cleanly on shutdown

### Default activation rule

The visualizer should be enabled by default when `sys.stdout.isatty()` is true.

If stdout is not a TTY, it should be disabled automatically so piping/logging output remains clean.

## UX Behavior

### Visual form

Use a single-line bar such as:

```text
[assistant] ████████░░░░░░░░░░░░░░
```

The exact bar characters can remain simple ASCII if needed, but the shape should clearly show activity.

### States

The visualizer should show at least:

- `idle`
- `listening`
- `assistant`
- `interrupted`

### Update policy

Update on:

- each playback frame written
- state changes such as interruption or listening

Do not introduce a new high-frequency render loop unless necessary. Event-driven redraws are preferred.

## Testing Strategy

Add tests for:

- frame-level intensity calculation from PCM bytes
- rendering of the bar for low and high levels
- state label transitions for listening and interrupted states
- no-op behavior when stdout is not a TTY

These tests should remain unit-level and not require actual audio devices.

## Risks And Mitigations

### Terminal flicker

Risk:

- overly frequent redraws can make the terminal noisy

Mitigation:

- keep rendering to a single line
- only update on audio frames or explicit state changes

### Playback regression

Risk:

- extra visualization logic could accidentally delay playback

Mitigation:

- keep level calculation trivial
- keep rendering logic lightweight
- never block audio writes on terminal drawing

### Non-interactive output corruption

Risk:

- carriage-return rendering can pollute logs or pipes

Mitigation:

- disable automatically when stdout is not a TTY

## Success Criteria

- Running `examples/ws_duplex_client.py` in a normal terminal shows a live playback bar by default.
- The bar reacts to assistant audio activity and state changes.
- The client still behaves correctly for interruption and playback.
- Non-TTY usage remains clean because the visualizer disables itself automatically.
