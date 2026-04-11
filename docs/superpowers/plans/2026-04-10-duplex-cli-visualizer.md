# Duplex CLI Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-on terminal VU-meter visualizer to `examples/ws_duplex_client.py` that reacts to assistant playback and state changes without altering the audio path.

**Architecture:** Extend the existing duplex client with a lightweight `TerminalVisualizer` that is fed normalized frame levels from `PlaybackController` and state changes from `DuplexState`/receive loop. Keep the feature auto-enabled only in interactive terminals and fully disabled for non-TTY output.

**Tech Stack:** Python, asyncio, PyAudio, terminal stdout rendering, pytest

---

## File Structure

- Modify: `examples/ws_duplex_client.py`
  Add visualizer state, level calculation, default-on TTY activation, and single-line rendering tied to playback/state events.

- Modify: `tests/test_ws_duplex_client.py`
  Add unit tests for audio level calculation, bar rendering, state transitions, and non-TTY fallback.

### Task 1: Add Terminal Visualizer Primitives

**Files:**
- Modify: `examples/ws_duplex_client.py`
- Modify: `tests/test_ws_duplex_client.py`
- Test: `tests/test_ws_duplex_client.py`

- [ ] **Step 1: Write the failing tests for visualizer level and rendering**

```python
def test_calculate_pcm_level_returns_zero_for_silence():
    client = _load_module()

    assert client.calculate_pcm_level(b"\x00\x00" * 32) == 0.0


def test_calculate_pcm_level_returns_positive_value_for_signal():
    client = _load_module()

    level = client.calculate_pcm_level((1000).to_bytes(2, "little", signed=True) * 32)

    assert 0.0 < level <= 1.0


def test_terminal_visualizer_renders_state_and_bar():
    client = _load_module()

    writes = []

    class FakeTTY:
        def isatty(self):
            return True

        def write(self, text):
            writes.append(text)

        def flush(self):
            writes.append("<flush>")

    visualizer = client.TerminalVisualizer(FakeTTY(), enabled=True)
    visualizer.set_state("assistant")
    visualizer.update_level(0.5)

    rendered = "".join(part for part in writes if part != "<flush>")
    assert "assistant" in rendered
    assert "#" in rendered or "=" in rendered
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_ws_duplex_client.py::test_calculate_pcm_level_returns_zero_for_silence tests/test_ws_duplex_client.py::test_calculate_pcm_level_returns_positive_value_for_signal tests/test_ws_duplex_client.py::test_terminal_visualizer_renders_state_and_bar -v`
Expected: FAIL because the visualizer helpers do not exist yet.

- [ ] **Step 3: Implement the minimal visualizer helpers**

```python
import math

VISUALIZER_WIDTH = 24


def calculate_pcm_level(audio: bytes) -> float:
    if not audio:
        return 0.0
    sample_count = len(audio) // 2
    if sample_count == 0:
        return 0.0

    total = 0.0
    for offset in range(0, len(audio) - 1, 2):
        sample = int.from_bytes(audio[offset : offset + 2], "little", signed=True)
        total += float(sample * sample)

    rms = math.sqrt(total / sample_count) / 32768.0
    return max(0.0, min(rms, 1.0))


class TerminalVisualizer:
    def __init__(self, stream, *, enabled: bool):
        self._stream = stream
        self._enabled = enabled and hasattr(stream, "isatty") and stream.isatty()
        self._state = "idle"
        self._last_render = ""

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_state(self, state: str) -> None:
        self._state = state

    def update_level(self, level: float) -> None:
        if not self._enabled:
            return
        filled = max(0, min(VISUALIZER_WIDTH, round(level * VISUALIZER_WIDTH)))
        bar = "#" * filled + "-" * (VISUALIZER_WIDTH - filled)
        self._render(f"[{self._state}] {bar}")

    def finish(self) -> None:
        if not self._enabled:
            return
        self._render("")

    def _render(self, text: str) -> None:
        clear_suffix = " " * max(0, len(self._last_render) - len(text))
        self._stream.write("\r" + text + clear_suffix)
        self._stream.flush()
        self._last_render = text
```

- [ ] **Step 4: Run the visualizer tests to verify they pass**

Run: `pytest tests/test_ws_duplex_client.py -k 'calculate_pcm_level or terminal_visualizer' -v`
Expected: PASS with the new helper tests green.

- [ ] **Step 5: Commit**

```bash
git add examples/ws_duplex_client.py tests/test_ws_duplex_client.py
git commit -m "feat: add duplex cli visualizer primitives"
```

### Task 2: Wire Visualizer Into Playback And State Changes

**Files:**
- Modify: `examples/ws_duplex_client.py`
- Modify: `tests/test_ws_duplex_client.py`
- Test: `tests/test_ws_duplex_client.py`

- [ ] **Step 1: Write the failing integration tests for playback/state updates**

```python
@pytest.mark.asyncio
async def test_playback_controller_updates_visualizer_from_audio_frames():
    client = _load_module()
    stream = FakeStream()

    class FakeVisualizer:
        def __init__(self):
            self.levels = []

        def update_level(self, level):
            self.levels.append(level)

        def set_state(self, state):
            pass

        def finish(self):
            pass

    visualizer = FakeVisualizer()
    playback = client.PlaybackController(stream, visualizer=visualizer)

    task = asyncio.create_task(playback.run())
    playback.enqueue((1000).to_bytes(2, "little", signed=True) * 64)
    await asyncio.sleep(0.05)
    await playback.close()
    await task

    assert visualizer.levels
    assert visualizer.levels[-1] > 0.0


def test_duplex_state_updates_visualizer_states():
    client = _load_module()

    class FakeVisualizer:
        def __init__(self):
            self.states = []

        def set_state(self, state):
            self.states.append(state)

    visualizer = FakeVisualizer()
    state = client.DuplexState(visualizer=visualizer)
    playback = FakePlayback()

    state.start_local_speech(playback)
    state.handle_event({"type": "listening"}, playback)
    state.handle_event({"type": "assistant_text", "text": "hi", "response_id": 1}, playback)
    state.handle_event({"type": "interrupted", "response_id": 1}, playback)

    assert visualizer.states == ["listening", "listening", "assistant", "interrupted"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_ws_duplex_client.py::test_playback_controller_updates_visualizer_from_audio_frames tests/test_ws_duplex_client.py::test_duplex_state_updates_visualizer_states -v`
Expected: FAIL because playback/state are not yet wired to a visualizer.

- [ ] **Step 3: Integrate the visualizer into the client runtime**

```python
# examples/ws_duplex_client.py key changes
@dataclass
class DuplexState:
    ...
    visualizer: TerminalVisualizer | None = None

    def _set_visualizer_state(self, state: str) -> None:
        if self.visualizer is not None:
            self.visualizer.set_state(state)

    def start_local_speech(self, playback) -> dict[str, object] | None:
        ...
        self._set_visualizer_state("listening")
        return {"type": "speech_start"}

    def handle_event(self, event: dict[str, object], playback) -> dict[str, object] | None:
        ...
        if event_type == "assistant_text":
            self._set_visualizer_state("assistant")
        elif event_type == "interrupted":
            self._set_visualizer_state("interrupted")
        elif event_type == "response_done":
            self._set_visualizer_state("idle")
```

```python
class PlaybackController:
    def __init__(self, stream, *, visualizer=None):
        self.stream = stream
        self._visualizer = visualizer
        ...

    async def run(self) -> None:
        ...
        await asyncio.to_thread(self.stream.write, audio)
        if self._visualizer is not None:
            self._visualizer.update_level(calculate_pcm_level(audio))

    async def close(self) -> None:
        ...
        if self._visualizer is not None:
            self._visualizer.finish()
```

```python
# in run_client()
visualizer = TerminalVisualizer(sys.stdout, enabled=True)
playback = PlaybackController(output_stream, visualizer=visualizer)
state = DuplexState(sample=sample, visualizer=visualizer)
```

- [ ] **Step 4: Run the duplex client test file**

Run: `pytest tests/test_ws_duplex_client.py -v`
Expected: PASS with visualizer integration covered and existing duplex tests still green.

- [ ] **Step 5: Commit**

```bash
git add examples/ws_duplex_client.py tests/test_ws_duplex_client.py
git commit -m "feat: wire duplex cli visualizer"
```

### Task 3: Add Default-On TTY Fallback And Final Verification

**Files:**
- Modify: `examples/ws_duplex_client.py`
- Modify: `tests/test_ws_duplex_client.py`
- Test: `tests/test_ws_duplex_client.py`

- [ ] **Step 1: Write the failing tests for default-on TTY and non-TTY fallback**

```python
def test_terminal_visualizer_disables_itself_for_non_tty_stream():
    client = _load_module()

    class FakePipe:
        def isatty(self):
            return False

        def write(self, text):
            raise AssertionError(f"write should not be called: {text!r}")

        def flush(self):
            raise AssertionError("flush should not be called")

    visualizer = client.TerminalVisualizer(FakePipe(), enabled=True)

    assert visualizer.enabled is False
    visualizer.set_state("assistant")
    visualizer.update_level(0.8)
    visualizer.finish()


def test_run_client_builds_visualizer_enabled_for_tty(monkeypatch):
    client = _load_module()
    created = {}

    class FakeTTY:
        def isatty(self):
            return True

        def write(self, text):
            pass

        def flush(self):
            pass

    real_visualizer = client.TerminalVisualizer

    def wrapped_visualizer(stream, *, enabled):
        created["stream"] = stream
        created["enabled"] = enabled
        return real_visualizer(stream, enabled=enabled)

    monkeypatch.setattr(client, "TerminalVisualizer", wrapped_visualizer)
    monkeypatch.setattr(client.sys, "stdout", FakeTTY())
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_ws_duplex_client.py -k 'non_tty_stream or visualizer_enabled_for_tty' -v`
Expected: FAIL until the default-on TTY behavior is fully wired.

- [ ] **Step 3: Finalize default-on behavior and clean shutdown**

```python
# examples/ws_duplex_client.py key change
def build_visualizer(stream) -> TerminalVisualizer:
    return TerminalVisualizer(stream, enabled=True)


async def run_client(...):
    ...
    visualizer = build_visualizer(sys.stdout)
    playback = PlaybackController(output_stream, visualizer=visualizer)
    state = DuplexState(sample=sample, visualizer=visualizer)
```

Keep `TerminalVisualizer` responsible for checking `isatty()` internally so default-on behavior is safe in interactive terminals and silent otherwise.

- [ ] **Step 4: Run final verification for the visualizer slice**

Run: `pytest tests/test_ws_duplex_client.py tests/test_conversation_ws.py tests/test_ws_playback_client.py -v`
Expected: PASS with duplex client, conversation, and playback client tests all green.

Run: `python -m py_compile examples/ws_duplex_client.py`
Expected: exit code 0 with no output.

- [ ] **Step 5: Commit**

```bash
git add examples/ws_duplex_client.py tests/test_ws_duplex_client.py
git commit -m "feat: add duplex cli playback visualizer"
```
