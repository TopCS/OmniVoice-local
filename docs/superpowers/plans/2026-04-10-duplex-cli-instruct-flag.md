# Duplex CLI Instruct Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--instruct` to `examples/ws_duplex_client.py` so the duplex CLI can start `/ws/conversation` sessions with inferred voice design when no explicit sample is provided.

**Architecture:** Keep the client as a thin wrapper over the existing websocket protocol. Extend `build_session_start(...)`, `run_client(...)`, and `main(...)` to accept `instruct`, pass it through in `session_start`, and keep the server-side precedence rule `sample > instruct` unchanged.

**Tech Stack:** Python, argparse, asyncio, pytest

---

## File Structure

- Modify: `examples/ws_duplex_client.py`
  Add `--instruct`, thread it through `build_session_start(...)`, `run_client(...)`, and `main(...)`.

- Modify: `tests/test_ws_duplex_client.py`
  Add regression coverage for `build_session_start(...)` and `main(...)` argument propagation.

### Task 1: Add `--instruct` To The Duplex CLI

**Files:**
- Modify: `examples/ws_duplex_client.py`
- Modify: `tests/test_ws_duplex_client.py`
- Test: `tests/test_ws_duplex_client.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_session_start_includes_instruct_when_present():
    client = _load_module()

    assert client.build_session_start(
        sample="voice-a",
        sample_rate=16000,
        language="it",
        instruct="female, calm, warm voice",
    ) == {
        "type": "session_start",
        "sample": "voice-a",
        "sample_rate": 16000,
        "language": "it",
        "instruct": "female, calm, warm voice",
    }


def test_main_propagates_instruct_to_run_client(monkeypatch):
    client = _load_module()
    captured = {}

    async def fake_run_client(
        url,
        sample,
        language,
        instruct,
        websockets_module,
        pyaudio_module,
        torch_module,
        silero_vad_module,
    ):
        captured.update(
            {
                "url": url,
                "sample": sample,
                "language": language,
                "instruct": instruct,
            }
        )

    monkeypatch.setattr(client, "run_client", fake_run_client)
    monkeypatch.setitem(sys.modules, "websockets", types.ModuleType("websockets"))
    monkeypatch.setitem(sys.modules, "pyaudio", types.ModuleType("pyaudio"))
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "silero_vad", types.ModuleType("silero_vad"))

    assert client.main([
        "--url", "ws://example/ws/conversation",
        "--sample", "voice-a",
        "--language", "it",
        "--instruct", "female, calm, warm voice",
    ]) == 0

    assert captured == {
        "url": "ws://example/ws/conversation",
        "sample": "voice-a",
        "language": "it",
        "instruct": "female, calm, warm voice",
    }
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_ws_duplex_client.py -k 'includes_instruct_when_present or propagates_instruct_to_run_client' -v`
Expected: FAIL because the duplex client does not yet accept or forward `instruct`.

- [ ] **Step 3: Implement the minimal CLI wiring**

```python
def build_session_start(
    *,
    sample: str | None = None,
    sample_rate: int = UPLINK_SAMPLE_RATE,
    language: str | None = None,
    instruct: str | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "session_start",
        "sample_rate": sample_rate,
    }
    if sample is not None:
        event["sample"] = sample
    if language is not None:
        event["language"] = language
    if instruct is not None:
        event["instruct"] = instruct
    return event


async def run_client(
    url: str,
    sample: str | None,
    language: str | None,
    instruct: str | None,
    websockets_module,
    pyaudio_module,
    torch_module,
    silero_vad_module,
) -> None:
    async with websockets_module.connect(url) as websocket:
        await websocket.send(
            json.dumps(
                build_session_start(
                    sample=sample,
                    sample_rate=UPLINK_SAMPLE_RATE,
                    language=language,
                    instruct=instruct,
                )
            )
        )
        ...


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--url", default="ws://localhost:8000/ws/conversation")
    parser.add_argument("--sample")
    parser.add_argument("--language")
    parser.add_argument(
        "--instruct",
        help="Optional inferred voice-design text. If --sample is also set, the selected sample still wins server-side.",
    )
    ...
    asyncio.run(
        run_client(
            args.url,
            args.sample,
            args.language,
            args.instruct,
            websockets,
            pyaudio,
            torch,
            silero_vad,
        )
    )
```

- [ ] **Step 4: Run the duplex client test file**

Run: `pytest tests/test_ws_duplex_client.py -v`
Expected: PASS with the new `instruct` wiring covered and existing client tests still green.

- [ ] **Step 5: Commit**

```bash
git add examples/ws_duplex_client.py tests/test_ws_duplex_client.py
git commit -m "feat: add duplex cli instruct flag"
```
