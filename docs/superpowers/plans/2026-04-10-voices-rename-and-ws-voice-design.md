# Voices Rename And WebSocket Voice Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the public voice asset surface from `samples` to `voices`, support inferred voice design over WebSocket when no explicit voice sample is provided, and allow a constrained set of expressive non-verbal tags to pass through the assistant/TTS pipeline.

**Architecture:** Keep existing payload compatibility for `sample`, but move filesystem/config/native endpoints to `voices` as the canonical public surface. Add compatibility aliases for the old `samples` names, extend WebSocket handlers to honor `instruct` when `sample` is absent, and relax assistant sanitization just enough to preserve a strict allowlist of expressive tags.

**Tech Stack:** FastAPI, WebSocket handlers, pytest, existing OmniVoice generation path, Docker Compose, Python logging

---

## File Structure

- Modify: `app/server.py`
  Rename canonical directory/config/API names to `voices`, add alias endpoints/env fallback, and update scanners/logs.

- Modify: `app/ws_handler.py`
  Support `instruct` for `/ws/tts` when no explicit `sample` is provided, while keeping `sample` precedence.

- Modify: `app/conversation_ws.py`
  Thread `instruct` through the duplex WebSocket session state and response path with `sample` precedence.

- Modify: `app/text_sanitizer.py`
  Preserve only the allowlisted expressive tags while continuing to strip unrelated formatting noise.

- Modify: `app/assistant_backends.py`
  Update the phone-agent prompt to allow the exact expressive tags and to discourage overuse.

- Modify: `compose.yaml`
  Bind-mount `./voices:/voices` and switch the canonical env var to `OMNIVOICE_VOICES_DIR`.

- Modify: `README.md`
  Update public terminology, new endpoints, alias/deprecation notes, WebSocket `instruct` behavior, and expressive-tag docs.

- Modify: `tests/test_ws_handler.py`
  Add WS TTS coverage for `instruct` behavior and `sample` precedence.

- Modify: `tests/test_conversation_ws.py`
  Add duplex WS coverage for `instruct`, alias endpoint behavior, env-var precedence, and expressive-tag sanitization/prompt policy.

### Task 1: Rename Public Voice Surface To `voices`

**Files:**
- Modify: `app/server.py`
- Modify: `compose.yaml`
- Modify: `README.md`
- Modify: `tests/test_conversation_ws.py`
- Test: `tests/test_conversation_ws.py`

- [ ] **Step 1: Write the failing tests for canonical `voices` endpoints and env precedence**

```python
def test_get_voices_lists_loaded_voice_assets(monkeypatch):
    server = _load_server_module(monkeypatch, conversation_enabled=False)
    server.voice_samples = {
        "azzurra": {"audio_path": "/voices/azzurra.wav", "ref_text": "ciao"}
    }
    client = TestClient(server.app)

    response = client.get("/voices")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "azzurra",
            "audio_file": "azzurra.wav",
            "has_transcript": True,
            "transcript_preview": "ciao",
        }
    ]


def test_samples_endpoints_remain_as_compatibility_aliases(monkeypatch):
    server = _load_server_module(monkeypatch, conversation_enabled=False)
    server.voice_samples = {
        "azzurra": {"audio_path": "/voices/azzurra.wav", "ref_text": "ciao"}
    }
    client = TestClient(server.app)

    voices_response = client.get("/voices")
    samples_response = client.get("/samples")

    assert voices_response.status_code == 200
    assert samples_response.status_code == 200
    assert samples_response.json() == voices_response.json()
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_conversation_ws.py -k 'get_voices_lists_loaded_voice_assets or samples_endpoints_remain_as_compatibility_aliases' -v`
Expected: FAIL because `/voices` does not exist yet and the docs/config still use `samples` as canonical.

- [ ] **Step 3: Implement the canonical rename with alias fallback**

```python
# app/server.py key changes
VOICES_DIR = Path(
    os.environ.get("OMNIVOICE_VOICES_DIR")
    or os.environ.get("OMNIVOICE_SAMPLES_DIR", "/voices")
)


def scan_voices(directory: Path) -> dict:
    ...  # renamed scanner logic, same structure


@app.get("/voices", response_model=list[SampleInfo], tags=["Voices"])
async def list_voices(_: None = Depends(require_api_key)):
    return _serialize_voice_infos(voice_samples)


@app.get("/samples", response_model=list[SampleInfo], tags=["Voices"])
async def list_samples_alias(_: None = Depends(require_api_key)):
    return await list_voices(_)


@app.post("/voices/reload", tags=["Voices"])
async def reload_voices(_: None = Depends(require_api_key)):
    global voice_samples
    voice_samples = scan_voices(VOICES_DIR)
    if conversation_service is not None:
        conversation_service.clear_voice_prompt_cache()
    return {"status": "ok", "count": len(voice_samples)}


@app.post("/samples/reload", tags=["Voices"])
async def reload_samples_alias(_: None = Depends(require_api_key)):
    return await reload_voices(_)
```

```yaml
# compose.yaml key changes
    volumes:
      - ./voices:/voices
    environment:
      - OMNIVOICE_VOICES_DIR=/voices
```

Update README examples and directory descriptions so `voices/` is canonical and `samples` is documented as a deprecated alias only where necessary.

- [ ] **Step 4: Run the voices-surface tests again**

Run: `pytest tests/test_conversation_ws.py -k 'voices|samples_endpoints_remain_as_compatibility_aliases' -v`
Expected: PASS with `/voices` canonical behavior and `/samples` compatibility confirmed.

- [ ] **Step 5: Commit**

```bash
git add app/server.py compose.yaml README.md tests/test_conversation_ws.py
git commit -m "feat: rename public voice surface to voices"
```

### Task 2: Support WebSocket `instruct` Voice Design With `sample` Precedence

**Files:**
- Modify: `app/ws_handler.py`
- Modify: `app/conversation_ws.py`
- Modify: `tests/test_ws_handler.py`
- Modify: `tests/test_conversation_ws.py`
- Test: `tests/test_ws_handler.py`
- Test: `tests/test_conversation_ws.py`

- [ ] **Step 1: Write the failing tests for `instruct` behavior and precedence**

```python
def test_ws_tts_uses_instruct_when_sample_missing():
    client, model = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50)
    )

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json(
            {
                "type": "synthesize",
                "text": "Ciao!",
                "instruct": "female, calm, warm voice",
                "output_format": "pcm",
            }
        )
        meta = ws.receive_json()
        audio = ws.receive_bytes()
        done = ws.receive_json()

    assert meta["type"] == "audio_chunk"
    assert isinstance(audio, bytes)
    assert done["type"] == "done"
    assert model.generate_calls[0]["instruct"] == "female, calm, warm voice"
    assert "voice_clone_prompt" not in model.generate_calls[0]


def test_ws_tts_sample_wins_over_instruct_when_both_present():
    client, model = _make_client(
        WSConfig(min_sentence_chars=1, max_sentence_chars=150, buffer_timeout_ms=50)
    )

    with client.websocket_connect("/ws/tts") as ws:
        ws.send_json(
            {
                "type": "synthesize",
                "text": "Ciao!",
                "sample": "agent-voice",
                "instruct": "ignore me",
                "output_format": "pcm",
            }
        )
        ws.receive_json()
        ws.receive_bytes()
        ws.receive_json()

    assert "voice_clone_prompt" in model.generate_calls[0]
    assert "instruct" not in model.generate_calls[0]
```

Add a duplex test variant showing `/ws/conversation` forwards `instruct` into the assistant/TTS response path when `sample` is absent.

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_ws_handler.py -k 'uses_instruct_when_sample_missing or sample_wins_over_instruct' -v`
Expected: FAIL because current handlers only partially support `instruct` and do not explicitly document or test precedence.

- [ ] **Step 3: Implement the minimal WebSocket voice-design routing**

```python
# app/ws_handler.py key idea
if state.voice_clone_prompt is not None:
    gen_kwargs["voice_clone_prompt"] = state.voice_clone_prompt
elif message.get("instruct"):
    gen_kwargs["instruct"] = str(message["instruct"])
```

```python
# app/conversation_ws.py key idea
@dataclass
class ConversationSessionState:
    ...
    instruct: str | None = None


if msg_type == "session_start":
    ...
    instruct = message.get("instruct")
    if instruct is not None and (not isinstance(instruct, str) or not instruct.strip()):
        await _send_error(...)
        return
    state.instruct = instruct.strip() if isinstance(instruct, str) else None
```

Then thread `state.instruct` into the conversation service response call. In `ConversationService.respond(...)`, when `sample` is absent and `instruct` is present, pass `instruct` into `_generate_sentence_audio(...)` instead of `voice_clone_prompt`.

- [ ] **Step 4: Run the WebSocket handler and conversation tests again**

Run: `pytest tests/test_ws_handler.py tests/test_conversation_ws.py -k 'instruct or sample_wins_over_instruct' -v`
Expected: PASS with both WS paths honoring `sample` precedence and inferred voice design fallback.

- [ ] **Step 5: Commit**

```bash
git add app/ws_handler.py app/conversation_ws.py tests/test_ws_handler.py tests/test_conversation_ws.py
git commit -m "feat: add websocket voice design fallback"
```

### Task 3: Allow Expressive Tags Through Assistant Prompt And Sanitizer

**Files:**
- Modify: `app/assistant_backends.py`
- Modify: `app/text_sanitizer.py`
- Modify: `tests/test_conversation_ws.py`
- Test: `tests/test_conversation_ws.py`

- [ ] **Step 1: Write the failing tests for expressive-tag allowlist**

```python
@pytest.mark.asyncio
async def test_ollama_assistant_backend_prompt_allows_only_supported_expression_tags():
    chat_calls = []

    class FakeAsyncClient:
        async def chat(self, **kwargs):
            chat_calls.append(kwargs)
            return {"message": {"content": "[sigh] Va bene."}}

    assistant_backends = _load_app_module(
        "assistant_backends_tags_under_test", "assistant_backends.py"
    )
    backend = assistant_backends.OllamaAssistantBackend(FakeAsyncClient(), "phone-agent-model")

    reply = await backend.generate_response("Vorrei aiuto")

    assert reply == "[sigh] Va bene."
    system_prompt = chat_calls[0]["messages"][0]["content"]
    assert "[laughter]" in system_prompt
    assert "[question-en]" in system_prompt
    assert "[dissatisfaction-hnn]" in system_prompt


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("[sigh] Va bene.", "[sigh] Va bene."),
        ("[unknown-tag] Va bene.", "Va bene."),
        ("[question-en] Puo aiutarmi?", "[question-en] Puo aiutarmi?"),
    ],
)
def test_text_sanitizer_preserves_only_allowlisted_expression_tags(raw_text, expected):
    text_sanitizer = _load_app_module("text_sanitizer_tags_under_test", "text_sanitizer.py")
    assert text_sanitizer.sanitize_assistant_text(raw_text) == expected
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `pytest tests/test_conversation_ws.py -k 'expression_tags or allowlisted_expression_tags' -v`
Expected: FAIL because the prompt and sanitizer do not yet encode the allowlist behavior.

- [ ] **Step 3: Implement the allowlist in prompt and sanitization**

```python
# app/assistant_backends.py key additions
ALLOWED_EXPRESSION_TAGS = (
    "[laughter]",
    "[sigh]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
)

PHONE_AGENT_SYSTEM_PROMPT = (
    ...
    "The only allowed non-verbal tags are: [laughter], [sigh], ... [dissatisfaction-hnn]. "
    "Use them sparingly and only when they improve spoken naturalness."
)
```

```python
# app/text_sanitizer.py key idea
ALLOWED_EXPRESSION_TAGS = {...}
TAG_RE = re.compile(r"\[[^\]]+\]")


def _preserve_allowed_tags(text: str) -> str:
    def _replace(match):
        tag = match.group(0)
        return tag if tag in ALLOWED_EXPRESSION_TAGS else ""
    return TAG_RE.sub(_replace, text)


def sanitize_assistant_text(text: Any) -> str:
    ...
    cleaned = _preserve_allowed_tags(cleaned)
    ...
```

- [ ] **Step 4: Run the conversation tests again**

Run: `pytest tests/test_conversation_ws.py -k 'expression_tags or allowlisted_expression_tags or sanitizes_assistant_output_before_tts' -v`
Expected: PASS with prompt allowlist and sanitizer behavior covered.

- [ ] **Step 5: Commit**

```bash
git add app/assistant_backends.py app/text_sanitizer.py tests/test_conversation_ws.py
git commit -m "feat: allow expressive speech tags"
```

### Task 4: Update Documentation And Final Verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_ws_handler.py`
- Test: `tests/test_conversation_ws.py`

- [ ] **Step 1: Update README for voices, aliases, WebSocket voice design, and expressive tags**

```markdown
### Voices directory

The canonical voice asset directory is now `./voices/`.

Compatibility aliases remain available for now:

- `GET /samples` -> alias of `GET /voices`
- `POST /samples/reload` -> alias of `POST /voices/reload`
- `OMNIVOICE_SAMPLES_DIR` -> deprecated fallback for `OMNIVOICE_VOICES_DIR`

### WebSocket inferred voice design

For `/ws/tts` and `/ws/conversation`:

- use `sample` for voice cloning
- use `instruct` for inferred voice design when `sample` is omitted
- if both are provided, `sample` wins

Example:

```json
{"type":"synthesize","text":"Ciao!","instruct":"female, calm, warm voice","output_format":"pcm"}
```

### Expressive tags

The assistant may use only these tags in spoken output:

- `[laughter]`
- `[sigh]`
- `[confirmation-en]`
- `[question-en]`
- `[question-ah]`
- `[question-oh]`
- `[question-ei]`
- `[question-yi]`
- `[surprise-ah]`
- `[surprise-oh]`
- `[surprise-wa]`
- `[surprise-yo]`
- `[dissatisfaction-hnn]`
```

- [ ] **Step 2: Run final verification for this slice**

Run: `pytest tests/test_ws_handler.py tests/test_conversation_ws.py -v`
Expected: PASS with voices aliases, `instruct` handling, and expressive-tag behavior green.

Run: `python -m py_compile app/server.py app/ws_handler.py app/conversation_ws.py app/assistant_backends.py app/text_sanitizer.py`
Expected: exit code 0 with no output.

- [ ] **Step 3: Commit**

```bash
git add app/server.py app/ws_handler.py app/conversation_ws.py app/assistant_backends.py app/text_sanitizer.py tests/test_ws_handler.py tests/test_conversation_ws.py README.md compose.yaml
git commit -m "feat: rename voices surface and add websocket voice design"
```
