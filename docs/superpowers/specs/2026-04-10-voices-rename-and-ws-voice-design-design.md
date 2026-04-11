# Voices Rename And WebSocket Voice Design Design

## Goal

Rename the project’s public voice assets surface from `samples` to `voices`, support inferred voice design via WebSocket when no explicit voice sample is provided, and allow the assistant to use a constrained set of non-verbal expression tags for more expressive spoken output.

## Current State

The repository currently uses `samples` as the public term for:

- the bind-mounted voice asset directory
- the `GET /samples` and `POST /samples/reload` endpoints
- the `OMNIVOICE_SAMPLES_DIR` environment variable
- most README examples

At the protocol level, voice cloning is selected through the `sample` field, while voice-design instructions are available in the REST API but are not yet fully exposed as a first-class path in the duplex WebSocket flow when no explicit voice sample is chosen.

The assistant is currently constrained to plain text, but it does not yet have an allowlist of non-verbal expression tags that can safely pass through to TTS.

## Scope

Included:

- rename the public directory/config/API surface from `samples` to `voices`
- add backward-compatible aliases for the old `samples` endpoints and environment variable
- support `instruct` on WebSocket flows when no explicit voice sample is chosen
- define clear precedence between `sample` and `instruct`
- allow a constrained set of expressive non-verbal tags to pass through the assistant pipeline
- update README and related documentation accordingly

Excluded:

- renaming the request payload field `sample` to `voice` in this slice
- introducing arbitrary user-defined expression tags
- changing the existing TTS model internals

## Recommended Approach

Keep `voices` as the new canonical public name for filesystem, config, and native HTTP endpoints, while retaining compatibility aliases for `samples` during a transition period.

For WebSocket requests, support both `sample` and `instruct` but keep `sample` as the higher-priority selector. For assistant output, expand the current plain-text policy so that only a strict allowlist of expressive tags survives sanitization and is documented as supported behavior.

## Alternatives Considered

### 1. Hard rename with no aliases

This is cleanest on paper, but it would break existing scripts, compose setups, and operational habits immediately.

### 2. Rename payload fields too in the same step

This would improve naming consistency, but it expands the blast radius into all request/WS clients at once and is not necessary to deliver the directory/API rename.

### 3. Keep `samples` publicly and rename only the filesystem

This avoids migrations but leaves the project inconsistent and makes the new `voices/` directory harder to reason about.

## Public Surface Rename

### Canonical names

New canonical names:

- directory: `voices/`
- env var: `OMNIVOICE_VOICES_DIR`
- HTTP endpoints:
  - `GET /voices`
  - `POST /voices/reload`

### Compatibility aliases

During the transition period, keep these aliases working:

- `GET /samples`
- `POST /samples/reload`
- `OMNIVOICE_SAMPLES_DIR`

Precedence rule:

- if both `OMNIVOICE_VOICES_DIR` and `OMNIVOICE_SAMPLES_DIR` are set, `OMNIVOICE_VOICES_DIR` wins

The old names should be treated as deprecated in documentation and may emit warnings in logs.

## Voice Selection Rules

### Existing payload field

Keep the existing payload field `sample` unchanged in this slice for compatibility.

This means current clients do not need to change their request schema immediately.

### New WebSocket behavior

For both `/ws/tts` and `/ws/conversation`:

- if `sample` is present and valid, use voice cloning
- else if `instruct` is present and non-empty, use inferred voice design
- else fall back to existing auto-voice behavior or whichever behavior is already defined for that path

Precedence rule:

- `sample` overrides `instruct` if both are present

This keeps voice cloning deterministic while making inferred voice design available over WebSocket without forcing a new payload schema.

## Expression Tags

### Allowed tags

The assistant may emit only the following non-verbal tags:

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

### Prompt policy

The assistant system prompt should be updated to say:

- plain text only still applies
- markdown/JSON/bullets remain forbidden
- the allowlisted expression tags above are permitted
- use them only when they improve spoken naturalness
- do not overuse them
- keep responses concise and suitable for phone conversation

### Sanitization rule

The sanitization layer must preserve these exact allowlisted tags while still stripping unrelated formatting noise.

Unknown tags should be removed or normalized away rather than passed through blindly.

## Code Structure

### Server config and scanning

Update config loading and scanning so the server reads from `voices/` through `OMNIVOICE_VOICES_DIR` by default.

The sample/voice registry can stay structurally similar internally, but public docs and logs should use `voice` or `voices` terminology where practical.

### Endpoint exposure

Expose canonical `voices` endpoints and keep `samples` aliases pointing to the same underlying handlers.

### WebSocket handlers

Update WebSocket message handling so both `sample` and `instruct` are threaded into the existing generation path. The handler should select voice cloning or inferred voice design based on the precedence rule above.

### Assistant governance

Update the current assistant governance layer so expressive tags are treated as valid output, but only from the allowlist.

## Testing Strategy

Add tests for:

- canonical `GET /voices` and `POST /voices/reload`
- compatibility behavior for `GET /samples` and `POST /samples/reload`
- env var precedence when both old and new directory variables are present
- `/ws/tts` with `instruct` and no `sample`
- `/ws/conversation` with `instruct` and no `sample`
- precedence when both `sample` and `instruct` are present
- sanitization preserving allowed expressive tags while stripping disallowed formatting
- assistant prompt containing the expression-tag policy

## Risks And Mitigations

### Mixed terminology during migration

Risk:

- code, docs, and operator habits could drift between `samples` and `voices`

Mitigation:

- define `voices` as canonical everywhere user-facing
- keep aliases explicit and temporary

### Overexpressive assistant output

Risk:

- the LLM may spam expressive tags

Mitigation:

- strong prompt guidance
- preserve only a strict allowlist
- keep responses short

### Voice selection ambiguity

Risk:

- users may not know which wins when both `sample` and `instruct` are present

Mitigation:

- document the precedence rule clearly
- test it explicitly

## Success Criteria

- The repo works with `voices/` as the canonical public voice directory.
- `GET /voices` and `POST /voices/reload` work, while `samples` aliases still function.
- WebSocket requests can use `instruct` for voice design when no explicit sample is given.
- `sample` remains higher priority than `instruct` when both are present.
- The assistant can emit only the allowlisted expressive tags, and those tags survive sanitization.
- README and related documentation consistently describe the new `voices` terminology and WebSocket voice-design behavior.
