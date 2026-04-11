# Compose Secrets To Env Design

## Goal

Remove hardcoded secrets from `compose.yaml` and move them into the local `.env` workflow so the tracked Compose file can be pushed safely.

## Current State

The project already ignores `.env` in `.gitignore` and ships `.env.example`, but `compose.yaml` still contains at least one real secret inline (`HF_TOKEN`).

That creates an avoidable risk: a normal `git push` from the main worktree can accidentally publish local secrets if they remain embedded in tracked files.

## Scope

Included:

- replace inline secret values in `compose.yaml` with `${...}` references
- update `.env.example` to document the supported secret variables
- update README so the local secret workflow is clear

Excluded:

- Docker secrets
- external secret managers
- restructuring all non-secret environment variables out of `compose.yaml`

## Recommended Approach

Use Docker Compose’s standard `.env` interpolation flow.

Tracked files should contain only variable references such as `${HF_TOKEN}` and `${OMNIVOICE_API_KEY}`. Real secret values live only in the ignored `.env` file.

This is the smallest correct change and matches the project’s current conventions because `.env` is already ignored and `.env.example` already exists.

## Alternatives Considered

### 1. Keep secrets inline in `compose.yaml`

This is the current unsafe state and should not continue.

### 2. Use a separate `.env.local` or `env_file`

This would also work, but it adds complexity without solving a problem the default `.env` flow cannot already handle.

### 3. Use Docker secrets immediately

This is more robust for production, but heavier than needed for the current local/operator workflow.

## Configuration Changes

### `compose.yaml`

Replace inline secret values with variable expansion:

- `HF_TOKEN=${HF_TOKEN}`
- `OMNIVOICE_API_KEY=${OMNIVOICE_API_KEY}` when that line is enabled or documented

Non-secret values remain directly in `compose.yaml`.

### `.env`

The real local file should contain secret values only on the developer/operator machine.

Example:

```dotenv
HF_TOKEN=hf_xxx_local_secret
OMNIVOICE_API_KEY=change-me-if-you-want-auth
```

### `.env.example`

The tracked template should include the variable names with empty or placeholder values and comments explaining when each one matters.

## Runtime Behavior

### `HF_TOKEN`

Optional.

If present, Hugging Face downloads can authenticate against gated or rate-limited resources.

If absent, startup still works when:

- the model is already available locally, or
- the model is publicly accessible without authentication

### `OMNIVOICE_API_KEY`

Optional.

If present, the existing API-key auth path is enabled for the HTTP endpoints that currently support it.

If absent, auth remains disabled, matching the current default behavior.

## Documentation Changes

README should explain the standard workflow clearly:

1. copy `.env.example` to `.env`
2. fill in secret values locally
3. run `docker compose up -d --build`

It should also make it explicit that:

- `.env` is local and not committed
- `compose.yaml` is safe to commit because it contains only variable references, not secrets

## Testing Strategy

Add light verification only:

- confirm `.env` remains ignored by git
- confirm `compose.yaml` no longer contains real secret literals
- confirm `.env.example` includes the documented variables

No runtime code-path changes are expected beyond Compose environment resolution.

## Risks And Mitigations

### Missing `.env`

Risk:

- operators may forget to create `.env`

Mitigation:

- make README and `.env.example` explicit
- keep non-secret defaults in `compose.yaml`

### Empty secret expansion

Risk:

- Compose expands `${VAR}` to empty string if the variable is missing

Mitigation:

- document which secrets are optional
- avoid treating missing `HF_TOKEN` or `OMNIVOICE_API_KEY` as fatal unless the runtime truly requires them

## Success Criteria

- `compose.yaml` contains no hardcoded secrets.
- `.env.example` documents `HF_TOKEN` and `OMNIVOICE_API_KEY` clearly.
- README tells operators to use `.env` for local secrets.
- The tracked repo can be pushed without exposing local secret values from Compose configuration.
