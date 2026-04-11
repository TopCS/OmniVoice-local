# Compose Secrets To Env Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove hardcoded secrets from `compose.yaml` and move them to the local `.env` workflow so the tracked Compose file is safe to push.

**Architecture:** Keep all non-secret runtime defaults in `compose.yaml`, replace inline secret values with `${...}` variable references, and use `.env.example` plus README instructions to define the expected local workflow. The repo stays safe to commit; real secret values stay only in ignored `.env`.

**Tech Stack:** Docker Compose environment interpolation, `.env`, `.env.example`, README

---

## File Structure

- Modify: `compose.yaml`
  Replace inline secret literals with `${HF_TOKEN}` and `${OMNIVOICE_API_KEY}` references.

- Modify: `.env.example`
  Document both supported secret variables and when they matter.

- Modify: `README.md`
  Explain the `.env` workflow clearly in the Quick Start / setup section.

### Task 1: Remove Secrets From `compose.yaml`

**Files:**
- Modify: `compose.yaml`
- Test: `compose.yaml`

- [ ] **Step 1: Write the failing verification check for hardcoded secrets**

Run: `python3 - <<'PY'
from pathlib import Path
text = Path('compose.yaml').read_text()
assert 'HF_TOKEN=hf_' not in text
assert '${HF_TOKEN}' in text
PY`
Expected: FAIL because `compose.yaml` still contains a real `HF_TOKEN` value.

- [ ] **Step 2: Replace inline secret values with environment references**

```yaml
environment:
  - HF_HOME=/model
  - HF_TOKEN=${HF_TOKEN}
  - OMNIVOICE_MODEL=k2-fsa/OmniVoice
  ...
  # Uncomment to enable API key authentication:
  # - OMNIVOICE_API_KEY=${OMNIVOICE_API_KEY}
```

Keep non-secret values unchanged.

- [ ] **Step 3: Re-run the secret-literal verification**

Run: `python3 - <<'PY'
from pathlib import Path
text = Path('compose.yaml').read_text()
assert 'HF_TOKEN=hf_' not in text
assert '${HF_TOKEN}' in text
assert '${OMNIVOICE_API_KEY}' in text
PY`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add compose.yaml
git commit -m "chore: move compose secrets to env vars"
```

### Task 2: Update `.env.example` For Local Secret Workflow

**Files:**
- Modify: `.env.example`
- Test: `.env.example`

- [ ] **Step 1: Write the failing content check for `.env.example`**

Run: `python3 - <<'PY'
from pathlib import Path
text = Path('.env.example').read_text()
assert 'HF_TOKEN=' in text
assert 'OMNIVOICE_API_KEY=' in text
PY`
Expected: FAIL because `.env.example` only documents `OMNIVOICE_API_KEY` today.

- [ ] **Step 2: Add the missing variables and comments**

```dotenv
# Hugging Face access (optional)
# Needed when downloading gated or rate-limited Hugging Face assets.
HF_TOKEN=

# Authentication (optional)
# When set, supported HTTP API endpoints require:
#   Authorization: Bearer <your-key>
OMNIVOICE_API_KEY=
```

- [ ] **Step 3: Re-run the `.env.example` content check**

Run: `python3 - <<'PY'
from pathlib import Path
text = Path('.env.example').read_text()
assert 'HF_TOKEN=' in text
assert 'OMNIVOICE_API_KEY=' in text
PY`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add .env.example
git commit -m "docs: document local compose secrets"
```

### Task 3: Update README Setup Instructions

**Files:**
- Modify: `README.md`
- Test: `README.md`

- [ ] **Step 1: Write the failing README content check**

Run: `python3 - <<'PY'
from pathlib import Path
text = Path('README.md').read_text()
assert 'cp .env.example .env' in text
assert 'HF_TOKEN' in text
assert '.env is local and not committed' in text
PY`
Expected: FAIL because the README does not yet clearly explain the local secret workflow.

- [ ] **Step 2: Update the Quick Start and setup notes**

```markdown
# 2. Configure local secrets (optional but recommended)
cp .env.example .env
# Edit .env to set HF_TOKEN if you need Hugging Face auth,
# and OMNIVOICE_API_KEY if you want HTTP API authentication.

... later in setup docs ...

`.env` is local and ignored by git. `compose.yaml` contains only variable references for secrets, so it is safe to commit.
```

- [ ] **Step 3: Re-run the README content check**

Run: `python3 - <<'PY'
from pathlib import Path
text = Path('README.md').read_text()
assert 'cp .env.example .env' in text
assert 'HF_TOKEN' in text
assert '.env` is local and ignored by git' in text or '.env is local and ignored by git' in text
PY`
Expected: PASS.

- [ ] **Step 4: Final focused verification**

Run: `python3 - <<'PY'
from pathlib import Path
compose = Path('compose.yaml').read_text()
env_example = Path('.env.example').read_text()
readme = Path('README.md').read_text()
assert 'HF_TOKEN=hf_' not in compose
assert '${HF_TOKEN}' in compose
assert 'HF_TOKEN=' in env_example
assert 'OMNIVOICE_API_KEY=' in env_example
assert 'cp .env.example .env' in readme
PY`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md compose.yaml .env.example
git commit -m "docs: move compose secrets to env workflow"
```
