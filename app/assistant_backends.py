from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


PHONE_AGENT_SYSTEM_PROMPT = (
    "You are a phone agent speaking with a caller. "
    "Be calm, direct, and reassuring. "
    "Keep replies concise, practical, and easy to say out loud. "
    "Do not repeat the caller's words back unless needed for clarity. "
    "Ask at most one short clarifying question when necessary. "
    "Output plain text only. "
    "Do not use markdown, bullet points, emojis, JSON, XML, code blocks, "
    "speaker labels, or stage directions."
)


@runtime_checkable
class AssistantBackend(Protocol):
    async def generate_response(
        self,
        transcript: str,
        *,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        language_hint: str | None = None,
    ) -> str: ...


class OllamaAssistantBackend:
    def __init__(
        self,
        client: Any,
        model: str,
        *,
        system_prompt: str = PHONE_AGENT_SYSTEM_PROMPT,
    ):
        self._client = client
        self._model = model
        self._system_prompt = system_prompt

    async def generate_response(
        self,
        transcript: str,
        *,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        language_hint: str | None = None,
    ) -> str:
        response = await self._client.chat(
            model=self._model,
            messages=_build_messages(
                self._system_prompt,
                str(transcript).strip(),
                session_id=session_id,
                history=history or [],
                language_hint=language_hint,
            ),
            stream=False,
        )
        return _extract_message_text(response)


def _build_messages(
    system_prompt: str,
    transcript: str,
    *,
    session_id: str | None,
    history: list[dict[str, str]],
    language_hint: str | None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]

    context_parts = []
    if session_id:
        context_parts.append(f"session_id={session_id}")
    if language_hint:
        context_parts.append(f"caller_language_hint={language_hint}")
    if context_parts:
        messages.append(
            {
                "role": "system",
                "content": f"Conversation context: {'; '.join(context_parts)}.",
            }
        )

    for turn in history[-3:]:
        user_text = str(turn.get("user") or "").strip()
        assistant_text = str(turn.get("assistant") or "").strip()
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    messages.append({"role": "user", "content": transcript})
    return messages


def _extract_message_text(response: Any) -> str:
    content = None

    if isinstance(response, dict):
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
    else:
        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is None and hasattr(message, "get"):
                content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Assistant backend returned no usable message content.")

    return content.strip()
