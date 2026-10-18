"""One model call, streamed, boiled down to a plain result.

Why stream? A big reasoning model can think for a long while, and a streaming
connection keeps sending bytes instead of sitting silent until something times
out. But the agent loop doesn't want chunks. It wants to know "what did the
model say, and which tools did it ask for?". That's this file's whole job.

Streamed tool calls arrive in pieces: the first chunk has the call id and the
function name, then the JSON arguments dribble in as fragments. We glue them
back together by index.
"""
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import openai

# HTTP statuses worth waiting out: timeouts, conflicts, rate limits, server-side trouble.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
# Errors that arrive *inside* a stream have no status code, so we go by the message.
BUSY_WORDS = ("overload", "temporarily", "try again", "capacity")

# How the "thinking" setting maps to the chat-template flags in NVIDIA's model card.
# "medium" asks for shorter reasoning (fewer tokens, so faster and kinder to rate limits).
THINKING_MODES = {
    "on": {"enable_thinking": True},
    "medium": {"enable_thinking": True, "medium_effort": True},
    "off": {"enable_thinking": False},
}


@dataclass(frozen=True)
class ModelSettings:
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 16384
    thinking: Optional[str] = "on"  # "on" | "medium" | "off" | None (don't send the flag at all)
    retries: int = 4                # extra attempts when the service is busy (waits 5s, 15s, 45s, 120s)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # still a JSON string; tools.run_tool validates it


@dataclass
class ModelReply:
    content: str
    reasoning: str
    tool_calls: list
    finish_reason: Optional[str]


def assemble(chunks, on_reasoning: Optional[Callable[[str], None]] = None) -> ModelReply:
    content, reasoning = [], []
    slots: dict = {}          # tool-call index -> {"id", "name", "arguments"}
    last_idx = None
    finish_reason = None

    for chunk in chunks:
        if not getattr(chunk, "choices", None):   # usage-only chunks have no choices
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason

        piece = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if piece:
            reasoning.append(piece)
            if on_reasoning:
                on_reasoning(piece)

        text = getattr(delta, "content", None)
        if text:
            content.append(text)

        for tc in getattr(delta, "tool_calls", None) or []:
            idx = getattr(tc, "index", None)
            if idx is None:
                # no index: a new id means a new call, otherwise it continues the last one
                idx = last_idx if (last_idx is not None and not getattr(tc, "id", None)) else len(slots)
            last_idx = idx
            slot = slots.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None) and not slot["name"]:
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["arguments"] += fn.arguments

    calls = [
        ToolCall(id=s["id"] or f"call_{i}", name=s["name"], arguments=s["arguments"])
        for i, s in sorted(slots.items())
    ]
    return ModelReply("".join(content), "".join(reasoning), calls, finish_reason)


def is_retryable(exc: Exception) -> bool:
    """Is this the 'service is busy, try again' kind of failure? Bad keys, unknown
    models and malformed requests are NOT retryable: waiting won't fix them."""
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS
    if isinstance(exc, openai.APIConnectionError):   # includes timeouts
        return True
    if isinstance(exc, openai.APIError):             # an error event in the middle of a stream
        return any(word in str(exc).lower() for word in BUSY_WORDS)
    return False


def _retry_after(exc: Exception) -> Optional[float]:
    """If the server told us how long to wait, believe it."""
    try:
        return float(exc.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def _delay_for(attempt: int, exc: Exception) -> float:
    delay = min(5 * 3 ** attempt, 120) * random.uniform(0.8, 1.2)   # 5, 15, 45, 120 seconds, give or take
    hinted = _retry_after(exc)
    return max(delay, hinted) if hinted else delay


def chat(client, *, model: str, messages: list, tools: list,
         settings: ModelSettings = ModelSettings(),
         on_reasoning: Optional[Callable[[str], None]] = None,
         on_retry: Optional[Callable[[int, float, Exception], None]] = None) -> ModelReply:
    kwargs = dict(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_tokens,
        stream=True,
    )
    if settings.thinking is not None:
        if settings.thinking not in THINKING_MODES:
            raise ValueError(f"thinking must be one of {list(THINKING_MODES)} or None, got {settings.thinking!r}")
        kwargs["extra_body"] = {"chat_template_kwargs": THINKING_MODES[settings.thinking]}

    # Retrying the whole call is safe: nothing has been acted on until we have a full reply.
    # (If a stream breaks halfway, on_reasoning may replay a little text. That's fine.)
    for attempt in range(settings.retries + 1):
        try:
            return assemble(client.chat.completions.create(**kwargs), on_reasoning)
        except openai.APIError as exc:
            if attempt >= settings.retries or not is_retryable(exc):
                raise
            delay = _delay_for(attempt, exc)
            if on_retry:
                on_retry(attempt + 1, delay, exc)
            time.sleep(delay)
