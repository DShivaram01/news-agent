"""The agent loop. This is the whole idea of an agent, in about 60 lines:

    ask the model -> did it want a tool? -> run it -> show it the result -> repeat

It stops when the model answers without asking for a tool, or when we hit the
step cap (so a confused model can't loop forever and burn your rate limit).
"""
import json
from dataclasses import dataclass
from typing import Callable, Optional

import llm
import tools as toolbox
from tracing import Tracer


@dataclass
class AgentResult:
    final: Optional[str]   # the model's answer, if it gave one
    steps: int             # how many model calls we made
    stopped: str           # "done" | "max_steps" | "malformed_tool_call" | "empty_response" | "truncated"


def _looks_like_unparsed_tool_call(text: str) -> bool:
    """Some models write the tool call as plain JSON text instead of using the
    tool_calls field. NVIDIA's docs call this a parsing failure. Catch it so we
    don't mistake it for a real answer."""
    if not text.strip().startswith("{"):
        return False
    try:
        obj = json.loads(text.strip())
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and "name" in obj and ("parameters" in obj or "arguments" in obj)


def run_agent(client, model: str, system_prompt: str, user_prompt: str,
              max_steps: int, tracer: Tracer,
              settings: llm.ModelSettings = llm.ModelSettings(),
              on_reasoning: Optional[Callable[[str], None]] = None,
              on_retry: Optional[Callable[[int, float, Exception], None]] = None) -> AgentResult:
    toolbox.reset_run_state()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    tracer.log("start", {"model": model, "user_prompt": user_prompt, "max_steps": max_steps,
                         "settings": vars(settings)})

    for step in range(1, max_steps + 1):
        def note_retry(attempt, delay, exc, step=step):
            tracer.log("retry", {"step": step, "attempt": attempt, "wait_seconds": round(delay, 1),
                                 "error": str(exc)})
            if on_retry:
                on_retry(attempt, delay, exc)

        reply = llm.chat(client, model=model, messages=messages, tools=toolbox.tool_schemas(),
                         settings=settings, on_reasoning=on_reasoning, on_retry=note_retry)
        tracer.log("model", {
            "step": step,
            "content": reply.content,
            "reasoning": reply.reasoning,
            "finish_reason": reply.finish_reason,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.tool_calls],
        })

        if not reply.tool_calls:
            # Thinking models can burn their whole token budget on reasoning and
            # get cut off before they say anything useful. Report that plainly.
            if reply.finish_reason == "length":
                tracer.log("stop", {"reason": "truncated"})
                return AgentResult(None, step, "truncated")
            if _looks_like_unparsed_tool_call(reply.content):
                tracer.log("stop", {"reason": "malformed_tool_call"})
                return AgentResult(None, step, "malformed_tool_call")
            if not reply.content.strip():
                tracer.log("stop", {"reason": "empty_response"})
                return AgentResult(None, step, "empty_response")
            tracer.log("stop", {"reason": "done"})
            return AgentResult(reply.content, step, "done")

        # The assistant message that asked for the tools has to go into the
        # history *before* the tool results, or the API will reject the request.
        # (We keep the model's reasoning out of the history. It's in the trace.)
        messages.append({
            "role": "assistant",
            "content": reply.content,
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in reply.tool_calls
            ],
        })
        for c in reply.tool_calls:
            result = toolbox.run_tool(c.name, c.arguments)
            tracer.log("tool", {"id": c.id, "name": c.name, "result": result})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})

    tracer.log("stop", {"reason": "max_steps"})
    return AgentResult(None, max_steps, "max_steps")
