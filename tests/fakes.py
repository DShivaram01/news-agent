"""A fake NVIDIA client for offline tests.

It plays scripted replies back as a *stream of small chunks*, the way the real
API does: reasoning in pieces, content in pieces, tool-call arguments dribbling
in a few characters at a time, then a finish chunk and a usage-only chunk. That
way every loop test also exercises the stitching code in llm.py.
"""
import json
from types import SimpleNamespace as NS


def chunk(content=None, reasoning=None, tool_calls=None, finish_reason=None):
    delta = NS(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    return NS(choices=[NS(delta=delta, finish_reason=finish_reason)])


def tc_chunk(index, id=None, name=None, arguments=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=arguments))


USAGE_CHUNK = NS(choices=[])  # the real API sometimes ends with a chunk that has no choices


def call(name, arguments, id="call_1"):
    return (id, name, arguments)


def spec(content="", calls=(), reasoning="", finish_reason=None):
    return {"content": content, "calls": list(calls), "reasoning": reasoning, "finish_reason": finish_reason}


def chunks_for(s):
    out = []
    r = s["reasoning"]
    if r:
        mid = len(r) // 2
        out += [chunk(reasoning=r[:mid]), chunk(reasoning=r[mid:])]
    c = s["content"]
    for i in range(0, len(c), 4):
        out.append(chunk(content=c[i:i + 4]))
    for idx, (id, name, args) in enumerate(s["calls"]):
        out.append(chunk(tool_calls=[tc_chunk(idx, id=id, name=name, arguments="")]))
        for i in range(0, len(args), 5):
            out.append(chunk(tool_calls=[tc_chunk(idx, arguments=args[i:i + 5])]))
    out.append(chunk(finish_reason=s["finish_reason"] or ("tool_calls" if s["calls"] else "stop")))
    out.append(USAGE_CHUNK)
    return out


class MidStreamError:
    """Script item: start streaming a little text, then blow up with this exception."""

    def __init__(self, exc):
        self.exc = exc

    def stream(self):
        yield chunk(content="par")
        raise self.exc


class FakeClient:
    """Hands back scripted replies in order and remembers what it was sent.
    A script item can also be an Exception (raised on the request, like an HTTP error)
    or a MidStreamError (raised partway through the stream)."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.sent = []       # snapshot of `messages` for each request
        self.requests = []   # every other keyword argument for each request
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({k: v for k, v in kwargs.items() if k != "messages"})
        self.sent.append(json.loads(json.dumps(kwargs["messages"])))
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, MidStreamError):
            return item.stream()
        return iter(chunks_for(item))
