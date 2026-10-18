"""What happens when NVIDIA says 'busy'? We wait and retry. When it says 'your key is bad'? We don't."""
import json

import openai
import pytest

try:                      # openai 3.x ships its own http library; 1.x and 2.x use httpx
    import httpx2 as httpx
except ImportError:
    import httpx

import llm
from agent import run_agent
from fakes import FakeClient, MidStreamError, spec
from llm import ModelSettings
from tracing import Tracer

REQUEST = httpx.Request("POST", "http://example.test/v1/chat/completions")


def status_error(cls, code, message="boom", headers=None):
    response = httpx.Response(code, request=REQUEST, headers=headers or {})
    return cls(message, response=response, body=None)


def overloaded():
    return status_error(openai.InternalServerError, 503, "Service temporarily overloaded")


@pytest.fixture
def sleeps(monkeypatch):
    """Record waits instead of actually sleeping."""
    waited = []
    monkeypatch.setattr(llm.time, "sleep", waited.append)
    return waited


def call_chat(client, retries=4, **kwargs):
    return llm.chat(client, model="m", messages=[], tools=[], settings=ModelSettings(retries=retries), **kwargs)


# --- classification --------------------------------------------------------------

@pytest.mark.parametrize("exc, expected", [
    (status_error(openai.InternalServerError, 503), True),
    (status_error(openai.InternalServerError, 502), True),
    (status_error(openai.RateLimitError, 429), True),
    (status_error(openai.APIStatusError, 529), True),
    (openai.APIConnectionError(request=REQUEST), True),
    (openai.APITimeoutError(request=REQUEST), True),
    (openai.APIError("Service temporarily overloaded", request=REQUEST, body=None), True),
    (status_error(openai.AuthenticationError, 401), False),
    (status_error(openai.NotFoundError, 404), False),
    (status_error(openai.BadRequestError, 400), False),
    (openai.APIError("messages must not be empty", request=REQUEST, body=None), False),
    (ValueError("nope"), False),
])
def test_which_errors_are_worth_retrying(exc, expected):
    assert llm.is_retryable(exc) is expected


# --- behaviour -------------------------------------------------------------------

def test_busy_then_success(sleeps):
    client = FakeClient([overloaded(), overloaded(), spec(content="finally")])
    seen = []
    reply = call_chat(client, on_retry=lambda attempt, delay, exc: seen.append(attempt))
    assert reply.content == "finally"
    assert len(client.requests) == 3
    assert seen == [1, 2]
    assert len(sleeps) == 2


def test_waits_grow_between_attempts(sleeps):
    client = FakeClient([overloaded()] * 4 + [spec(content="ok")])
    call_chat(client)
    assert 4 <= sleeps[0] <= 6          # ~5s
    assert 12 <= sleeps[1] <= 18        # ~15s
    assert 36 <= sleeps[2] <= 54        # ~45s
    assert 96 <= sleeps[3] <= 144       # ~120s


def test_gives_up_after_the_configured_number_of_retries(sleeps):
    client = FakeClient([overloaded()] * 10)
    with pytest.raises(openai.InternalServerError):
        call_chat(client, retries=2)
    assert len(client.requests) == 3     # first try + 2 retries
    assert len(sleeps) == 2


def test_zero_retries_means_fail_immediately(sleeps):
    client = FakeClient([overloaded()] * 3)
    with pytest.raises(openai.InternalServerError):
        call_chat(client, retries=0)
    assert len(client.requests) == 1 and sleeps == []


def test_bad_key_fails_at_once_without_waiting(sleeps):
    client = FakeClient([status_error(openai.AuthenticationError, 401)] * 3)
    with pytest.raises(openai.AuthenticationError):
        call_chat(client)
    assert len(client.requests) == 1 and sleeps == []


def test_retry_after_header_is_respected(sleeps):
    err = status_error(openai.RateLimitError, 429, headers={"retry-after": "77"})
    call_chat(FakeClient([err, spec(content="ok")]))
    assert sleeps == [77.0]


def test_overload_error_in_the_middle_of_a_stream_is_retried(sleeps):
    mid = openai.APIError("Service temporarily overloaded", request=REQUEST, body=None)
    client = FakeClient([MidStreamError(mid), spec(content="whole answer")])
    reply = call_chat(client)
    assert reply.content == "whole answer"     # the half-finished attempt left no trace
    assert len(client.requests) == 2


def test_connection_drop_is_retried(sleeps):
    client = FakeClient([openai.APIConnectionError(request=REQUEST), spec(content="ok")])
    assert call_chat(client).content == "ok"


# --- inside the agent --------------------------------------------------------------

def test_agent_survives_a_busy_spell_and_traces_it(tmp_path, sleeps):
    tracer = Tracer(tmp_path)
    client = FakeClient([overloaded(), spec(content="hello")])
    waits = []
    result = run_agent(client, "m", "sys", "hi", 3, tracer,
                       settings=ModelSettings(retries=3), on_retry=lambda a, d, e: waits.append(a))
    assert (result.stopped, result.final) == ("done", "hello")
    assert waits == [1]
    records = [json.loads(line) for line in tracer.path.read_text().splitlines()]
    retry = next(r for r in records if r["kind"] == "retry")
    assert retry["step"] == 1 and retry["attempt"] == 1 and "overloaded" in retry["error"]


def test_agent_raises_when_the_service_never_recovers(tmp_path, sleeps):
    client = FakeClient([overloaded()] * 10)
    with pytest.raises(openai.InternalServerError):
        run_agent(client, "m", "sys", "hi", 3, Tracer(tmp_path), settings=ModelSettings(retries=1))
