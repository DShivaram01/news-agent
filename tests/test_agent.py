"""Offline tests for the loop: a scripted fake client plays the model, so we can
check behaviour without a network, a key, or a flaky free-tier endpoint."""
import json

import pytest

import tools as toolbox
from agent import run_agent
from fakes import FakeClient, call, spec
from llm import ModelSettings
from tracing import Tracer


@pytest.fixture
def tracer(tmp_path):
    return Tracer(tmp_path)


def run(client, tracer, max_steps=5, **kwargs):
    return run_agent(client, "fake-model", "sys", "hello", max_steps, tracer, **kwargs)


# --- the loop ---------------------------------------------------------------

def test_tool_call_then_final_answer(tracer):
    client = FakeClient([
        spec(calls=[call("get_time", '{"timezone": "UTC"}', id="abc")]),
        spec(content="It is some time in UTC."),
    ])
    result = run(client, tracer)

    assert result.stopped == "done"
    assert result.final == "It is some time in UTC."
    assert result.steps == 2

    # second request must contain: assistant tool_calls, then the matching tool result
    second = client.sent[1]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "abc"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "abc"
    assert "iso" in json.loads(second[-1]["content"])


def test_answer_without_any_tool(tracer):
    result = run(FakeClient([spec(content="Hi there.")]), tracer)
    assert (result.stopped, result.final, result.steps) == ("done", "Hi there.", 1)


def test_two_tool_calls_in_one_reply_both_get_results(tracer):
    client = FakeClient([
        spec(calls=[
            call("get_time", '{"timezone": "America/Chicago"}', id="a"),
            call("get_time", '{"timezone": "Asia/Kolkata"}', id="b"),
        ]),
        spec(content="done"),
    ])
    run(client, tracer)
    tool_msgs = [m for m in client.sent[1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]
    assert "-05:00" in json.loads(tool_msgs[0]["content"])["iso"] or "-06:00" in json.loads(tool_msgs[0]["content"])["iso"]
    assert json.loads(tool_msgs[1]["content"])["iso"].endswith("+05:30")


def test_step_cap_stops_a_runaway_model(tracer):
    looping = spec(calls=[call("get_time", '{"timezone": "UTC"}')])
    client = FakeClient([looping] * 10)
    result = run(client, tracer, max_steps=3)
    assert result.stopped == "max_steps"
    assert result.final is None
    assert len(client.sent) == 3  # it must not make a 4th call


def test_bad_tool_calls_become_errors_not_crashes(tracer):
    client = FakeClient([
        spec(calls=[
            call("no_such_tool", "{}", id="1"),
            call("get_time", "{not json", id="2"),
            call("get_time", '{"timezone": "Mars/Olympus"}', id="3"),
            call("get_time", '{"timezone": 5}', id="4"),
        ]),
        spec(content="sorry about that"),
    ])
    result = run(client, tracer)
    assert result.stopped == "done"
    tool_msgs = [json.loads(m["content"]) for m in client.sent[1] if m["role"] == "tool"]
    assert len(tool_msgs) == 4
    assert all("error" in m for m in tool_msgs)


def test_tool_call_written_as_plain_text_is_flagged(tracer):
    text = '{"name": "get_time", "parameters": {"timezone": "UTC"}}'
    result = run(FakeClient([spec(content=text)]), tracer)
    assert result.stopped == "malformed_tool_call"
    assert result.final is None


def test_plain_json_answer_is_not_mistaken_for_a_tool_call(tracer):
    result = run(FakeClient([spec(content='{"answer": 42}')]), tracer)
    assert result.stopped == "done"


def test_empty_reply_is_reported(tracer):
    result = run(FakeClient([spec(content="   ")]), tracer)
    assert result.stopped == "empty_response"


def test_thinking_only_reply_counts_as_empty(tracer):
    result = run(FakeClient([spec(reasoning="hmm, let me think about this")]), tracer)
    assert result.stopped == "empty_response"


def test_running_out_of_tokens_is_reported_as_truncated(tracer):
    result = run(FakeClient([spec(reasoning="thinking...", content="The answ", finish_reason="length")]), tracer)
    assert result.stopped == "truncated"
    assert result.final is None


# --- reasoning, settings, trace -----------------------------------------------

def test_reasoning_is_traced_but_never_sent_back(tracer):
    client = FakeClient([
        spec(reasoning="I should check the clock.", calls=[call("get_time", '{"timezone": "UTC"}')]),
        spec(content="ok"),
    ])
    run(client, tracer)
    for message in client.sent[1]:
        assert "reasoning" not in message and "reasoning_content" not in message
    records = [json.loads(line) for line in tracer.path.read_text().splitlines()]
    assert records[1]["reasoning"] == "I should check the clock."


def test_reasoning_callback_receives_the_stream(tracer):
    seen = []
    run(FakeClient([spec(reasoning="abcdef", content="ok")]), tracer, on_reasoning=seen.append)
    assert "".join(seen) == "abcdef"
    assert len(seen) > 1  # it arrived in pieces


def test_settings_reach_the_request(tracer):
    client = FakeClient([spec(content="ok")])
    run(client, tracer, settings=ModelSettings(temperature=0.6, top_p=0.9, max_tokens=123, thinking="medium"))
    req = client.requests[0]
    assert req["stream"] is True
    assert req["temperature"] == 0.6 and req["top_p"] == 0.9 and req["max_tokens"] == 123
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True}}
    assert req["tool_choice"] == "auto"
    assert req["tools"][0]["function"]["name"] == "get_time"


def test_trace_file_records_the_run(tracer):
    run(FakeClient([spec(content="hi")]), tracer)
    kinds = [json.loads(line)["kind"] for line in tracer.path.read_text().splitlines()]
    assert kinds == ["start", "model", "stop"]


# --- the tool itself ----------------------------------------------------------

def test_get_time_valid_timezone():
    out = json.loads(toolbox.run_tool("get_time", '{"timezone": "Asia/Kolkata"}'))
    assert out["timezone"] == "Asia/Kolkata"
    assert out["iso"].endswith("+05:30")


def test_get_time_defaults_to_utc_when_args_empty():
    out = json.loads(toolbox.run_tool("get_time", ""))
    assert out["timezone"] == "UTC"


@pytest.mark.parametrize("bad", ["", "America", "../etc/passwd", "Not/AZone"])
def test_get_time_rejects_bad_timezones_cleanly(bad):
    out = json.loads(toolbox.run_tool("get_time", json.dumps({"timezone": bad})))
    assert "error" in out


def test_schemas_match_registry():
    names = [s["function"]["name"] for s in toolbox.tool_schemas()]
    assert names == list(toolbox.TOOLS)
