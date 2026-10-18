"""Tests for the stream-stitching layer, using hand-built chunks for the awkward cases."""
from types import SimpleNamespace as NS

import pytest

import llm
from fakes import USAGE_CHUNK, FakeClient, chunk, spec, tc_chunk


def test_fragmented_arguments_are_glued_together():
    reply = llm.assemble([
        chunk(tool_calls=[tc_chunk(0, id="a", name="get_time", arguments="")]),
        chunk(tool_calls=[tc_chunk(0, arguments='{"timezo')]),
        chunk(tool_calls=[tc_chunk(0, arguments='ne": "UTC"}')]),
        chunk(finish_reason="tool_calls"),
    ])
    assert len(reply.tool_calls) == 1
    c = reply.tool_calls[0]
    assert (c.id, c.name, c.arguments) == ("a", "get_time", '{"timezone": "UTC"}')
    assert reply.finish_reason == "tool_calls"


def test_parallel_calls_are_kept_apart_by_index():
    reply = llm.assemble([
        chunk(tool_calls=[tc_chunk(0, id="a", name="get_time", arguments='{"timezone":')]),
        chunk(tool_calls=[tc_chunk(1, id="b", name="get_time", arguments='{"timezone": "Asia/Tokyo"}')]),
        chunk(tool_calls=[tc_chunk(0, arguments=' "UTC"}')]),
    ])
    assert [(c.id, c.arguments) for c in reply.tool_calls] == [
        ("a", '{"timezone": "UTC"}'),
        ("b", '{"timezone": "Asia/Tokyo"}'),
    ]


def test_calls_without_an_index_are_split_on_new_ids():
    reply = llm.assemble([
        chunk(tool_calls=[tc_chunk(None, id="a", name="get_time", arguments='{"timezone"')]),
        chunk(tool_calls=[tc_chunk(None, arguments=': "UTC"}')]),   # continuation: no id
        chunk(tool_calls=[tc_chunk(None, id="b", name="get_time", arguments="{}")]),
    ])
    assert [(c.id, c.arguments) for c in reply.tool_calls] == [("a", '{"timezone": "UTC"}'), ("b", "{}")]


def test_missing_call_id_gets_a_fallback():
    reply = llm.assemble([chunk(tool_calls=[tc_chunk(0, name="get_time", arguments="{}")])])
    assert reply.tool_calls[0].id == "call_0"


def test_reasoning_and_content_are_kept_separate():
    seen = []
    reply = llm.assemble(
        [chunk(reasoning="think "), chunk(reasoning="more"), chunk(content="Hel"), chunk(content="lo"),
         chunk(finish_reason="stop")],
        on_reasoning=seen.append,
    )
    assert (reply.reasoning, reply.content) == ("think more", "Hello")
    assert seen == ["think ", "more"]


def test_reasoning_field_alias_is_understood():
    delta = NS(content=None, reasoning="via alias", tool_calls=None)  # no reasoning_content attribute
    reply = llm.assemble([NS(choices=[NS(delta=delta, finish_reason=None)])])
    assert reply.reasoning == "via alias"


def test_usage_only_chunks_and_null_deltas_are_ignored():
    reply = llm.assemble([chunk(content=None), USAGE_CHUNK, chunk(content="ok"), USAGE_CHUNK])
    assert reply.content == "ok"
    assert reply.tool_calls == []
    assert reply.finish_reason is None


def test_empty_stream_gives_an_empty_reply():
    reply = llm.assemble([])
    assert (reply.content, reply.reasoning, reply.tool_calls) == ("", "", [])


# --- chat(): what we send ------------------------------------------------------

@pytest.mark.parametrize("mode, expected", [
    ("on", {"enable_thinking": True}),
    ("medium", {"enable_thinking": True, "medium_effort": True}),
    ("off", {"enable_thinking": False}),
])
def test_thinking_modes_map_to_template_flags(mode, expected):
    client = FakeClient([spec(content="hi")])
    llm.chat(client, model="m", messages=[], tools=[], settings=llm.ModelSettings(thinking=mode))
    assert client.requests[0]["extra_body"] == {"chat_template_kwargs": expected}


def test_thinking_none_sends_no_flag():
    client = FakeClient([spec(content="hi")])
    llm.chat(client, model="m", messages=[], tools=[], settings=llm.ModelSettings(thinking=None))
    assert "extra_body" not in client.requests[0]


def test_unknown_thinking_mode_fails_loudly_before_any_request():
    client = FakeClient([spec(content="hi")])
    with pytest.raises(ValueError):
        llm.chat(client, model="m", messages=[], tools=[], settings=llm.ModelSettings(thinking="loud"))
    assert client.requests == []
