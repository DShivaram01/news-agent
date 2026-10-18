"""End-to-end check with the *real* openai client against a tiny local server that
streams server-sent events the way NVIDIA's endpoint does. Still offline (localhost
only), but it proves the real library's chunk objects work with our code: reasoning
fields surfacing, tool-call fragments, and extra_body reaching the wire."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import OpenAI

import llm
from agent import run_agent
from llm import ModelSettings
from tracing import Tracer


def _event(delta, finish=None):
    return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close the connection when done, so no Content-Length needed

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)

        if self.server.fail_first > 0:     # pretend to be a swamped free-tier endpoint
            self.server.fail_first -= 1
            payload = json.dumps({"error": {"message": "Service temporarily overloaded"}}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        got_tool_result = any(m["role"] == "tool" for m in body["messages"])

        if not got_tool_result:
            args = '{"timezone": "Asia/Kolkata"}'
            events = [
                _event({"role": "assistant", "reasoning_content": "Need the time. "}),
                _event({"reasoning_content": "Call the tool."}),
                _event({"tool_calls": [{"index": 0, "id": "call_9", "type": "function",
                                        "function": {"name": "get_time", "arguments": ""}}]}),
                _event({"tool_calls": [{"index": 0, "function": {"arguments": args[:12]}}]}),
                _event({"tool_calls": [{"index": 0, "function": {"arguments": args[12:]}}]}),
                _event({}, finish="tool_calls"),
            ]
        else:
            events = [_event({"content": "It is "}), _event({"content": "evening in Kolkata."}),
                      _event({}, finish="stop")]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for e in events:
            self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests = []
    srv.fail_first = 0
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_real_client_full_run(server, tmp_path):
    client = OpenAI(base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="test", max_retries=0)
    tracer = Tracer(tmp_path)
    seen = []

    result = run_agent(client, "nvidia/some-model", "sys", "what time is it in Kolkata?", 4, tracer,
                       settings=ModelSettings(thinking="on"), on_reasoning=seen.append)

    assert (result.stopped, result.steps) == ("done", 2)
    assert result.final == "It is evening in Kolkata."
    assert "".join(seen) == "Need the time. Call the tool."

    first, second = server.requests
    assert first["stream"] is True
    assert first["chat_template_kwargs"] == {"enable_thinking": True}   # extra_body is merged into the body
    assert first["tools"][0]["function"]["name"] == "get_time"

    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_9"
    assert json.loads(tool_msg["content"])["iso"].endswith("+05:30")


def test_real_client_waits_out_a_real_503(server, tmp_path, monkeypatch):
    server.fail_first = 2
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)   # don't actually wait
    client = OpenAI(base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="test", max_retries=0)
    seen = []

    result = run_agent(client, "m", "sys", "what time is it in Kolkata?", 4, Tracer(tmp_path),
                       settings=ModelSettings(retries=3), on_retry=lambda a, d, e: seen.append(str(e)))

    assert (result.stopped, result.steps) == ("done", 2)
    assert len(waits) == 2
    assert all("temporarily overloaded" in s for s in seen)    # the message you saw in PowerShell
    assert len(server.requests) == 4                           # 2 rejected, then 2 real steps
