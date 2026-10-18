"""Tests for news.py and the fetch_news tool. The fixtures are small copies of what the
real feeds look like, quirks included (double-encoded entities, ISO dates, missing
dates, Google's publisher suffix). Time is frozen so the tests never go stale."""
import functools
import json
import threading
import urllib.error
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import news
import tools
import try_tool
from agent import run_agent
from fakes import FakeClient, call, spec
from tracing import Tracer

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 17, 3, 0, tzinfo=timezone.utc)

PRN = {"name": "PR Newswire", "market": "US", "url": "http://t/prn"}
YAHOO = {"name": "Yahoo Finance", "market": "US", "url": "http://t/yahoo"}
GOOGLE = {"name": "Google News", "market": "US", "url": "http://t/google", "source_in_title": True}
ET = {"name": "ET Stocks", "market": "IN", "url": "http://t/et"}


def fx(name):
    return (FIXTURES / name).read_bytes()


MAPPING = {
    PRN["url"]: fx("prnewswire_sample.xml"),
    YAHOO["url"]: fx("yahoo_sample.xml"),
    GOOGLE["url"]: fx("google_sample.xml"),
    ET["url"]: fx("et_sample.xml"),
}


def make_fetcher(mapping):
    def fetch(url):
        value = mapping[url]
        if isinstance(value, Exception):
            raise value
        return value
    return fetch


def run(feeds, mapping=MAPPING, **kwargs):
    kwargs.setdefault("hours", 24)
    kwargs.setdefault("max_items", 50)
    return news.collect(feeds, now=NOW, fetcher=make_fetcher(mapping), **kwargs)


def by_title(items, fragment):
    return next(i for i in items if fragment in i["title"])


# --- parsing -----------------------------------------------------------------------

def test_prnewswire_quirks_are_cleaned_up():
    items = news.parse_feed(MAPPING[PRN["url"]], PRN)
    acme = by_title(items, "Acme")
    assert acme["title"] == "Acme Robotics & Partners Win $50 Million Defense Contract - ACMR"  # &amp;amp; fixed
    assert acme["summary"] == "Acme Robotics & Partners have been awarded a $50 million contract."  # tags stripped
    assert acme["published"] == "2026-10-17T02:08:00Z"
    assert acme["source"] == "PR Newswire" and acme["market"] == "US"
    assert by_title(items, "Ballard")["summary"].endswith("third quarter results…")  # &hellip; unescaped


def test_a_title_ending_in_a_ticker_is_not_split_like_a_publisher():
    acme = by_title(news.parse_feed(MAPPING[PRN["url"]], PRN), "Acme")
    assert acme["title"].endswith("- ACMR") and acme["source"] == "PR Newswire"


def test_angle_brackets_and_line_breaks_in_titles_survive():
    beta = by_title(news.parse_feed(MAPPING[PRN["url"]], PRN), "Beta")
    assert beta["title"] == 'Beta "Corp" says <NYSE: BETA> order book is "full" right now'


def test_yahoo_iso_dates_source_element_and_missing_date():
    items = news.parse_feed(MAPPING[YAHOO["url"]], YAHOO)
    wab = by_title(items, "Westinghouse")
    assert wab["published"] == "2026-10-17T01:43:50Z" and wab["source"] == "Yahoo Finance"
    assert wab["summary"] == ""
    assert by_title(items, "Gamma")["published"] is None       # never invent a date


def test_google_publisher_suffix_is_split_off_and_repeated_summary_dropped():
    items = news.parse_feed(MAPPING[GOOGLE["url"]], GOOGLE)
    zenith = by_title(items, "Zenith")
    assert (zenith["title"], zenith["source"], zenith["summary"]) == ("Zenith Defence bags Rs 800 crore order", "Economic Times", "")
    foo = by_title(items, "Foo")
    assert (foo["title"], foo["source"]) == ("Foo - Bar Corp wins offshore wind deal", "Reuters")  # splits on the LAST " - "


def test_google_titles_are_left_alone_when_the_flag_is_off():
    plain = dict(GOOGLE, source_in_title=False)
    assert by_title(news.parse_feed(MAPPING[GOOGLE["url"]], plain), "Zenith")["title"].endswith("- Economic Times")


def test_indian_time_zone_is_converted_to_utc():
    hal = by_title(news.parse_feed(MAPPING[ET["url"]], ET), "Hindustan")
    assert hal["published"] == "2026-10-17T02:30:00Z"


def test_ids_are_stable_and_unique():
    first = news.parse_feed(MAPPING[PRN["url"]], PRN)
    again = news.parse_feed(MAPPING[PRN["url"]], PRN)
    assert [i["id"] for i in first] == [i["id"] for i in again]
    assert len({i["id"] for i in first}) == len(first)
    assert all(len(i["id"]) == 12 for i in first)


@pytest.mark.parametrize("body", [
    fx("html_block_page.html"),
    b'{"error": "blocked"}',
    b"Access denied",
    b"\x00\x01 not xml at all",
    b"",
])
def test_non_feeds_that_return_200_are_reported_not_silently_empty(body):
    with pytest.raises(ValueError, match="couldn't read"):
        news.parse_feed(body, PRN)


def test_a_genuinely_empty_feed_is_fine():
    assert news.parse_feed(fx("empty_feed.xml"), PRN) == []


# --- collecting ----------------------------------------------------------------------

def test_collect_mixes_sources_dedupes_and_puts_undated_items_last():
    result = run([PRN, YAHOO, ET])
    titles = [i["title"] for i in result["items"]]
    assert result["count"] == 8 and len(titles) == 8
    assert sum("Westinghouse" in t for t in titles) == 1        # same story in two feeds, kept once
    assert not any("Old announcement" in t or "stale" in t for t in titles)   # outside 24h
    assert "Gamma" in titles[-1]                                 # no date -> last
    dated = [i["published"] for i in result["items"] if "published" in i]
    assert dated == sorted(dated, reverse=True)                  # newest first
    assert result["feed_errors"] == [] and result["feeds_ok"] == ["PR Newswire", "Yahoo Finance", "ET Stocks"]


def test_hours_window_is_respected():
    result = run([PRN, YAHOO, ET], hours=1)                      # cutoff 02:00Z
    titles = " | ".join(i["title"] for i in result["items"])
    assert result["count"] == 5
    for expected in ("Beta", "Ballard", "Acme", "Hindustan", "Gamma"):   # Gamma has no date, so it stays
        assert expected in titles
    assert "Strive" not in titles and "Westinghouse" not in titles


def test_keyword_filter_is_case_insensitive_and_checks_summaries_too():
    assert [i["title"][:4] for i in run([PRN, YAHOO, ET], keyword="CONTRACT")["items"]] == ["Acme"]
    assert [i["title"][:4] for i in run([PRN, YAHOO, ET], keyword="defence ministry")["items"]] == ["Hind"]   # summary-only match


def test_max_items_takes_turns_between_feeds():
    result = run([PRN, YAHOO, ET], max_items=3)
    assert result["count"] == 3
    assert {i["source"] for i in result["items"]} == {"PR Newswire", "Yahoo Finance", "ET Stocks"}
    assert {i["market"] for i in result["items"]} == {"US", "IN"}


def test_one_broken_feed_does_not_sink_the_rest():
    mapping = dict(MAPPING, **{YAHOO["url"]: urllib.error.URLError("boom"), ET["url"]: fx("html_block_page.html")})
    result = run([PRN, YAHOO, ET], mapping)
    assert result["feeds_ok"] == ["PR Newswire"] and result["count"] == 4
    errors = {e["feed"]: e["error"] for e in result["feed_errors"]}
    assert "URLError" in errors["Yahoo Finance"] and "couldn't read" in errors["ET Stocks"]


def test_all_feeds_failing_still_returns_a_clean_result():
    boom = {f["url"]: TimeoutError("slow") for f in (PRN, YAHOO)}
    result = run([PRN, YAHOO], boom)
    assert result["count"] == 0 and len(result["feed_errors"]) == 2


def test_no_feeds_at_all():
    assert run([])["count"] == 0


def test_empty_values_are_left_out_of_items_to_save_tokens():
    items = run([YAHOO])["items"]
    gamma = by_title(items, "Gamma")
    assert "published" not in gamma and "summary" not in gamma
    assert "id" in gamma and "link" in gamma


def test_untrusted_content_warning_travels_with_the_result():
    assert "untrusted" in run([PRN])["note"]


# --- feeds.json ------------------------------------------------------------------------

def test_the_shipped_feed_list_is_valid():
    everything = json.loads(news.FEEDS_FILE.read_text())["feeds"]
    assert len(everything) == len({f["name"] for f in everything})
    enabled = news.load_feeds()
    assert enabled and all(f["url"].startswith("https://") for f in enabled)
    assert {f["market"] for f in enabled} == {"US", "IN"}
    assert all(not f["enabled"] for f in everything if "news.google.com" in f["url"])   # Google stays opt-in
    assert {f["market"] for f in news.load_feeds(market="IN")} == {"IN"}


@pytest.mark.parametrize("feed, message", [
    ({"name": "x", "market": "US"}, "missing 'url'"),
    ({"name": "x", "market": "MARS", "url": "https://a"}, "market"),
    ({"name": "x", "market": "US", "url": "ftp://a"}, "http"),
])
def test_bad_feed_entries_fail_loudly(tmp_path, feed, message):
    path = tmp_path / "feeds.json"
    path.write_text(json.dumps({"feeds": [feed]}))
    with pytest.raises(ValueError, match=message):
        news.load_feeds(path)


def test_duplicate_feed_names_are_rejected(tmp_path):
    path = tmp_path / "feeds.json"
    feed = {"name": "x", "market": "US", "url": "https://a"}
    path.write_text(json.dumps({"feeds": [feed, feed]}))
    with pytest.raises(ValueError, match="duplicate"):
        news.load_feeds(path)


# --- the real download path, against a local server -----------------------------------------

class FeedHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.headers_seen.append(dict(self.headers))
        if self.path == "/feed.xml":
            body = fx("et_sample.xml")
            self.send_response(200)
            self.send_header("Content-Type", "application/rss+xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/moved":
            self.send_response(302)
            self.send_header("Location", "/feed.xml")
            self.end_headers()
        else:
            self.send_error(404)


@pytest.fixture
def feed_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FeedHandler)
    srv.headers_seen = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_fetch_bytes_downloads_and_identifies_itself(feed_server):
    raw = news.fetch_bytes(f"http://127.0.0.1:{feed_server.server_port}/feed.xml")
    assert raw == fx("et_sample.xml")
    assert "stock-news-agent" in feed_server.headers_seen[0]["User-Agent"]


def test_fetch_bytes_follows_redirects(feed_server):
    assert news.fetch_bytes(f"http://127.0.0.1:{feed_server.server_port}/moved") == fx("et_sample.xml")


def test_fetch_bytes_raises_on_404(feed_server):
    with pytest.raises(urllib.error.HTTPError):
        news.fetch_bytes(f"http://127.0.0.1:{feed_server.server_port}/nope")


def test_fetch_bytes_refuses_oversized_responses(feed_server, monkeypatch):
    monkeypatch.setattr(news, "MAX_BYTES", 100)
    with pytest.raises(ValueError, match="larger than"):
        news.fetch_bytes(f"http://127.0.0.1:{feed_server.server_port}/feed.xml")


def test_collect_end_to_end_over_real_http(feed_server):
    feed = {"name": "Local ET", "market": "IN", "url": f"http://127.0.0.1:{feed_server.server_port}/feed.xml"}
    result = news.collect([feed], now=NOW)
    assert result["count"] == 2 and result["feed_errors"] == []


# --- the tool ----------------------------------------------------------------------------------

@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Point the tool at fixture data and freeze time."""
    feeds = [dict(f, enabled=True) for f in (PRN, YAHOO, ET)] + [dict(GOOGLE, enabled=False)]
    path = tmp_path / "feeds.json"
    path.write_text(json.dumps({"feeds": feeds}))
    monkeypatch.setattr(news, "FEEDS_FILE", path)
    monkeypatch.setattr(news, "fetch_bytes", make_fetcher(MAPPING))
    monkeypatch.setattr(news, "collect", functools.partial(news.collect, now=NOW))
    return path


def tool(args):
    return json.loads(tools.run_tool("fetch_news", json.dumps(args)))


def test_tool_forgives_sloppy_argument_formats(wired):
    out = tool({"market": "us", "max_items": "3"})      # lowercase market, number as text
    assert out["count"] == 3 and {i["market"] for i in out["items"]} == {"US"}


def test_tool_with_no_arguments_uses_sensible_defaults(wired):
    out = json.loads(tools.run_tool("fetch_news", ""))
    assert out["count"] == 8 and {i["market"] for i in out["items"]} == {"US", "IN"}


def test_tool_skips_disabled_feeds(wired):
    assert not any("Zenith" in i["title"] for i in tool({})["items"])


@pytest.mark.parametrize("bad", [
    {"max_items": 500}, {"max_items": 0}, {"hours": 0}, {"hours": 9999},
    {"market": "MARS"}, {"keyword": "x" * 200},
])
def test_tool_rejects_out_of_range_arguments_with_an_error_message(wired, bad):
    assert "error" in tool(bad)


def test_tool_explains_when_a_market_has_no_enabled_feeds(wired, tmp_path, monkeypatch):
    only_us = tmp_path / "us_only.json"
    only_us.write_text(json.dumps({"feeds": [dict(PRN, enabled=True)]}))
    monkeypatch.setattr(news, "FEEDS_FILE", only_us)
    assert "No enabled feeds" in tool({"market": "IN"})["error"]


def test_tool_output_is_valid_json_even_with_quotes_and_non_ascii(wired):
    raw = tools.run_tool("fetch_news", json.dumps({"keyword": "order book"}))
    assert json.loads(raw)["items"][0]["title"] == 'Beta "Corp" says <NYSE: BETA> order book is "full" right now'


def test_the_model_can_use_the_tool_inside_the_loop(wired, tmp_path):
    client = FakeClient([
        spec(calls=[call("fetch_news", '{"market": "IN", "max_items": 5}', id="n1")]),
        spec(content="Hindustan Aeronautics got a Rs 2,000 crore order (Economic Times)."),
    ])
    result = run_agent(client, "m", "sys", "scan the news", 4, Tracer(tmp_path))
    assert (result.stopped, result.steps) == ("done", 2)
    tool_msg = client.sent[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "n1"
    payload = json.loads(tool_msg["content"])
    assert any("Hindustan" in i["title"] for i in payload["items"]) and "untrusted" in payload["note"]
    assert {t["function"]["name"] for t in client.requests[0]["tools"]} == {"get_time", "fetch_news"}


# --- try_tool.py ---------------------------------------------------------------------------------------

def test_try_tool_runs_a_tool_from_key_value_pairs(capsys):
    assert try_tool.main(["try_tool.py", "get_time", "timezone=Asia/Kolkata"]) == 0
    assert json.loads(capsys.readouterr().out)["timezone"] == "Asia/Kolkata"


def test_try_tool_reads_numbers_and_rejects_malformed_pairs(capsys):
    assert try_tool.parse_value("5") == 5 and try_tool.parse_value("IN") == "IN"
    assert try_tool.main(["try_tool.py", "fetch_news", "oops"]) == 2


def test_try_tool_shows_error_messages_for_bad_calls(capsys):
    try_tool.main(["try_tool.py", "get_time", "timezone=Mars/Base"])
    assert "error" in json.loads(capsys.readouterr().out)
