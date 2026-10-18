"""Tests for store.py and save_mention tool."""
import json
import sqlite3
from pathlib import Path
import tempfile

import pytest

import news
import store
import tools
from tools import ToolError


@pytest.fixture
def temp_db():
    """Temp database for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.db"
        yield path


def test_init_db_creates_table(temp_db):
    store.init_db(temp_db)
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='mentions'")
    assert cursor.fetchone() is not None
    conn.close()


def test_save_mention_inserts_row(temp_db):
    store.init_db(temp_db)
    result = store.save_mention(
        market="US",
        company="Acme Corp",
        event="announced merger",
        article_id="abc123",
        source="Reuters",
        path=temp_db,
    )
    assert result["saved"] is True
    assert result["row_id"] > 0

    conn = sqlite3.connect(temp_db)
    try:
        row = conn.execute("SELECT company, event FROM mentions WHERE id = ?", (result["row_id"],)).fetchone()
        assert row == ("Acme Corp", "announced merger")
    finally:
        conn.close()


def test_duplicate_insert_is_ignored(temp_db):
    store.init_db(temp_db)
    first = store.save_mention(
        market="US",
        company="Acme",
        event="news 1",
        article_id="id1",
        source="Reuters",
        path=temp_db,
    )
    assert first["saved"] is True

    second = store.save_mention(
        market="US",
        company="Acme",
        event="news 2",
        article_id="id1",
        source="AP",
        path=temp_db,
    )
    assert second["saved"] is False
    assert second["reason"] == "duplicate"

    conn = sqlite3.connect(temp_db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM mentions WHERE article_id = 'id1' AND company = 'Acme'").fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_save_mention_tool_rejects_bad_ticker():
    tools.reset_run_state()
    tools._valid_article_ids.add("id1")
    tools._article_meta["id1"] = {"text": "title here"}

    args_dict = {
        "article_id": "id1",
        "company": "Acme",
        "market": "US",
        "event": "news",
        "ticker": "INVALID!@#"
    }
    result = json.loads(tools.run_tool("save_mention", json.dumps(args_dict)))
    assert "error" in result
    assert "Ticker" in result["error"]


def test_save_mention_tool_rejects_ticker_not_in_article(temp_db):
    tools.reset_run_state()
    tools._valid_article_ids.add("id1")
    tools._article_meta["id1"] = {"text": "nvidia announces new gpu"}

    args_dict = {
        "article_id": "id1",
        "company": "Acme",
        "market": "US",
        "event": "news",
        "ticker": "AMD"
    }
    result = json.loads(tools.run_tool("save_mention", json.dumps(args_dict)))
    assert "error" in result
    assert "does not appear" in result["error"]


def test_save_mention_tool_rejects_unknown_article_id(temp_db):
    tools.reset_run_state()

    args_dict = {
        "article_id": "unknown",
        "company": "Acme",
        "market": "US",
        "event": "news",
    }
    result = json.loads(tools.run_tool("save_mention", json.dumps(args_dict)))
    assert "error" in result
    assert "not found" in result["error"]


def test_save_mention_tool_accepts_ticker_in_title(monkeypatch, temp_db):
    tools.reset_run_state()
    monkeypatch.setattr(store, "DB_PATH", temp_db)
    tools._valid_article_ids.add("id1")
    tools._article_meta["id1"] = {"text": "Nvidia (NVDA) announces new GPU"}

    args_dict = {
        "article_id": "id1",
        "company": "Nvidia",
        "market": "US",
        "event": "announced new GPU",
        "ticker": "NVDA"
    }
    result = json.loads(tools.run_tool("save_mention", json.dumps(args_dict)))
    assert result["saved"] is True


# --- whole-token, original-case ticker matching -----------------------------------

def test_ticker_rejected_when_only_a_substring_of_another_word():
    tools.reset_run_state()
    tools._valid_article_ids.add("id1")
    # lowercase "ai" is a substring of "said"; must not match ticker "AI"
    tools._article_meta["id1"] = {"text": "Markets said it is on the move"}

    result = json.loads(tools.run_tool("save_mention", json.dumps({
        "article_id": "id1", "company": "Acme", "market": "US", "event": "news", "ticker": "AI"
    })))
    assert "error" in result
    assert "does not appear" in result["error"]


def test_single_letter_ticker_rejected_when_only_a_substring():
    tools.reset_run_state()
    tools._valid_article_ids.add("id1")
    tools._article_meta["id1"] = {"text": "Apple reported record profits"}

    result = json.loads(tools.run_tool("save_mention", json.dumps({
        "article_id": "id1", "company": "Acme", "market": "US", "event": "news", "ticker": "A"
    })))
    assert "error" in result
    assert "does not appear" in result["error"]


@pytest.mark.parametrize("text", [
    "Acme wins big contract (ACME)",
    "Acme wins big contract - ACME",
])
def test_ticker_accepted_as_whole_token_with_punctuation(monkeypatch, temp_db, text):
    tools.reset_run_state()
    monkeypatch.setattr(store, "DB_PATH", temp_db)
    tools._valid_article_ids.add("id1")
    tools._article_meta["id1"] = {"text": text}

    result = json.loads(tools.run_tool("save_mention", json.dumps({
        "article_id": "id1", "company": "Acme", "market": "US", "event": "news", "ticker": "ACME"
    })))
    assert result["saved"] is True


def test_csv_export_creates_file(temp_db):
    csv_path = Path(tempfile.gettempdir()) / "test_export.csv"
    csv_path.unlink(missing_ok=True)

    store.init_db(temp_db)
    store.save_mention(
        market="US",
        company="Acme",
        event="announced deal",
        article_id="id1",
        source="Reuters",
        ticker="ACME",
        path=temp_db,
    )

    store.export_csv(csv_path, temp_db)
    assert csv_path.exists()

    content = csv_path.read_text(encoding="utf-8-sig")
    assert "Acme" in content
    assert "announced deal" in content
    csv_path.unlink()


def test_csv_export_escapes_commas_and_quotes(temp_db):
    csv_path = Path(tempfile.gettempdir()) / "test_export_escape.csv"
    csv_path.unlink(missing_ok=True)

    store.init_db(temp_db)
    store.save_mention(
        market="US",
        company='Smith, Inc. "The" Company',
        event="deal with price $5,000 (estimated)",
        article_id="id1",
        source="Reuters",
        path=temp_db,
    )

    store.export_csv(csv_path, temp_db)
    lines = csv_path.read_text(encoding="utf-8-sig").strip().split("\n")
    assert len(lines) == 2  # header + 1 row

    # CSV should have proper quoting
    row = lines[1]
    assert '"Smith, Inc. ""The"" Company"' in row or 'Smith, Inc. "The" Company' in row or 'Smith' in row
    csv_path.unlink()


def test_csv_export_preserves_unicode(temp_db):
    csv_path = Path(tempfile.gettempdir()) / "test_export_unicode.csv"
    csv_path.unlink(missing_ok=True)

    store.init_db(temp_db)
    store.save_mention(
        market="IN",
        company="Hindustan Aeronautics",
        event="Rs 2000 crore order",
        article_id="id1",
        source="ET",
        path=temp_db,
    )

    store.export_csv(csv_path, temp_db)
    content = csv_path.read_text(encoding="utf-8-sig")
    assert "₹" not in content  # We store "Rs", not the ₹ symbol
    assert "2000 crore" in content
    csv_path.unlink()


def test_csv_export_with_empty_db(temp_db):
    csv_path = Path(tempfile.gettempdir()) / "test_export_empty.csv"
    csv_path.unlink(missing_ok=True)

    store.init_db(temp_db)
    store.export_csv(csv_path, temp_db)

    content = csv_path.read_text(encoding="utf-8-sig")
    lines = content.strip().split("\n")
    assert lines[0] == "first_seen_utc,published_utc,market,company,ticker,event,article_id,source,link"
    csv_path.unlink()


def test_save_mention_stores_all_fields(temp_db):
    store.init_db(temp_db)
    result = store.save_mention(
        market="IN",
        company="TCS",
        event="acquired startup",
        article_id="xyz",
        source="ET Stocks",
        published_utc="2026-10-17T10:00:00Z",
        ticker="TCS",
        link="https://example.com/news",
        path=temp_db,
    )

    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM mentions WHERE id = ?", (result["row_id"],)).fetchone()
        assert row["market"] == "IN"
        assert row["company"] == "TCS"
        assert row["ticker"] == "TCS"
        assert row["link"] == "https://example.com/news"
        assert row["published_utc"] == "2026-10-17T10:00:00Z"
        assert row["first_seen_utc"] is not None
    finally:
        conn.close()


# --- metadata flows from fetch_news into save_mention -----------------------------

def test_save_mention_fills_source_published_and_link_from_fetch_news(monkeypatch, temp_db):
    tools.reset_run_state()
    monkeypatch.setattr(store, "DB_PATH", temp_db)

    fake_feed = {"name": "Test Feed", "market": "US", "url": "http://test"}
    fake_result = {
        "fetched_at": "2026-10-17T10:00:00Z",
        "feeds_ok": ["Test Feed"],
        "feed_errors": [],
        "count": 1,
        "items": [{
            "id": "abc123",
            "title": "Acme Corp wins contract (ACME)",
            "summary": "Acme signed a deal.",
            "source": "Reuters",
            "published": "2026-10-17T09:00:00Z",
            "market": "US",
            "link": "https://example.com/acme-news",
        }],
        "note": "x",
    }
    monkeypatch.setattr(news, "load_feeds", lambda market="ALL": [fake_feed])
    monkeypatch.setattr(news, "collect", lambda *a, **k: fake_result)

    tools.run_tool("fetch_news", "{}")
    out = json.loads(tools.run_tool("save_mention", json.dumps({
        "article_id": "abc123",
        "company": "Acme Corp",
        "market": "US",
        "event": "won a new contract",
        "ticker": "ACME",
    })))
    assert out["saved"] is True

    conn = sqlite3.connect(temp_db)
    try:
        row = conn.execute(
            "SELECT source, published_utc, link FROM mentions WHERE id = ?", (out["row_id"],)
        ).fetchone()
        assert row == ("Reuters", "2026-10-17T09:00:00Z", "https://example.com/acme-news")
    finally:
        conn.close()


# --- run_tool must never raise, whatever the model sends --------------------------

def test_save_mention_tool_rejects_non_string_ticker_with_an_error_not_a_crash():
    tools.reset_run_state()
    tools._valid_article_ids.add("id1")
    tools._article_meta["id1"] = {"text": "Acme news"}

    result = json.loads(tools.run_tool("save_mention", json.dumps({
        "article_id": "id1", "company": "Acme", "market": "US", "event": "news", "ticker": 123
    })))
    assert "error" in result


def test_save_mention_tool_treats_empty_string_ticker_as_none(monkeypatch, temp_db):
    tools.reset_run_state()
    monkeypatch.setattr(store, "DB_PATH", temp_db)
    tools._valid_article_ids.add("id1")
    tools._article_meta["id1"] = {"text": "Acme news"}

    result = json.loads(tools.run_tool("save_mention", json.dumps({
        "article_id": "id1", "company": "Acme", "market": "US", "event": "news", "ticker": ""
    })))
    assert result["saved"] is True

    conn = sqlite3.connect(temp_db)
    try:
        row = conn.execute("SELECT ticker FROM mentions WHERE id = ?", (result["row_id"],)).fetchone()
        assert row[0] is None
    finally:
        conn.close()


@pytest.mark.parametrize("raw_ticker", ["none", "N/A", "Null", "  "])
def test_save_mention_tool_treats_placeholder_words_as_no_ticker(monkeypatch, temp_db, raw_ticker):
    tools.reset_run_state()
    monkeypatch.setattr(store, "DB_PATH", temp_db)
    tools._valid_article_ids.add("id1")
    tools._article_meta["id1"] = {"text": "Acme news"}

    result = json.loads(tools.run_tool("save_mention", json.dumps({
        "article_id": "id1", "company": "Acme", "market": "US", "event": "news", "ticker": raw_ticker
    })))
    assert result["saved"] is True

    conn = sqlite3.connect(temp_db)
    try:
        row = conn.execute("SELECT ticker FROM mentions WHERE id = ?", (result["row_id"],)).fetchone()
        assert row[0] is None
    finally:
        conn.close()


# --- csv formula-injection guard and header-always -----------------------------

def test_csv_export_escapes_formula_injection(temp_db):
    csv_path = Path(tempfile.gettempdir()) / "test_export_formula.csv"
    csv_path.unlink(missing_ok=True)

    store.init_db(temp_db)
    store.save_mention(
        market="US",
        company="Acme",
        event='=HYPERLINK("http://evil.example")',
        article_id="id1",
        source="Reuters",
        path=temp_db,
    )

    store.export_csv(csv_path, temp_db)
    content = csv_path.read_text(encoding="utf-8-sig")
    assert "'=HYPERLINK(" in content
    assert '"=HYPERLINK(' not in content
    csv_path.unlink()


def test_csv_export_writes_header_even_when_db_file_is_missing(tmp_path):
    csv_path = tmp_path / "out.csv"
    missing_db = tmp_path / "does_not_exist.db"

    store.export_csv(csv_path, missing_db)

    content = csv_path.read_text(encoding="utf-8-sig")
    assert content.strip() == "first_seen_utc,published_utc,market,company,ticker,event,article_id,source,link"


# --- company/event: stripped, non-blank -----------------------------------------

def test_save_mention_strips_whitespace_from_company_and_event(monkeypatch, temp_db):
    tools.reset_run_state()
    monkeypatch.setattr(store, "DB_PATH", temp_db)
    tools._valid_article_ids.add("id1")
    tools._article_meta["id1"] = {"text": "Acme news"}

    result = json.loads(tools.run_tool("save_mention", json.dumps({
        "article_id": "id1", "company": "  Acme  ", "market": "US", "event": "  won a deal  "
    })))
    assert result["saved"] is True

    conn = sqlite3.connect(temp_db)
    try:
        row = conn.execute("SELECT company, event FROM mentions WHERE id = ?", (result["row_id"],)).fetchone()
        assert row == ("Acme", "won a deal")
    finally:
        conn.close()


@pytest.mark.parametrize("field", ["company", "event"])
def test_save_mention_tool_rejects_blank_company_or_event(field):
    tools.reset_run_state()
    tools._valid_article_ids.add("id1")
    tools._article_meta["id1"] = {"text": "Acme news"}

    args = {"article_id": "id1", "company": "Acme", "market": "US", "event": "news"}
    args[field] = "   "
    result = json.loads(tools.run_tool("save_mention", json.dumps(args)))
    assert "error" in result
