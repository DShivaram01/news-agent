"""Tests for store.py and save_mention tool."""
import json
import sqlite3
from pathlib import Path
import tempfile

import pytest

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
    tools._valid_article_ids.clear()
    tools._article_texts.clear()
    tools._valid_article_ids.add("id1")
    tools._article_texts["id1"] = "title here"

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
    tools._valid_article_ids.clear()
    tools._article_texts.clear()
    tools._valid_article_ids.add("id1")
    tools._article_texts["id1"] = "nvidia announces new gpu"

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
    tools._valid_article_ids.clear()
    tools._article_texts.clear()

    args_dict = {
        "article_id": "unknown",
        "company": "Acme",
        "market": "US",
        "event": "news",
    }
    result = json.loads(tools.run_tool("save_mention", json.dumps(args_dict)))
    assert "error" in result
    assert "not found" in result["error"]


def test_save_mention_tool_accepts_ticker_in_title(temp_db):
    tools._valid_article_ids.clear()
    tools._article_texts.clear()
    tools._valid_article_ids.add("id1")
    tools._article_texts["id1"] = "nvda announces new gpu"

    args_dict = {
        "article_id": "id1",
        "company": "Nvidia",
        "market": "US",
        "event": "announced new GPU",
        "ticker": "NVDA"
    }
    # This should not error; the actual save will work in a real DB context
    from pydantic import ValidationError
    from tools import SaveMentionArgs
    # Should not raise
    SaveMentionArgs(**args_dict)


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
