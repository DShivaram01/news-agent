"""SQLite database for tracking company mentions in news."""
import csv
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

CSV_HEADER = ["first_seen_utc", "published_utc", "market", "company", "ticker", "event", "article_id", "source", "link"]
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

DB_PATH = Path(__file__).parent / "data" / "mentions.db"


def init_db(path: Optional[Path] = None) -> None:
    """Create tables if they don't exist."""
    path = path or DB_PATH
    path.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_seen_utc TEXT NOT NULL,
            published_utc TEXT,
            market TEXT NOT NULL,
            company TEXT NOT NULL,
            ticker TEXT,
            event TEXT NOT NULL,
            article_id TEXT NOT NULL,
            source TEXT NOT NULL,
            link TEXT,
            UNIQUE(article_id, company)
        )
    """)
    conn.commit()
    conn.close()


def save_mention(
    market: str,
    company: str,
    event: str,
    article_id: str,
    source: str,
    published_utc: Optional[str] = None,
    ticker: Optional[str] = None,
    link: Optional[str] = None,
    path: Optional[Path] = None,
) -> dict:
    """Save a mention to the database. Returns the inserted row id or None if duplicate."""
    path = path or DB_PATH
    path.parent.mkdir(exist_ok=True)
    init_db(path)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """INSERT INTO mentions
               (first_seen_utc, published_utc, market, company, ticker, event, article_id, source, link)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, published_utc, market, company, ticker, event, article_id, source, link)
        )
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return {"saved": True, "row_id": row_id}
    except sqlite3.IntegrityError:
        return {"saved": False, "reason": "duplicate"}
    finally:
        conn.close()


def _csv_safe(value: str) -> str:
    """Prefix a leading formula character so Excel/Sheets never execute cell content as a formula."""
    if value and value[0] in _FORMULA_PREFIXES:
        return "'" + value
    return value


def export_csv(csv_path: Optional[Path] = None, db_path: Optional[Path] = None) -> None:
    """Export all mentions to CSV with UTF-8 BOM. Always writes the header, even with no data."""
    csv_path = csv_path or Path(__file__).parent / "outputs" / "mentions.csv"
    db_path = db_path or DB_PATH

    csv_path.parent.mkdir(exist_ok=True)

    rows = []
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT first_seen_utc, published_utc, market, company, ticker, event, article_id, source, link "
                "FROM mentions ORDER BY first_seen_utc DESC"
            ).fetchall()
        finally:
            conn.close()

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([_csv_safe(row[col] or "") for col in CSV_HEADER])
