"""SQLite database for tracking company mentions in news."""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

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


def export_csv(csv_path: Optional[Path] = None, db_path: Optional[Path] = None) -> None:
    """Export all mentions to CSV with UTF-8 BOM."""
    csv_path = csv_path or Path(__file__).parent / "outputs" / "mentions.csv"
    db_path = db_path or DB_PATH

    csv_path.parent.mkdir(exist_ok=True)
    if not db_path.exists():
        csv_path.write_text("", encoding="utf-8-sig")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT first_seen_utc, published_utc, market, company, ticker, event, article_id, source, link FROM mentions ORDER BY first_seen_utc DESC"
    ).fetchall()
    conn.close()

    lines = ["first_seen_utc,published_utc,market,company,ticker,event,article_id,source,link"]
    for row in rows:
        fields = [
            row["first_seen_utc"],
            row["published_utc"] or "",
            row["market"],
            row["company"],
            row["ticker"] or "",
            row["event"],
            row["article_id"],
            row["source"],
            row["link"] or "",
        ]
        escaped = [f'"{f.replace(chr(34), chr(34)+chr(34))}"' if any(c in f for c in (',', '"', '\n')) else f for f in fields]
        lines.append(",".join(escaped))

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
