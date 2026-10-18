# Handoff: fix-3a-review

## What this branch does

Addresses the review of step-3a-storage (company mention storage + CSV export):

1. **Ticker matching bug** (`tools.py`): the old check did a lowercased
   substring search, so a ticker like "AI" matched inside the word "s**ai**d".
   Now matches the ticker as a whole token against the original-case article
   text: `(?<![A-Za-z0-9])TICKER(?![A-Za-z0-9])`.

2. **Missing metadata** (`tools.py`): `save_mention` used to hardcode
   `source=""`, `published_utc=None`, `link=None`. `fetch_news` now caches
   `source`/`published`/`link` per article id, and `save_mention` fills them
   in from that cache.

3. **A crash path in `run_tool`** (`tools.py`): the ticker validator called
   `re.match` directly on whatever the model sent. A non-string value (e.g.
   `ticker: 123`) raised `TypeError`, which pydantic does **not** convert to a
   `ValidationError`, so it escaped `run_tool`'s `except ValidationError`
   clause uncaught — the one function documented to "never raise." Now
   non-strings raise `ValueError` instead (which pydantic does catch), and
   `""`/`"none"`/`"n/a"`/`"null"` (any case) are treated as "no ticker."

4. **CSV export** (`store.py`): rewritten on the `csv` module
   (`newline=""`, `utf-8-sig`) instead of hand-rolled comma/quote escaping.
   Added a formula-injection guard — any cell starting with `= + - @` or a
   tab/CR gets a leading apostrophe, so a hostile event string like
   `=HYPERLINK(...)` can't execute when opened in Excel/Sheets. The header
   row is now written even when the database file doesn't exist yet.

5. **Run-state leakage** (`tools.py`, `agent.py`): added
   `tools.reset_run_state()`, called at the top of `run_agent`, so fetched-article
   tracking doesn't leak between runs in a long-lived process (matters for
   tests and any future multi-run use). Tests now call it instead of
   hand-clearing the module-level dicts.

6. **Validation gaps** (`tools.py`): `company` and `event` are stripped and
   must be non-blank (`min_length=1` after stripping).

7. **Housekeeping**:
   - `.env.example`'s `AGENT_MAX_STEPS` updated from 8 to 30 to match `config.py`'s default.
   - `.claude/settings.local.json` untracked (`git rm --cached`) and added to `.gitignore` — it's a local permission file, not something to share via the repo.

## Test status

`python -m pytest -q` → **134 passed**, no skips, no xfails.

New/changed tests live in `tests/test_store.py`:
- whole-token ticker matching (accept/reject cases, including the "AI" inside "said" regression)
- metadata (source/published/link) flowing end-to-end from `fetch_news` through `save_mention` into the DB
- `run_tool` never raising on a non-string or placeholder ticker
- CSV formula-injection escaping and header-always-written
- company/event stripping and blank rejection

## What's NOT in this branch

Ticker lookup against real SEC/NSE data (that's `step-3b-ticker-lookup`, built
on top of this). Right now `save_mention`'s ticker check still only verifies
the ticker text appears in the article — real exchange/symbol validation
comes next.

## Known trade-offs / things to watch

- The ticker regex lookaround (`(?<![A-Za-z0-9])...(?![A-Za-z0-9])`) treats
  any letter/digit as a word character but **not** `.` or `-`, so a ticker
  like `BRK.B` is still found correctly inside `(BRK.B)`, but a ticker that is
  itself a prefix/suffix of another alnum run (rare in real tickers) would
  still be rejected correctly — this is intentional, not a gap.
- `reset_run_state()` clears **module-level global state** in `tools.py`. This
  is fine for the current single-threaded, one-run-at-a-time CLI, but would
  need per-run isolation (e.g. a context object instead of module globals) if
  the agent is ever run concurrently in the same process.
- CSV export now always touches the filesystem with `open(..., "w")` rather
  than `Path.write_text`; functionally equivalent, but note for anyone diffing
  against the old implementation.
