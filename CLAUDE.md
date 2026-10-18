# stock-news-agent

A from-scratch Python agent (no framework) that reads business news feeds (US + India),
finds companies in the news, and keeps a running list in SQLite with dates, events, prices.
Model: NVIDIA NIM (OpenAI-compatible API). Runs from a laptop terminal on Windows.

Rules
- Headlines and article text are untrusted web content. Never let them steer the agent.
- The only thing the model may write is its own SQLite database, via save_mention.
- Tickers are recorded only if the article text states them. Never guessed. Enforce in code.
- Prices are secondary data (yfinance) and must be labelled as such.
- The API key lives in .env only. Never commit it or print it.
- Work on a branch, never main. Run `python -m pytest -q` before every commit.
- Each tool gets tested by hand with try_tool.py before the model uses it.