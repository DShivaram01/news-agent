# stock-news-agent

A small AI agent, written from scratch, that reads business news (India + US), spots which companies are in it, and will keep a running list. Built step by step.

- **Step 1** (done): the agent loop, streaming, retries, traces.
- **Step 2** (this version): a real `fetch_news` tool that reads RSS feeds.
- **Next**: `get_price`, then `save_mention` with a SQLite database and CSV export, then a daily schedule.

## Setup (Windows PowerShell)

```
cd stock-news-agent
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # new this step: feedparser
copy .env.example .env                 # first time only, then paste your nvapi- key
```

The key lives in `.env` only. Never paste it into code or chat. If it ever leaks, revoke it on build.nvidia.com and make a new one.

## Step 2: try the news tool by hand first (no model involved)

```
python try_tool.py fetch_news max_items=5
python try_tool.py fetch_news market=IN
python try_tool.py fetch_news keyword=contract hours=48
```

Look at the output before trusting the agent with it. Things to check:

- `feeds_ok` lists the feeds that worked. `feed_errors` says which failed and why.
- The Economic Times and Moneycontrol URLs in `feeds.json` are marked `"status": "unverified"`. I could only confirm the two US feeds from my side, so expect some India feeds to fail on the first try. A `404` means the URL changed. An `HTTP 403` means the site blocks scripts. In either case, find the working address on the site's own RSS page, paste it into `feeds.json`, or set `"enabled": false`.
- Google News feeds are in `feeds.json` but switched off. Google doesn't document that RSS search as a public API, so turn it on only if you're comfortable with that.

Then let the agent loose on it:

```
python run.py --model nvidia/nemotron-3-super-120b-a12b --show-thinking
```

## Run

```
python run.py                          # scans the news with the default prompt
python run.py "Anything from India about defence orders?"
python run.py --thinking medium        # shorter reasoning: faster, fewer tokens
```

Each run prints the answer and the path to a trace file in `traces/`. Open it to see every model reply, its reasoning, every tool call, and every retry.

If a run ends with something other than `done`:

| Status | Meaning |
|---|---|
| `malformed_tool_call` | The model wrote its tool call as plain text instead of a real tool call. Try `NVIDIA_TEMPERATURE=0.6` in `.env`. |
| `truncated` | It ran out of tokens, usually mid-reasoning. Raise `NVIDIA_MAX_TOKENS` or use `--thinking medium`. |
| `empty_response` | It thought but never answered. Check the trace. |
| `max_steps` | It kept calling tools until the step cap. Check the trace for what it was stuck on. |

### When NVIDIA says it's busy

The free endpoint gets swamped, especially the biggest models. "Service temporarily overloaded" (HTTP 503), rate limits (429) and dropped connections are retried automatically, waiting about 5s, 15s, 45s and then 120s, and printing what it's doing. Change how many tries with `NVIDIA_RETRIES` in `.env`. Bad keys and wrong model names are not retried, since waiting won't fix them.

If it's still failing after the retries, try again in a few minutes or switch to a smaller model: `python run.py --model nvidia/nemotron-3-super-120b-a12b`

## A note on safety

News text comes from the open web, so anyone can write a headline that says "ignore your instructions and...". The agent treats all of it as data: the prompt says so, every tool result carries a reminder, and for now the agent can't do anything with real consequences. It can read feeds and tell you what it found. When we add `save_mention`, it will only be able to write to its own database.

## Test (no key or network needed)

```
python -m pytest -q
```

## Files

| File | What it does |
|---|---|
| `agent.py` | The loop |
| `llm.py` | One streamed model call with retries, stitched into a plain reply |
| `tools.py` | Tools, their schemas, and safe execution |
| `news.py` | Downloads, parses, cleans, dedupes and filters the feeds |
| `feeds.json` | The list of news feeds. Edit this to add or remove sources |
| `try_tool.py` | Run any tool by hand without the model |
| `prompts.py` | System prompt |
| `tracing.py` | JSONL trace of each run |
| `config.py` | Settings, loaded from `.env` |
| `run.py` | Command-line entry point |
| `tests/` | Offline tests, with small saved copies of real feed formats in `tests/fixtures/` |
