"""Entry point.   python run.py "What time is it in Tokyo?" """
import argparse
import dataclasses
import sys

import openai
from openai import OpenAI

import config
import llm
from agent import run_agent
from llm import THINKING_MODES
from prompts import SYSTEM_PROMPT
from tracing import Tracer

DEFAULT_PROMPT = "Scan the latest headlines from the last 24 hours and tell me which companies are in the news and why."


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent once.")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--thinking", choices=list(THINKING_MODES), help="override NVIDIA_THINKING")
    parser.add_argument("--show-thinking", action="store_true", help="print the model's reasoning as it streams")
    args = parser.parse_args()

    settings = config.model_settings()
    if args.thinking:
        settings = dataclasses.replace(settings, thinking=args.thinking)

    # The SDK's own retries are quick and short, so we turn them off and use our own
    # patient ones (llm.py), which wait longer and tell you what's happening.
    # timeout applies between chunks of a stream, so 120s is generous.
    client = OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=config.get_api_key(),
                    max_retries=0, timeout=120)
    tracer = Tracer(config.TRACE_DIR)

    def on_retry(attempt: int, delay: float, exc: Exception) -> None:
        print(f"NVIDIA says it's busy ({exc}). Waiting {delay:.0f}s, then retry {attempt} of {settings.retries}...",
              file=sys.stderr, flush=True)

    on_reasoning = None
    if args.show_thinking:
        def on_reasoning(piece: str) -> None:
            print(piece, end="", file=sys.stderr, flush=True)

    try:
        result = run_agent(client, args.model, SYSTEM_PROMPT, args.prompt, args.max_steps,
                           tracer, settings=settings, on_reasoning=on_reasoning, on_retry=on_retry)
    except openai.AuthenticationError:
        print("NVIDIA rejected the API key. Check NVIDIA_API_KEY in your .env.")
        return 2
    except openai.NotFoundError:
        print(f"NVIDIA doesn't recognise the model '{args.model}'. Check the name on build.nvidia.com.")
        return 2
    except openai.APIError as e:
        print(f"API error: {e}")
        if llm.is_retryable(e):
            print(f"Still failing after {settings.retries} retries. The free endpoint is probably swamped. "
                  "Try again in a few minutes, or switch to a smaller model, e.g. "
                  "--model nvidia/nemotron-3-super-120b-a12b")
        return 2

    if args.show_thinking:
        print("\n---", file=sys.stderr)
    if result.final:
        print(result.final)
    print(f"\n[{result.stopped} after {result.steps} step(s)] trace: {tracer.path}")
    return 0 if result.stopped == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
