"""Run one tool by hand, with no model involved. This is how we check a tool works
before letting the agent near it.

    python try_tool.py fetch_news
    python try_tool.py fetch_news market=IN max_items=5
    python try_tool.py fetch_news keyword=contract hours=48
    python try_tool.py get_time timezone=Asia/Tokyo

Arguments are key=value pairs (values like 5 or true are read as JSON, anything
else as text). Plain key=value avoids PowerShell's habit of mangling quotes.
"""
import json
import sys

import tools


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main(argv: list) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Tools:", ", ".join(tools.TOOLS))
        return 0

    name, pairs = argv[1], argv[2:]
    args = {}
    for pair in pairs:
        if "=" not in pair:
            print(f"Expected key=value, got {pair!r}")
            return 2
        key, value = pair.split("=", 1)
        args[key] = parse_value(value)

    result = tools.run_tool(name, json.dumps(args))
    try:
        print(json.dumps(json.loads(result), indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
