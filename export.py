"""Export mentions to CSV."""
import sys
import store


def main() -> int:
    try:
        store.export_csv()
        print("Exported to outputs/mentions.csv")
        return 0
    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
