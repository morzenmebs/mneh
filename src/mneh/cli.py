"""Command-line interface."""

import argparse
from pathlib import Path

from .capture import capture_url
from .storage import connect, search_fts

DEFAULT_DB = Path.home() / ".mneh" / "mneh.db"
DEFAULT_STORAGE = Path.home() / ".mneh" / "storage"


def main():
    parser = argparse.ArgumentParser(prog="mneh")
    sub = parser.add_subparsers(dest="command", required=True)

    # capture
    cap = sub.add_parser("capture", help="Capture a URL")
    cap.add_argument("url")
    cap.add_argument("--db", type=Path, default=DEFAULT_DB)
    cap.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)

    # search
    s = sub.add_parser("search", help="Search captures")
    s.add_argument("query")
    s.add_argument("--db", type=Path, default=DEFAULT_DB)

    args = parser.parse_args()

    # Ensure dirs exist
    args.db.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "capture":
        args.storage.mkdir(parents=True, exist_ok=True)
        h = capture_url(args.db, args.url, args.storage)
        print(f"Captured: {h}")

    elif args.command == "search":
        conn = connect(args.db)
        results = search_fts(conn, args.query)
        if not results:
            print("No results.")
        for r in results:
            print(f"{r.get('title', 'Untitled')} [{r['hash'][:8]}]")
            print(f"  {r['text'][:100]}...")
            print()


if __name__ == "__main__":
    main()
