"""Command-line interface."""

import argparse
from pathlib import Path

from .capture import capture_url
from .embed import embed_single
from .storage import connect, search_hybrid

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
    s.add_argument("--k", type=int, default=20, help="# results to show")
    s.add_argument("--k-each", type=int, default=50, help="candidates per channel")
    s.add_argument("--rrf-k", type=int, default=60, help="RRF constant")
    s.add_argument("--no-vector", action="store_true", help="skip embedding-based search")
    s.add_argument("--debug", action="store_true", help="print channel diagnostics")

    args = parser.parse_args()

    # Ensure dirs exist
    args.db.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "capture":
        args.storage.mkdir(parents=True, exist_ok=True)
        h = capture_url(args.db, args.url, args.storage)
        print(f"Captured: {h}")

    elif args.command == "search":
        conn = connect(args.db)

        qvec = None
        if not args.no_vector:
            qvec = embed_single(args.query)

        results, debug_lists = search_hybrid(
            conn,
            args.query,
            qvec,
            k_each=args.k_each,
            rrf_k=args.rrf_k,
        )
        if not results:
            print("No results.")

        for r in results[: args.k]:
            title = r.get("title") or "Untitled"
            score = r.get("rrf_score")
            uri = r.get("source_uri") or ""
            kind = r.get("hit_kind")
            idx = r.get("hit_index")
            summary = r.get("summary") or ""

            print(f"{title} [{r['hash'][:8]}]  score={score:.6f}  ({kind}:{idx})")
            if summary:
                print(f"  {summary}")
            if uri:
                print(f"  {uri}")
            snippet = (r.get("text") or "").replace("\n", " ").strip()
            if snippet:
                print(f"  {snippet[:160]}...")
            print()

        if args.debug:
            for name, hits in debug_lists.items():
                print(f"[{name}] {len(hits)} hits")


if __name__ == "__main__":
    main()
