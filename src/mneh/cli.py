"""Command-line interface.

Commands:
    mneh capture <source>           Ingest a URL or arXiv reference
    mneh search <query>             Search the corpus
    mneh show <hash>                Inspect a capture
    mneh list                       List all captures
    mneh delete <hash>              Remove a capture

Flags:
    -v          verbose (more detail)
    -vv         very verbose (even more detail)
    -u          show URLs (for list)
    -q QUERY    resolve query to top result (for show)
    --json      machine-readable output
    --db PATH   use alternate database (default: ~/.mneh/mneh.db)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .capture import capture_url
from .embed import embed_single
from .storage import (
    connect,
    search_hybrid,
    list_captures,
    get_capture,
    get_last_capture,
    delete_capture,
    HitKey,
)
from .display import (
    format_list,
    format_show,
    format_show_json,
    format_search,
    format_search_json,
    aggregate_search_results,
    filter_noise,
)

DEFAULT_DB = Path.home() / ".mneh" / "mneh.db"
DEFAULT_STORAGE = Path.home() / ".mneh" / "storage"


def main():
    parser = argparse.ArgumentParser(
        prog="mneh",
        description="Local-first capture + retrieval for your exocortex.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    mneh capture "https://example.com/article"
    mneh search "coordination failure game theory"
    mneh search "react hooks" -v
    mneh list
    mneh list -uv
    mneh show a1b2c3d4
    mneh show --last
    mneh show -q "moloch"
    mneh delete a1b2c3d4

RRF Score Guide (with k=60, 4 channels):
    >0.05   strong hit (top ranks in multiple channels)
    0.02-0.05   moderate (mid-ranks or single-channel)
    <0.02   weak (probably noise, filtered by default)
        """,
    )

    # Global options
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="database path (default: ~/.mneh/mneh.db)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # capture
    cap = sub.add_parser("capture", help="Capture a URL or arXiv reference")
    cap.add_argument("url", help="URL or arXiv reference (e.g. arXiv:2002.12327v3)")
    cap.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_STORAGE,
        help="raw file storage directory",
    )

    # search
    s = sub.add_parser("search", help="Search captures")
    s.add_argument("query", help="search query")
    s.add_argument("-v", "--verbose", action="count", default=0, help="verbose output (-v for handles/chunks)")
    s.add_argument("-k", type=int, default=20, help="number of results (default: 20)")
    s.add_argument("--json", action="store_true", help="JSON output")
    s.add_argument("--no-vector", action="store_true", help="skip embedding search (FTS only)")

    # show
    sh = sub.add_parser("show", help="Show a capture")
    sh.add_argument("hash", nargs="?", help="hash or hash prefix")
    sh.add_argument("--last", action="store_true", help="show most recent capture")
    sh.add_argument("-q", "--query", help="show top result for query")
    sh.add_argument("-v", "--verbose", action="count", default=0, help="verbose (-v handles, -vv chunks)")
    sh.add_argument("--json", action="store_true", help="JSON output")

    # list
    ls = sub.add_parser("list", help="List captures")
    ls.add_argument("-v", "--verbose", action="store_true", help="show chunk/handle counts")
    ls.add_argument("-u", "--urls", action="store_true", help="show URLs")
    ls.add_argument("--json", action="store_true", help="JSON output")

    # delete
    d = sub.add_parser("delete", help="Delete a capture")
    d.add_argument("hash", help="hash or hash prefix")
    d.add_argument(
        "--storage",
        type=Path,
        default=DEFAULT_STORAGE,
        help="raw file storage directory",
    )

    args = parser.parse_args()

    # Ensure dirs exist
    args.db.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "capture":
        cmd_capture(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "delete":
        cmd_delete(args)


def cmd_capture(args):
    """Capture a URL or arXiv reference."""
    args.storage.mkdir(parents=True, exist_ok=True)
    h = capture_url(args.db, args.url, args.storage)
    print(f"Captured: {h[:8]}")


def cmd_search(args):
    """Search the corpus."""
    conn = connect(args.db)

    qvec = None
    if not args.no_vector:
        qvec = embed_single(args.query)

    results, debug_lists = search_hybrid(conn, args.query, qvec)

    if not results:
        print("No results.")
        return

    # Build score map from debug_lists for aggregation
    hit_scores: dict[HitKey, float] = {}
    for r in results:
        hk = HitKey(r["hash"], r["hit_kind"], r["hit_index"])
        hit_scores[hk] = r["rrf_score"]

    docs = aggregate_search_results(results, hit_scores)
    docs = filter_noise(docs)

    if not docs:
        print("No results above noise threshold.")
        return

    if args.json:
        print(format_search_json(docs[:args.k]))
    else:
        print(format_search(docs, args.query, verbose=args.verbose > 0, limit=args.k))


def cmd_show(args):
    """Show a capture."""
    conn = connect(args.db)

    # Resolve what to show
    if args.last:
        detail = get_last_capture(conn)
        if not detail:
            print("No captures.")
            sys.exit(1)
    elif args.query:
        # Search and get top result
        qvec = embed_single(args.query)
        results, _ = search_hybrid(conn, args.query, qvec)
        if not results:
            print(f"No results for query: {args.query}")
            sys.exit(1)
        # Get the top-scoring document hash
        top_hash = results[0]["hash"]
        detail = get_capture(conn, top_hash)
        if not detail:
            print(f"Could not load capture: {top_hash}")
            sys.exit(1)
    elif args.hash:
        detail = get_capture(conn, args.hash)
        if not detail:
            print(f"Not found or ambiguous: {args.hash}")
            sys.exit(1)
    else:
        print("Specify a hash, --last, or -q QUERY")
        sys.exit(1)

    if args.json:
        print(format_show_json(detail))
    else:
        print(format_show(detail, verbose=args.verbose))


def cmd_list(args):
    """List all captures."""
    conn = connect(args.db)
    captures = list_captures(conn)

    if args.json:
        import json
        out = [
            {
                "hash": c.hash,
                "title": c.title,
                "source_uri": c.source_uri,
                "captured_at": c.captured_at,
                "date": c.date,
                "chunk_count": c.chunk_count,
                "handle_count": c.handle_count,
            }
            for c in captures
        ]
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(format_list(captures, show_urls=args.urls, verbose=args.verbose))


def cmd_delete(args):
    """Delete a capture."""
    conn = connect(args.db)

    # Show what we're about to delete
    detail = get_capture(conn, args.hash)
    if not detail:
        print(f"Not found or ambiguous: {args.hash}")
        sys.exit(1)

    deleted = delete_capture(conn, args.hash, storage_dir=args.storage)
    if deleted:
        print(f"Deleted: {deleted[:8]} ({detail.title or 'Untitled'})")
    else:
        print(f"Failed to delete: {args.hash}")
        sys.exit(1)


if __name__ == "__main__":
    main()
