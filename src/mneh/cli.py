"""Command-line interface."""

import argparse
import json
import hashlib
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .capture import capture_url, fetch_text
from .extract import extract_metadata
from .embed import embed_single
from .storage import connect, search_hybrid
from .display import collapse_hits_by_document, filter_by_score, format_result

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
    cap.add_argument("--prompt-file", type=Path, help="override SYSTEM_PROMPT from a file")

    # probe (prompt iteration, no DB writes)
    p = sub.add_parser("probe", help="Fetch + extract handles for manual inspection")
    p.add_argument("urls", nargs="+", help="3-5 varied URLs is a good start")
    p.add_argument("--prompt-file", type=Path, help="override SYSTEM_PROMPT from a file")
    p.add_argument("--out", type=Path, help="write results to a JSON file")

    pc = sub.add_parser("probe-compare", help="Compare two probe JSON outputs")
    pc.add_argument("a", type=Path, help="first probe JSON")
    pc.add_argument("b", type=Path, help="second probe JSON")

    # search
    s = sub.add_parser("search", help="Search captures")
    s.add_argument("query")
    s.add_argument("--db", type=Path, default=DEFAULT_DB)
    s.add_argument("--k", type=int, default=20, help="# results to show")
    s.add_argument("--k-each", type=int, default=50, help="candidates per channel")
    s.add_argument("--rrf-k", type=int, default=60, help="RRF constant")
    s.add_argument("--no-vector", action="store_true", help="skip embedding-based search")
    s.add_argument("--debug", action="store_true", help="print channel diagnostics")
    s.add_argument("--min-rrf", type=float, default=None, help="absolute score floor (after fusion)")
    s.add_argument(
        "--drop-bottom",
        type=float,
        default=0.10,
        help="drop bottom fraction of doc-level scores (default: 0.10)",
    )
    s.add_argument("--no-dedup", action="store_true", help="do not collapse multiple hits per doc")
    s.add_argument("--snippet-chars", type=int, default=360, help="snippet window size")
    s.add_argument("--width", type=int, default=92, help="wrap width")

    args = parser.parse_args()

    # Ensure dirs exist
    args.db.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "capture":
        args.storage.mkdir(parents=True, exist_ok=True)
        system_prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else None
        h = capture_url(args.db, args.url, args.storage, system_prompt=system_prompt)
        print(f"Captured: {h}")

    elif args.command == "probe":
        system_prompt = None
        if args.prompt_file:
            system_prompt = args.prompt_file.read_text(encoding="utf-8")

        run: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
            "documents": [],
        }

        for i, url in enumerate(args.urls, start=1):
            text, traf_meta = fetch_text(url)
            content_hash = hashlib.sha256(text.encode()).hexdigest()

            meta = extract_metadata(
                text,
                url=url,
                hints={
                    "title": traf_meta.get("title"),
                    "author": traf_meta.get("author"),
                    "date": traf_meta.get("date"),
                },
                system_prompt=system_prompt,
            )

            handles = meta.get("handles") or []
            # Basic duplicate check (exact, case-insensitive)
            norm = [h.strip().lower() for h in handles if h and h.strip()]
            dup_count = len(norm) - len(set(norm))

            title = meta.get("title") or "Untitled"
            summary = meta.get("summary") or ""

            print("=" * 92)
            print(f"[{i}/{len(args.urls)}] {title}  [{content_hash[:8]}]")
            print(f"  {url}")
            if summary:
                print(f"  {summary}")
            print(f"  handles: {len(handles)}" + (f"  (dupes: {dup_count})" if dup_count else ""))
            for j, htxt in enumerate(handles, start=1):
                prefix = f"    {j:02d}) "
                wrapped = textwrap.fill(
                    htxt,
                    width=92,
                    initial_indent=prefix,
                    subsequent_indent=" " * len(prefix),
                )
                print(wrapped)

            run["documents"].append(
                {
                    "url": url,
                    "content_hash": content_hash,
                    "title": meta.get("title"),
                    "author": meta.get("author"),
                    "date": meta.get("date"),
                    "summary": summary,
                    "handles": handles,
                }
            )

        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print("=" * 92)
            print(f"Wrote: {args.out}")

    elif args.command == "probe-compare":
        a = json.loads(args.a.read_text(encoding="utf-8"))
        b = json.loads(args.b.read_text(encoding="utf-8"))

        def index(d: dict[str, Any]) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for doc in d.get("documents", []):
                out[str(doc.get("url"))] = doc
            return out

        ia = index(a)
        ib = index(b)
        urls = sorted(set(ia) | set(ib))

        print("=" * 92)
        print(f"Compare: {args.a}  vs  {args.b}")
        for url in urls:
            da = ia.get(url)
            db = ib.get(url)
            print("-" * 92)
            print(url)
            if da is None:
                print("  only in B")
                continue
            if db is None:
                print("  only in A")
                continue

            ha = set((h or "").strip() for h in da.get("handles", []) if h)
            hb = set((h or "").strip() for h in db.get("handles", []) if h)
            added = sorted(hb - ha)
            removed = sorted(ha - hb)
            inter = len(ha & hb)
            union = max(1, len(ha | hb))
            jacc = inter / union
            print(f"  A: {len(ha)} handles   B: {len(hb)} handles   Jaccard: {jacc:.3f}")
            if added:
                print("  + added:")
                for h in added:
                    print(f"    + {h}")
            if removed:
                print("  - removed:")
                for h in removed:
                    print(f"    - {h}")

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
            return

        if args.no_dedup:
            for r in results[: args.k]:
                doc = collapse_hits_by_document([r])[0]
                print(format_result(doc, query=args.query, snippet_chars=args.snippet_chars, width=args.width))
                print()
        else:
            docs = collapse_hits_by_document(results)
            docs, applied = filter_by_score(docs, drop_bottom_frac=args.drop_bottom, min_score=args.min_rrf)

            if not docs:
                msg = "No results above threshold."
                if applied is not None:
                    msg += f" (threshold={applied:.6f})"
                print(msg)
                return

            for doc in docs[: args.k]:
                print(format_result(doc, query=args.query, snippet_chars=args.snippet_chars, width=args.width))
                print()

        if args.debug:
            for name, hits in debug_lists.items():
                print(f"[{name}] {len(hits)} hits")


if __name__ == "__main__":
    main()
