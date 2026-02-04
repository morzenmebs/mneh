# mneh

Local-first capture + retrieval for building an exocortex without manual metadata work.

## Install

```bash
cd mneh
uv sync
export OPENAI_API_KEY="..."
```

## Usage

```bash
# Capture a URL
mneh capture "https://example.com/article"

# Search
mneh search "coordination failure game theory"
mneh search "react hooks" -v          # verbose: show handles/chunks with scores

# List captures
mneh list                              # hash, date, title
mneh list -u                           # add URLs
mneh list -v                           # add chunk/handle counts
mneh list -uv                          # both

# Inspect a capture
mneh show a1b2c3d4                     # by hash prefix
mneh show --last                       # most recent
mneh show -q "moloch"                  # top result for query
mneh show a1b2c3d4 -v                  # add handles list
mneh show a1b2c3d4 -vv                 # add chunk previews

# Delete a capture
mneh delete a1b2c3d4

# JSON output (for any command)
mneh list --json
mneh show --last --json
mneh search "query" --json
```

## Architecture

- **Captures**: raw text stored content-addressed in `~/.mneh/storage/`
- **Handles**: 12-30 query-shaped strings per document (Doc2Query-style)
- **Chunks**: semantic paragraph-based chunking (~500 tokens)
- **Search**: hybrid FTS + vector, fused with Reciprocal Rank Fusion

## RRF Score Guide

With k=60 and 4 channels (FTS chunks, FTS handles, vec chunks, vec handles):

- `>0.05`: strong hit (top ranks in multiple channels)
- `0.02-0.05`: moderate (mid-ranks or single-channel)
- `<0.02`: weak (filtered as noise by default)
