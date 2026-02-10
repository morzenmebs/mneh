"""LLM-based metadata + retrieval-handle extraction.

We generate two different artifacts:
1) `summary` (for display)
2) `handles` (for retrieval)

Handles are *query-shaped strings* — short noun phrases / keyword bundles / a few
question-shaped items — so that we avoid the "semantic midpoint" failure mode
of long catch-all summaries.
"""

from __future__ import annotations

import json
from openai import OpenAI


SYSTEM_PROMPT = """Extract metadata and generate retrieval handles from documents.

You MUST output JSON only.

GOALS
- `summary`: for humans. 1-3 sentences. 30-80 words.
- `handles`: for retrieval. Many short query-shaped strings that cover distinct
  aspects of the document without collapsing them into one "midpoint".

OUTPUT JSON SHAPE
{
  "title": string|null,
  "author": string|null,
  "date": string|null,
  "summary": string|null,
  "handles": [string, ...]
}

RULES FOR HANDLES
- Produce 12–30 handles.
- Each handle MUST be 3–12 words.
- Handles should look like what someone would type into a search box:
  - keyword bundles / noun phrases (preferred)
  - occasional question-shaped items (<= 30% of handles)
  - occasional "how to ..." or "why ..." phrasing is fine
- Avoid duplicates and near-duplicates.
- Cover *different* domains / subthemes separately if the document is broad.
- Include concrete entities: names, terms, frameworks, jargon.
- Do NOT include citations, markdown, or numbering.

METADATA RULES
- Use provided hints if they seem accurate; infer from URL if author/site is obvious.
- If metadata isn't clearly present and can't be inferred, use null.

EXAMPLES

Input: [Article about React hooks and state management patterns]
Output: {
  "title": "Modern React State Management",
  "author": "Sarah Chen",
  "date": "2024-03-15",
  "summary": "A practical overview of state management patterns in modern React, focusing on hooks and performance tradeoffs.",
  "handles": [
    "React useState vs useReducer",
    "useContext for shared state",
    "custom hooks reusable logic",
    "state colocation best practices",
    "React performance useMemo useCallback",
    "React memoization pitfalls",
    "Redux vs Zustand comparison",
    "form state management patterns",
    "async data fetching caching",
    "lifting state up composition"
  ]
}

Input: [Essay covering historical philosophy and modern AI ethics]
Output: {
  "title": "From Aristotle to Algorithms",
  "author": null,
  "date": "2024",
  "summary": "Connects Aristotelian virtue ethics to contemporary debates in AI alignment and machine ethics, contrasting character-based and rule-based approaches.",
  "handles": [
    "Aristotle virtue ethics eudaimonia",
    "phronesis practical wisdom",
    "golden mean moral character",
    "moral psychology character development",
    "AI alignment value learning",
    "reward hacking specification gaming",
    "machine ethics deontology vs consequentialism",
    "how to encode human values",
    "virtue ethics for AI systems",
    "normative constraints in alignment"
  ]
}

Now extract from the following document. Respond with JSON only."""


def normalize_extracted_metadata(payload: object) -> dict:
    """Coerce model output to the expected metadata shape."""
    base = {
        "title": None,
        "author": None,
        "date": None,
        "summary": None,
        "handles": [],
    }
    if not isinstance(payload, dict):
        return base

    out = dict(base)
    for field in ("title", "author", "date", "summary"):
        value = payload.get(field)
        if isinstance(value, str):
            clean = value.strip()
            out[field] = clean or None

    raw_handles = payload.get("handles")
    handles: list[str] = []
    if isinstance(raw_handles, list):
        for item in raw_handles:
            if isinstance(item, str):
                clean = item.strip()
                if clean:
                    handles.append(clean)
    out["handles"] = handles
    return out


def extract_metadata(
    content: str,
    url: str | None = None,
    hints: dict | None = None,
    model: str = "gpt-5.2",
) -> dict:
    """Extract title, author, date, summary, handles."""
    client = OpenAI()

    # Build user message with context
    user_content = ""
    if url:
        user_content += f"URL: {url}\n\n"
    if hints:
        hint_parts = [f"{k}: {v}" for k, v in hints.items() if v]
        if hint_parts:
            user_content += "Metadata hints (verify/use if accurate):\n" + "\n".join(hint_parts) + "\n\n"
    user_content += f"Document:\n{content[:50000]}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )

    raw = json.loads(response.choices[0].message.content)
    return normalize_extracted_metadata(raw)
