"""LLM-based metadata and topic extraction."""

import json
from openai import OpenAI

SYSTEM_PROMPT = """Extract metadata and topic descriptors from documents for semantic search indexing.

RULES:
- Each topic MUST be 75-150 words. Count carefully.
- Default to 1 topic for focused documents
- BUT: if the document applies ideas across 3+ distinct domains (e.g., game theory AND academic publishing AND AI safety AND economics), create separate topics for each domain cluster so they retrieve independently
- Topics should be keyword-rich for search, not narrative summaries
- Include specific terms, names, concepts, and jargon someone searching would use
- Use provided hints if they seem accurate; infer from URL if author/site is obvious
- If metadata isn't clearly present and can't be inferred, use null

EXAMPLES:

Input: [Article about React hooks and state management patterns]
Output: {"title": "Modern React State Management", "author": "Sarah Chen", "date": "2024-03-15", "topics": ["React hooks patterns and state management strategies including useState, useReducer, useContext for local and shared state. Custom hooks for reusable logic extraction. Comparison with Redux and Zustand for global state. Performance optimization with useMemo, useCallback, React.memo. State colocation principles, lifting state up vs composition. Practical patterns for form handling, async data fetching, and caching."]}

Input: [Essay covering both historical philosophy and modern AI ethics]
Output: {"title": "From Aristotle to Algorithms", "author": null, "date": "2024", "topics": ["Ancient Greek virtue ethics and Aristotelian thought: eudaimonia, practical wisdom (phronesis), moral character development, the golden mean. Connections to contemporary character-based ethical frameworks and moral psychology.", "Artificial intelligence alignment and machine ethics: value learning, reward hacking, specification gaming. Challenges in encoding human values into optimization objectives. Debates between deontological constraints versus consequentialist approaches in AI safety research."]}

Input: [Blog post about sourdough bread baking techniques]
Output: {"title": null, "author": "The Home Baker", "date": null, "topics": ["Sourdough bread fermentation and baking techniques: starter maintenance, hydration ratios, autolyse method, bulk fermentation timing, stretch-and-fold development, cold retard proofing. Scoring patterns, dutch oven steam methods, crust development. Troubleshooting dense crumb, poor oven spring, over-proofing signs."]}

Now extract from the following document. Respond with JSON only:"""


def extract_metadata(content: str, url: str = None, hints: dict = None) -> dict:
    """Extract title, author, date, and topics from content."""
    client = OpenAI()
    
    # Build user message with context
    user_content = ""
    if url:
        user_content += f"URL: {url}\n\n"
    if hints:
        hint_parts = [f"{k}: {v}" for k, v in hints.items() if v]
        if hint_parts:
            user_content += f"Metadata hints (verify/use if accurate):\n" + "\n".join(hint_parts) + "\n\n"
    user_content += f"Document:\n{content[:50000]}"
    
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    
    return json.loads(response.choices[0].message.content)
