"""LLM-based metadata and topic extraction."""

import json
from openai import OpenAI

SYSTEM_PROMPT = """Extract metadata and topic descriptors from the document.

Output 1 topic descriptor unless the document genuinely covers multiple distinct subjects. Most documents need only 1-2. Do not artificially split a focused piece.

Each topic descriptor should be a paragraph of 2-4 sentences describing a key theme, written to enable semantic search.

Respond with JSON only:
{
  "title": "document title if identifiable, else null",
  "author": "author if identifiable, else null", 
  "date": "publication date if identifiable, else null",
  "topics": ["First topic paragraph...", "Second if needed..."]
}"""


def extract_metadata(content: str) -> dict:
    """Extract title, author, date, and topics from content."""
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # cheap, good enough for extraction
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content[:50000]},  # truncate if huge
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    
    return json.loads(response.choices[0].message.content)
