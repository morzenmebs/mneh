"""Embedding via OpenAI."""

from openai import OpenAI

HARD_LIMIT_TOKENS = 7500


def embed_texts(texts: list[str], model: str = "text-embedding-3-large") -> list[list[float]]:
    """Embed a batch of texts, return list of vectors."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    
    client = OpenAI()
    
    # Safety truncation if somehow still too long
    safe_texts = []
    for t in texts:
        tokens = enc.encode(t)
        if len(tokens) > HARD_LIMIT_TOKENS:
            t = enc.decode(tokens[:HARD_LIMIT_TOKENS])
        safe_texts.append(t)
    
    response = client.embeddings.create(
        model=model,
        input=safe_texts,
    )
    
    return [item.embedding for item in response.data]


def embed_single(text: str, model: str = "text-embedding-3-large") -> list[float]:
    """Embed a single text."""
    return embed_texts([text], model)[0]
