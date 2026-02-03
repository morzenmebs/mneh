"""Embedding via OpenAI."""

from openai import OpenAI


def embed_texts(texts: list[str], model: str = "text-embedding-3-large") -> list[list[float]]:
    """Embed a batch of texts, return list of vectors."""
    client = OpenAI()
    
    response = client.embeddings.create(
        model=model,
        input=texts,
    )
    
    return [item.embedding for item in response.data]


def embed_single(text: str, model: str = "text-embedding-3-large") -> list[float]:
    """Embed a single text."""
    return embed_texts([text], model)[0]
