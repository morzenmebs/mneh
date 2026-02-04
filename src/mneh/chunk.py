"""Token-aware text chunking."""

import tiktoken

ENCODER = tiktoken.get_encoding("cl100k_base")
MAX_CHUNK_TOKENS = 500  # target size
OVERLAP_TOKENS = 50  # overlap between chunks
HARD_LIMIT_TOKENS = 7500  # safety


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def decode_tokens(tokens: list[int]) -> str:
    return ENCODER.decode(tokens)


def encode_text(text: str) -> list[int]:
    return ENCODER.encode(text)


def chunk_text(text: str, target_tokens: int = MAX_CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    """Split text into overlapping chunks respecting token limits."""
    tokens = encode_text(text)
    
    if len(tokens) <= target_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + target_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(decode_tokens(chunk_tokens))
        
        # Move forward by (target - overlap) to create overlap
        start += target_tokens - overlap
        
        # Avoid tiny final chunk
        if len(tokens) - start < overlap * 2:
            # Just include remainder in last chunk
            if start < len(tokens):
                chunks[-1] = decode_tokens(tokens[start - (target_tokens - overlap):])
            break
    
    return chunks
