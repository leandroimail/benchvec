"""Chunking helpers for the raw Simple Wikipedia fallback pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, cast


class ChunkLike(Protocol):
    """Structural type for chunks returned by chonkie."""

    text: str
    token_count: int


class TokenChunkerProtocol(Protocol):
    """Callable protocol for token chunkers."""

    def __call__(self, text: str) -> Sequence[ChunkLike]: ...


@dataclass(slots=True)
class ChunkedRecord:
    """Prepared chunk record ready for local embedding."""

    chunk_id: str
    parent_document_id: str
    text: str
    token_count: int


def build_token_chunker(
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> TokenChunkerProtocol:
    """Build the deterministic token chunker mandated by PRD03."""

    from chonkie import TokenChunker  # type: ignore[attr-defined]

    return cast(
        TokenChunkerProtocol,
        TokenChunker(
            tokenizer=tokenizer_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
    )


def chunk_document(
    document_id: str,
    text: str,
    chunker: TokenChunkerProtocol,
) -> list[ChunkedRecord]:
    """Split a canonical document into deterministic chunks."""

    chunks = chunker(text)
    chunked_records: list[ChunkedRecord] = []
    for index, chunk in enumerate(chunks):
        chunked_records.append(
            ChunkedRecord(
                chunk_id=f"{document_id}_c{index}",
                parent_document_id=document_id,
                text=chunk.text,
                token_count=chunk.token_count,
            )
        )
    return chunked_records


def chunker_to_metadata(
    tokenizer_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    """Return serializable chunking parameters for manifests."""

    return {
        "tokenizer": tokenizer_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
