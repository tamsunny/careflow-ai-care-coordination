"""Persistent local Qdrant index for synthetic clinic policy text."""

import json
import math
import re
from collections.abc import Callable, Sequence
from numbers import Real
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from .policy_corpus import SYNTHETIC_POLICY_CORPUS
from .policy_models import (
    PolicyChunk,
    PolicyDocument,
    PolicyIndexResult,
    PolicySearchResult,
)

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]
DEFAULT_PERSIST_DIRECTORY = Path("data/qdrant_policy_index")
MANIFEST_VERSION = 1


class PolicyIndexError(RuntimeError):
    """Base error for local policy indexing and retrieval."""


class PolicyIndexConfigurationError(PolicyIndexError):
    """The persisted index configuration does not match this indexer."""


class PolicyEmbeddingError(PolicyIndexError):
    """The configured embedder failed or returned invalid vectors."""


class PolicyIndexDataError(PolicyIndexError):
    """A stored point is missing required policy metadata."""


class LocalPolicyIndex:
    """Index and search synthetic policy documents in local persistent Qdrant.

    The embedding function and its model ID are supplied by the caller. The
    persisted manifest prevents a later process from querying or indexing the
    collection with a different model or vector dimension.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        embedding_model_id: str,
        embedding_dimension: int,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = "synthetic_policy_handbook",
        max_chunk_chars: int = 800,
    ) -> None:
        if not callable(embedder):
            raise PolicyIndexConfigurationError("A text embedder function is required")
        if not isinstance(embedding_model_id, str) or not embedding_model_id.strip():
            raise PolicyIndexConfigurationError("An embedding model ID is required")
        if (
            isinstance(embedding_dimension, bool)
            or not isinstance(embedding_dimension, int)
            or embedding_dimension < 1
        ):
            raise PolicyIndexConfigurationError(
                "embedding_dimension must be a positive integer"
            )
        if not isinstance(collection_name, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{3,64}", collection_name
        ):
            raise PolicyIndexConfigurationError(
                "collection_name must be 3-64 letters, digits, underscores, or hyphens"
            )
        if (
            isinstance(max_chunk_chars, bool)
            or not isinstance(max_chunk_chars, int)
            or max_chunk_chars < 40
        ):
            raise PolicyIndexConfigurationError(
                "max_chunk_chars must be an integer of at least 40"
            )

        self.embedding_model_id = embedding_model_id.strip()
        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name
        self.max_chunk_chars = max_chunk_chars
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._manifest_path = self.persist_directory / (
            f".policy-index-{self.collection_name}.json"
        )
        self._client = QdrantClient(path=str(self.persist_directory))
        try:
            collection_exists = self._client.collection_exists(self.collection_name)
            manifest_exists = self._manifest_path.exists()
            if manifest_exists:
                self._validate_manifest()
            elif collection_exists:
                raise PolicyIndexConfigurationError(
                    "Existing Qdrant collection has no model manifest; use a new "
                    "persistent directory to avoid mixing embedding configurations"
                )
            else:
                self._write_manifest()

            if not collection_exists:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
            self._validate_collection()
        except Exception:
            self._client.close()
            raise

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LocalPolicyIndex":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def index_documents(
        self,
        documents: Sequence[PolicyDocument | dict[str, Any]] = SYNTHETIC_POLICY_CORPUS,
    ) -> PolicyIndexResult:
        """Synchronize this collection to the supplied complete corpus.

        Deterministic point IDs make repeated indexing idempotent. Points no
        longer represented by the supplied corpus are removed after upserting.
        """
        validated_documents = tuple(
            PolicyDocument.model_validate(document) for document in documents
        )
        if not validated_documents:
            raise PolicyIndexConfigurationError(
                "Refusing to replace the policy index with an empty corpus"
            )
        source_ids = [document.source_id for document in validated_documents]
        if len(source_ids) != len(set(source_ids)):
            raise PolicyIndexConfigurationError(
                "Each policy source_id must be unique in the corpus"
            )

        chunks = tuple(
            chunk
            for document in validated_documents
            for chunk in _split_document(document, self.max_chunk_chars)
        )
        vectors = self._embed([chunk.text for chunk in chunks])
        points = [
            PointStruct(
                id=_point_id(self.embedding_model_id, chunk),
                vector=vector,
                payload={
                    "source_id": chunk.source_id,
                    "title": chunk.title,
                    "section_name": chunk.section_name,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "embedding_model_id": self.embedding_model_id,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        existing_ids = self._all_point_ids()
        expected_ids = {point.id for point in points}
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        stale_ids = [point_id for point_id in existing_ids if point_id not in expected_ids]
        if stale_ids:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=stale_ids),
                wait=True,
            )
        return PolicyIndexResult(
            documents_indexed=len(validated_documents),
            chunks_indexed=len(chunks),
            stale_chunks_removed=len(stale_ids),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_id: str | None = None,
        section_name: str | None = None,
    ) -> tuple[PolicySearchResult, ...]:
        """Return semantic nearest-neighbors, optionally filtered by metadata."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        for filter_name, filter_value in (
            ("source_id", source_id),
            ("section_name", section_name),
        ):
            if filter_value is not None and (
                not isinstance(filter_value, str) or not filter_value.strip()
            ):
                raise ValueError(f"{filter_name} filter must be a non-empty string")

        vector = self._embed([query.strip()])[0]
        filter_conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in (
                ("source_id", source_id),
                ("section_name", section_name),
            )
            if value is not None
        ]
        query_filter = Filter(must=filter_conditions) if filter_conditions else None
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        results = []
        for hit in response.points:
            payload = hit.payload or {}
            try:
                results.append(
                    PolicySearchResult(
                        chunk_id=str(hit.id),
                        source_id=payload["source_id"],
                        title=payload["title"],
                        section_name=payload["section_name"],
                        text=payload["text"],
                        score=float(hit.score),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PolicyIndexDataError(
                    "Qdrant returned a policy point with incomplete metadata"
                ) from exc
        return tuple(results)

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            raw_vectors = self._embedder(texts)
        except Exception as exc:
            raise PolicyEmbeddingError("The configured text embedder failed") from exc
        if not isinstance(raw_vectors, Sequence) or isinstance(raw_vectors, (str, bytes)):
            raise PolicyEmbeddingError("Embedder must return a sequence of vectors")
        if len(raw_vectors) != len(texts):
            raise PolicyEmbeddingError(
                "Embedder must return exactly one vector for each input text"
            )

        vectors: list[list[float]] = []
        for vector in raw_vectors:
            if (
                not isinstance(vector, Sequence)
                or isinstance(vector, (str, bytes))
                or len(vector) != self.embedding_dimension
            ):
                raise PolicyEmbeddingError(
                    "Embedding vector dimension does not match the configured dimension"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in vector
            ):
                raise PolicyEmbeddingError(
                    "Embedding vectors must contain only finite numeric values"
                )
            vectors.append([float(value) for value in vector])
        return vectors

    def _all_point_ids(self) -> list[str | int]:
        point_ids: list[str | int] = []
        offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.extend(point.id for point in points)
            if next_offset is None:
                return point_ids
            offset = next_offset

    def _validate_manifest(self) -> None:
        try:
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PolicyIndexConfigurationError(
                "The local policy index manifest is unreadable"
            ) from exc
        expected = {
            "version": MANIFEST_VERSION,
            "collection_name": self.collection_name,
            "embedding_model_id": self.embedding_model_id,
            "embedding_dimension": self.embedding_dimension,
        }
        if manifest != expected:
            raise PolicyIndexConfigurationError(
                "The existing policy index uses a different collection or embedding "
                "configuration; use a new persistent directory to rebuild it"
            )

    def _write_manifest(self) -> None:
        manifest = {
            "version": MANIFEST_VERSION,
            "collection_name": self.collection_name,
            "embedding_model_id": self.embedding_model_id,
            "embedding_dimension": self.embedding_dimension,
        }
        temporary_path = self._manifest_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self._manifest_path)

    def _validate_collection(self) -> None:
        collection = self._client.get_collection(self.collection_name)
        vectors = collection.config.params.vectors
        vector_size = getattr(vectors, "size", None)
        if vector_size != self.embedding_dimension:
            raise PolicyIndexConfigurationError(
                "The existing Qdrant collection has an unexpected vector dimension"
            )


def _split_document(
    document: PolicyDocument,
    max_chunk_chars: int,
) -> tuple[PolicyChunk, ...]:
    chunks: list[PolicyChunk] = []
    for section in document.sections:
        for chunk_index, text in enumerate(
            _split_text(section.text, max_chunk_chars)
        ):
            chunks.append(
                PolicyChunk(
                    source_id=document.source_id,
                    title=document.title,
                    section_name=section.name,
                    chunk_index=chunk_index,
                    text=text,
                )
            )
    return tuple(chunks)


def _split_text(text: str, max_chunk_chars: int) -> tuple[str, ...]:
    normalized = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        parts = (
            _split_long_sentence(sentence, max_chunk_chars)
            if len(sentence) > max_chunk_chars
            else [sentence]
        )
        for part in parts:
            candidate = f"{current} {part}".strip() if current else part
            if current and len(candidate) > max_chunk_chars:
                chunks.append(current)
                current = part
            else:
                current = candidate
    if current:
        chunks.append(current)
    return tuple(chunks)


def _split_long_sentence(sentence: str, max_chunk_chars: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for word in sentence.split():
        if len(word) > max_chunk_chars:
            if current:
                parts.append(current)
                current = ""
            word_parts = [
                word[start : start + max_chunk_chars]
                for start in range(0, len(word), max_chunk_chars)
            ]
            parts.extend(word_parts[:-1])
            current = word_parts[-1]
            continue
        candidate = f"{current} {word}".strip() if current else word
        if current and len(candidate) > max_chunk_chars:
            parts.append(current)
            current = word
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _point_id(embedding_model_id: str, chunk: PolicyChunk) -> str:
    identity = "\x1f".join(
        (
            embedding_model_id,
            chunk.source_id,
            chunk.section_name,
            str(chunk.chunk_index),
        )
    )
    return str(uuid5(NAMESPACE_URL, identity))
