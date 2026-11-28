"""Persistent local Qdrant index for synthetic clinic policy text."""

import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from numbers import Real
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from .policy_ingestion import PARSER_VERSION as DEFAULT_PARSER_VERSION
from .policy_corpus import SYNTHETIC_POLICY_CORPUS
from .policy_models import (
    PolicyChunk,
    PolicyDocument,
    PolicyIndexResult,
    PolicySearchResult,
)

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]
DEFAULT_PERSIST_DIRECTORY = Path("data/qdrant_policy_index")
MANIFEST_VERSION = 2
INDEX_SCHEMA_VERSION = 3
CHUNKING_VERSION = "sentence-page-v1"


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
        parser_version: str = DEFAULT_PARSER_VERSION,
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
        if not isinstance(parser_version, str) or not parser_version.strip():
            raise PolicyIndexConfigurationError("A parser version is required")

        self.embedding_model_id = embedding_model_id.strip()
        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name
        self.max_chunk_chars = max_chunk_chars
        self.parser_version = parser_version.strip()
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._legacy_manifest = False
        self._incompatible_pipeline = False
        self._corpus_fingerprint: str | None = None
        self._index_state = "ready"
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

            if not collection_exists:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
                self._write_manifest()
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
        *,
        reindex: bool = False,
        skip_if_unchanged: bool = False,
    ) -> PolicyIndexResult:
        """Synchronize this collection to the supplied complete corpus.

        Deterministic point IDs make repeated indexing idempotent. Points no
        longer represented by the supplied corpus are removed after upserting.
        Changed corpus or pipeline versions require the explicit ``reindex`` flag.
        ``skip_if_unchanged`` validates the stored point IDs and avoids embedding
        or upserting an already-current corpus.
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
        if not isinstance(reindex, bool):
            raise PolicyIndexConfigurationError("reindex must be a boolean")
        if not isinstance(skip_if_unchanged, bool):
            raise PolicyIndexConfigurationError(
                "skip_if_unchanged must be a boolean"
            )
        if reindex and skip_if_unchanged:
            raise PolicyIndexConfigurationError(
                "skip_if_unchanged cannot be combined with reindex"
            )

        corpus_fingerprint = _corpus_fingerprint(validated_documents)
        if (self._legacy_manifest or self._incompatible_pipeline) and not reindex:
            raise PolicyIndexConfigurationError(
                "The existing index uses an incompatible M3 or ingestion pipeline "
                "version. Pass reindex=True to explicitly update this collection, "
                "or create a separate versioned collection."
            )
        if (
            self._corpus_fingerprint is not None
            and self._corpus_fingerprint != corpus_fingerprint
            and not reindex
        ):
            raise PolicyIndexConfigurationError(
                "The policy corpus changed; pass reindex=True to explicitly "
                "synchronize this collection."
            )

        chunks = tuple(
            chunk
            for document in validated_documents
            for chunk in _split_document(document, self.max_chunk_chars)
        )
        if (
            skip_if_unchanged
            and self._corpus_fingerprint == corpus_fingerprint
            and self._index_state == "ready"
        ):
            expected_ids = {
                _point_id(self.embedding_model_id, chunk) for chunk in chunks
            }
            if expected_ids != set(self._all_point_ids()):
                raise PolicyIndexConfigurationError(
                    "The stored M4 index does not match its manifest; use a new "
                    "versioned collection instead of silently repairing it."
                )
            return PolicyIndexResult(
                documents_indexed=len(validated_documents),
                chunks_indexed=len(chunks),
                stale_chunks_removed=0,
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
                    "category": chunk.category,
                    "specialty": chunk.specialty,
                    "plan_ids": list(chunk.plan_ids),
                    "version": chunk.version,
                    "effective_date": (
                        chunk.effective_date.isoformat()
                        if chunk.effective_date is not None
                        else None
                    ),
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "synthetic_data": chunk.synthetic_data,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        existing_ids = self._all_point_ids()
        expected_ids = {point.id for point in points}
        stale_ids = [
            point_id for point_id in existing_ids if point_id not in expected_ids
        ]
        self._corpus_fingerprint = corpus_fingerprint
        self._index_state = "indexing"
        self._write_manifest()
        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            if stale_ids:
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=stale_ids),
                    wait=True,
                )
        except Exception:
            self._incompatible_pipeline = True
            raise
        self._legacy_manifest = False
        self._incompatible_pipeline = False
        self._index_state = "ready"
        self._write_manifest()
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
        category: str | None = None,
        specialty: str | None = None,
        plan_id: str | None = None,
        version: str | None = None,
    ) -> tuple[PolicySearchResult, ...]:
        """Return semantic nearest-neighbors, optionally filtered by metadata."""
        _validate_search_options(
            query,
            top_k,
            source_id=source_id,
            section_name=section_name,
            category=category,
            specialty=specialty,
            plan_id=plan_id,
            version=version,
        )
        self._assert_index_searchable()
        vector = self._embed([query.strip()])[0]
        query_filter = _metadata_filter(
            source_id=source_id,
            section_name=section_name,
            category=category,
            specialty=specialty,
            plan_id=plan_id,
            version=version,
        )
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        results = [
            _result_from_payload(hit.id, hit.payload, dense_score=float(hit.score))
            for hit in response.points
        ]
        return tuple(
            sorted(
                results,
                key=lambda result: (
                    -(result.dense_score or 0.0),
                    result.chunk_id,
                ),
            )
        )

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_id: str | None = None,
        section_name: str | None = None,
        category: str | None = None,
        specialty: str | None = None,
        plan_id: str | None = None,
        version: str | None = None,
    ) -> tuple[PolicySearchResult, ...]:
        """Return lexical BM25 results from indexed chunks with the same filters."""
        _validate_search_options(
            query,
            top_k,
            source_id=source_id,
            section_name=section_name,
            category=category,
            specialty=specialty,
            plan_id=plan_id,
            version=version,
        )
        self._assert_index_searchable()
        query_tokens = _tokenize(query)
        if not query_tokens:
            return ()
        query_filter = _metadata_filter(
            source_id=source_id,
            section_name=section_name,
            category=category,
            specialty=specialty,
            plan_id=plan_id,
            version=version,
        )
        points = self._scroll_policy_points(query_filter)
        tokenized_documents = [
            _tokenize((point.payload or {}).get("text", "")) for point in points
        ]
        if not tokenized_documents or not any(
            set(query_tokens).intersection(document_tokens)
            for document_tokens in tokenized_documents
        ):
            return ()

        bm25 = BM25Okapi(tokenized_documents)
        scores = bm25.get_scores(query_tokens)
        results = []
        for point, document_tokens, raw_score in zip(
            points, tokenized_documents, scores
        ):
            if not set(query_tokens).intersection(document_tokens):
                continue
            result = _result_from_payload(point.id, point.payload)
            results.append(result.model_copy(update={"bm25_score": float(raw_score)}))
        return tuple(
            sorted(
                results,
                key=lambda result: (
                    -(result.bm25_score or 0.0),
                    result.chunk_id,
                ),
            )[:top_k]
        )

    def _assert_index_searchable(self) -> None:
        if self._index_state != "ready":
            raise PolicyIndexConfigurationError(
                "The policy index is incomplete; explicitly reindex it before searching"
            )

    def _scroll_policy_points(self, query_filter: Filter | None) -> list[Any]:
        points = []
        offset = None
        while True:
            batch, offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if offset is None:
                return points

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
        if not isinstance(manifest, dict):
            raise PolicyIndexConfigurationError(
                "The local policy index manifest must contain a JSON object"
            )
        legacy_manifest = {
            "version": 1,
            "collection_name": self.collection_name,
            "embedding_model_id": self.embedding_model_id,
            "embedding_dimension": self.embedding_dimension,
        }
        if manifest == legacy_manifest:
            self._legacy_manifest = True
            return

        expected = self._manifest_configuration()
        if set(manifest) != set(expected) | {"corpus_fingerprint", "index_state"}:
            raise PolicyIndexConfigurationError(
                "The existing policy index manifest has an incompatible schema. "
                "Use a new versioned collection or persistent directory."
            )
        if (
            manifest.get("collection_name") != self.collection_name
            or manifest.get("embedding_model_id") != self.embedding_model_id
            or manifest.get("embedding_dimension") != self.embedding_dimension
        ):
            raise PolicyIndexConfigurationError(
                "The existing policy index uses a different collection or embedding "
                "configuration. Use a new versioned collection or persistent directory."
            )
        self._incompatible_pipeline = any(
            manifest.get(key) != value
            for key, value in expected.items()
            if key not in {
                "collection_name",
                "embedding_model_id",
                "embedding_dimension",
            }
        )
        index_state = manifest.get("index_state")
        if index_state not in {"ready", "indexing"}:
            raise PolicyIndexConfigurationError(
                "The policy index manifest has an invalid indexing state"
            )
        self._index_state = index_state
        if index_state != "ready":
            self._incompatible_pipeline = True
        corpus_fingerprint = manifest.get("corpus_fingerprint")
        if corpus_fingerprint is not None and (
            not isinstance(corpus_fingerprint, str)
            or not re.fullmatch(r"[a-f0-9]{64}", corpus_fingerprint)
        ):
            raise PolicyIndexConfigurationError(
                "The policy index manifest has an invalid corpus fingerprint"
            )
        self._corpus_fingerprint = corpus_fingerprint

    def _write_manifest(self) -> None:
        manifest = {
            **self._manifest_configuration(),
            "corpus_fingerprint": self._corpus_fingerprint,
            "index_state": self._index_state,
        }
        temporary_path = self._manifest_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self._manifest_path)

    def _manifest_configuration(self) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "collection_name": self.collection_name,
            "embedding_model_id": self.embedding_model_id,
            "embedding_dimension": self.embedding_dimension,
            "parser_version": self.parser_version,
            "chunking_version": CHUNKING_VERSION,
            "max_chunk_chars": self.max_chunk_chars,
        }

    def _validate_collection(self) -> None:
        collection = self._client.get_collection(self.collection_name)
        vectors = collection.config.params.vectors
        vector_size = getattr(vectors, "size", None)
        if vector_size != self.embedding_dimension:
            raise PolicyIndexConfigurationError(
                "The existing Qdrant collection has an unexpected vector dimension"
            )


def _validate_search_options(
    query: str,
    top_k: int,
    *,
    source_id: str | None,
    section_name: str | None,
    category: str | None,
    specialty: str | None,
    plan_id: str | None,
    version: str | None,
) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    for filter_name, filter_value in (
        ("source_id", source_id),
        ("section_name", section_name),
        ("category", category),
        ("specialty", specialty),
        ("plan_id", plan_id),
        ("version", version),
    ):
        if filter_value is not None and (
            not isinstance(filter_value, str) or not filter_value.strip()
        ):
            raise ValueError(f"{filter_name} filter must be a non-empty string")
    if plan_id is not None and not re.fullmatch(r"SYN-PLAN-[A-Z0-9-]+", plan_id):
        raise ValueError("plan_id filter must use the synthetic SYN-PLAN prefix")


def _metadata_filter(
    *,
    source_id: str | None,
    section_name: str | None,
    category: str | None,
    specialty: str | None,
    plan_id: str | None,
    version: str | None,
) -> Filter | None:
    values = (
        ("source_id", source_id),
        ("section_name", section_name),
        ("category", category),
        ("specialty", specialty),
        ("plan_ids", plan_id),
        ("version", version),
    )
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in values
        if value is not None
    ]
    return Filter(must=conditions) if conditions else None


def _result_from_payload(
    point_id: str | int,
    raw_payload: dict[str, Any] | None,
    *,
    dense_score: float | None = None,
) -> PolicySearchResult:
    payload = raw_payload or {}
    try:
        return PolicySearchResult(
            chunk_id=str(point_id),
            source_id=payload["source_id"],
            title=payload["title"],
            section_name=payload["section_name"],
            text=payload["text"],
            score=dense_score,
            dense_score=dense_score,
            category=payload.get("category"),
            specialty=payload.get("specialty"),
            plan_ids=tuple(payload.get("plan_ids") or ()),
            version=payload.get("version"),
            effective_date=payload.get("effective_date"),
            page_start=payload.get("page_start"),
            page_end=payload.get("page_end"),
            synthetic_data=payload.get("synthetic_data", True),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyIndexDataError(
            "Qdrant returned a policy point with incomplete metadata"
        ) from exc


def _tokenize(text: str) -> list[str]:
    return re.findall(
        r"syn-(?:plan|policy)-[a-z0-9-]+|[a-z0-9]+",
        text.casefold(),
    )


def _split_document(
    document: PolicyDocument,
    max_chunk_chars: int,
) -> tuple[PolicyChunk, ...]:
    chunks: list[PolicyChunk] = []
    for section in document.sections:
        page_texts = (
            tuple((page.page_number, page.text) for page in section.pages)
            if section.pages
            else ((None, section.text),)
        )
        chunk_index = 0
        for page_number, page_text in page_texts:
            for text in _split_text(page_text, max_chunk_chars):
                chunks.append(
                    PolicyChunk(
                        source_id=document.source_id,
                        title=document.title,
                        section_name=section.name,
                        chunk_index=chunk_index,
                        text=text,
                        category=document.category,
                        specialty=document.specialty,
                        plan_ids=document.plan_ids,
                        version=document.version,
                        effective_date=document.effective_date,
                        page_start=page_number,
                        page_end=page_number,
                        synthetic_data=document.synthetic_data,
                    )
                )
                chunk_index += 1
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


def _corpus_fingerprint(documents: Sequence[PolicyDocument]) -> str:
    canonical_documents = [
        document.model_dump(mode="json", exclude_none=True)
        for document in documents
    ]
    serialized = json.dumps(
        canonical_documents,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
