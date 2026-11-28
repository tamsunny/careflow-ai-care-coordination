"""Independent dense, BM25, RRF, and cross-encoder policy retrieval."""

import math
from collections.abc import Callable, Sequence
from numbers import Real
from typing import Any

from .policy_index import LocalPolicyIndex
from .policy_models import PolicySearchFilters, PolicySearchResult

ScorePairs = Callable[[Sequence[tuple[str, str]]], Sequence[Real]]


class PolicyRetrievalError(RuntimeError):
    """Base error for hybrid retrieval and reranking."""


class PolicyRerankerError(PolicyRetrievalError):
    """The injected cross-encoder scorer failed or returned invalid scores."""


class PolicyRetrievalConfigurationError(PolicyRetrievalError):
    """Hybrid retrieval or reranking configuration is invalid."""


class CrossEncoderReranker:
    """Apply a caller-supplied cross-encoder scorer to query/passage pairs.

    A local model's ``CrossEncoder.predict`` method can be injected as
    ``score_pairs``. The project does not select or download a model implicitly.
    """

    def __init__(self, score_pairs: ScorePairs) -> None:
        if not callable(score_pairs):
            raise PolicyRetrievalConfigurationError(
                "A cross-encoder score_pairs function is required"
            )
        self._score_pairs = score_pairs

    def rerank(
        self,
        query: str,
        candidates: Sequence[PolicySearchResult],
        *,
        top_k: int = 5,
    ) -> tuple[PolicySearchResult, ...]:
        _validate_query_and_limit(query, top_k)
        candidate_tuple = tuple(candidates)
        if not candidate_tuple:
            return ()

        try:
            raw_scores = self._score_pairs(
                tuple((query.strip(), result.text) for result in candidate_tuple)
            )
            scores = list(raw_scores)
        except Exception as exc:
            raise PolicyRerankerError(
                "The configured cross-encoder scorer failed"
            ) from exc
        if len(scores) != len(candidate_tuple):
            raise PolicyRerankerError(
                "The cross-encoder must return one score per candidate"
            )

        scored_candidates = []
        for original_rank, (result, score) in enumerate(
            zip(candidate_tuple, scores)
        ):
            if (
                isinstance(score, bool)
                or not isinstance(score, Real)
                or not math.isfinite(float(score))
            ):
                raise PolicyRerankerError(
                    "Cross-encoder scores must be finite numeric values"
                )
            scored_candidates.append(
                (
                    original_rank,
                    result.model_copy(update={"reranker_score": float(score)}),
                )
            )
        scored_candidates.sort(
            key=lambda item: (-item[1].reranker_score, item[0])
        )
        return tuple(result for _, result in scored_candidates[:top_k])


class PolicyHybridRetriever:
    """Coordinate independently callable dense, BM25, RRF, and reranked search."""

    def __init__(
        self,
        index: LocalPolicyIndex,
        *,
        reranker: CrossEncoderReranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
            raise PolicyRetrievalConfigurationError("rrf_k must be a positive integer")
        self._index = index
        self._reranker = reranker
        self._rrf_k = rrf_k

    def search_dense(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: PolicySearchFilters | None = None,
    ) -> tuple[PolicySearchResult, ...]:
        return self._index.search(
            query,
            top_k=top_k,
            **_filter_kwargs(filters),
        )

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: PolicySearchFilters | None = None,
    ) -> tuple[PolicySearchResult, ...]:
        return self._index.search_bm25(
            query,
            top_k=top_k,
            **_filter_kwargs(filters),
        )

    def search_hybrid(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
        filters: PolicySearchFilters | None = None,
    ) -> tuple[PolicySearchResult, ...]:
        _validate_query_and_limit(query, top_k)
        _validate_query_and_limit(query, candidate_k)
        if candidate_k < top_k:
            raise PolicyRetrievalConfigurationError(
                "candidate_k must be greater than or equal to top_k"
            )
        filter_kwargs = _filter_kwargs(filters)
        dense_results = self._index.search(
            query,
            top_k=candidate_k,
            **filter_kwargs,
        )
        bm25_results = self._index.search_bm25(
            query,
            top_k=candidate_k,
            **filter_kwargs,
        )
        return _reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            rrf_k=self._rrf_k,
            top_k=top_k,
        )

    def search_hybrid_reranked(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
        filters: PolicySearchFilters | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> tuple[PolicySearchResult, ...]:
        _validate_query_and_limit(query, top_k)
        _validate_query_and_limit(query, candidate_k)
        if candidate_k < top_k:
            raise PolicyRetrievalConfigurationError(
                "candidate_k must be greater than or equal to top_k"
            )
        selected_reranker = reranker or self._reranker
        if selected_reranker is None:
            raise PolicyRetrievalConfigurationError(
                "Inject a cross-encoder reranker before requesting reranked results"
            )
        candidates = self.search_hybrid(
            query,
            top_k=candidate_k,
            candidate_k=candidate_k,
            filters=filters,
        )
        return selected_reranker.rerank(query, candidates, top_k=top_k)


def _filter_kwargs(filters: PolicySearchFilters | None) -> dict[str, Any]:
    return {} if filters is None else filters.model_dump(exclude_none=True)


def _validate_query_and_limit(query: str, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


def _reciprocal_rank_fusion(
    dense_results: Sequence[PolicySearchResult],
    bm25_results: Sequence[PolicySearchResult],
    *,
    rrf_k: int,
    top_k: int,
) -> tuple[PolicySearchResult, ...]:
    fused: dict[str, dict[str, Any]] = {}
    for channel, results in (
        ("dense_score", dense_results),
        ("bm25_score", bm25_results),
    ):
        for rank, result in enumerate(results, start=1):
            candidate = fused.setdefault(
                result.chunk_id,
                {
                    "result": result,
                    "dense_score": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                },
            )
            channel_score = (
                result.dense_score if channel == "dense_score" else result.bm25_score
            )
            candidate[channel] = channel_score
            candidate["rrf_score"] += 1.0 / (rrf_k + rank)

    combined = []
    for chunk_id, candidate in fused.items():
        dense_score = candidate["dense_score"]
        result = candidate["result"].model_copy(
            update={
                "score": dense_score,
                "dense_score": dense_score,
                "bm25_score": candidate["bm25_score"],
                "rrf_score": candidate["rrf_score"],
                "reranker_score": None,
            }
        )
        combined.append((chunk_id, result))
    combined.sort(key=lambda item: (-item[1].rrf_score, item[0]))
    return tuple(result for _, result in combined[:top_k])
