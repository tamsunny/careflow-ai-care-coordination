"""Configured real embedding and CPU cross-encoder adapters for M4."""

import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from .policy_retrieval import CrossEncoderReranker


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PolicyProviderConfigurationError(RuntimeError):
    """A real policy embedding or reranker provider is not configured."""


class PolicyEmbeddingProviderError(RuntimeError):
    """A policy embedding provider failed without exposing provider details."""


class PolicyCrossEncoderProviderError(RuntimeError):
    """The configured CPU cross-encoder failed without exposing details."""


@dataclass(frozen=True)
class EmbeddingProviderConfig:
    model_id: str
    dimension: int
    api_key: str | None = field(default=None, repr=False)
    api_base: str | None = field(default=None, repr=False)


def load_embedding_provider_config(
    environment: Mapping[str, str] | None = None,
) -> EmbeddingProviderConfig:
    values = os.environ if environment is None else environment
    model_id = values.get("CAREFLOW_EMBEDDING_MODEL", "").strip()
    if not model_id:
        raise PolicyProviderConfigurationError(
            "Set CAREFLOW_EMBEDDING_MODEL to a LiteLLM embedding model ID"
        )

    dimension_text = values.get("CAREFLOW_EMBEDDING_DIMENSION", "").strip()
    try:
        dimension = int(dimension_text)
    except (TypeError, ValueError):
        raise PolicyProviderConfigurationError(
            "Set CAREFLOW_EMBEDDING_DIMENSION to the model's output dimension"
        ) from None
    if dimension < 1:
        raise PolicyProviderConfigurationError(
            "CAREFLOW_EMBEDDING_DIMENSION must be a positive integer"
        )

    api_key = values.get("CAREFLOW_EMBEDDING_API_KEY", "").strip() or None
    api_base = values.get("CAREFLOW_EMBEDDING_API_BASE", "").strip() or None
    return EmbeddingProviderConfig(
        model_id=model_id,
        dimension=dimension,
        api_key=api_key,
        api_base=api_base,
    )


def create_litellm_embedder(
    config: EmbeddingProviderConfig | None = None,
) -> Callable[[Sequence[str]], list[list[float]]]:
    """Build a batch embedder using LiteLLM and environment-only credentials."""
    provider_config = config or load_embedding_provider_config()

    def embed_texts(texts: Sequence[str]) -> list[list[float]]:
        inputs = list(texts)
        if not inputs:
            return []

        request: dict[str, Any] = {
            "model": provider_config.model_id,
            "input": inputs,
        }
        if provider_config.api_key:
            request["api_key"] = provider_config.api_key
        if provider_config.api_base:
            request["api_base"] = provider_config.api_base

        try:
            from litellm import embedding

            response = embedding(**request)
            rows = _field(response, "data")
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                raise PolicyEmbeddingProviderError(
                    "Embedding provider returned an invalid response"
                )
            vectors = []
            for row in rows:
                vector = _field(row, "embedding")
                if not isinstance(vector, Sequence) or isinstance(
                    vector, (str, bytes)
                ):
                    raise PolicyEmbeddingProviderError(
                        "Embedding provider returned an invalid vector"
                    )
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    for value in vector
                ):
                    raise PolicyEmbeddingProviderError(
                        "Embedding provider returned invalid vector values"
                    )
                vectors.append([float(value) for value in vector])
            if len(vectors) != len(inputs):
                raise PolicyEmbeddingProviderError(
                    "Embedding provider returned an unexpected vector count"
                )
            return vectors
        except PolicyEmbeddingProviderError:
            raise
        except Exception:
            raise PolicyEmbeddingProviderError(
                "Embedding provider request failed"
            ) from None

    return embed_texts


def create_cpu_cross_encoder_reranker(
    model_name: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> CrossEncoderReranker:
    """Load a sentence-transformers CrossEncoder on CPU, without downloading silently."""
    values = os.environ if environment is None else environment
    resolved_model = (
        model_name
        or values.get("CAREFLOW_CROSS_ENCODER_MODEL", "").strip()
        or DEFAULT_CROSS_ENCODER_MODEL
    )
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise PolicyProviderConfigurationError(
            "Install requirements-m4-evaluation.txt to use the CPU cross-encoder"
        ) from None

    try:
        model = CrossEncoder(resolved_model, device="cpu")
    except Exception:
        raise PolicyCrossEncoderProviderError(
            "The CPU cross-encoder could not be loaded"
        ) from None

    def score_pairs(pairs: Sequence[tuple[str, str]]) -> Sequence[Real]:
        try:
            scores = model.predict(
                list(pairs),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            if isinstance(scores, Real) and not isinstance(scores, bool):
                return [scores]
            if not isinstance(scores, Sequence) or isinstance(
                scores, (str, bytes)
            ):
                raise PolicyCrossEncoderProviderError(
                    "The CPU cross-encoder returned invalid scores"
                )
            return scores
        except PolicyCrossEncoderProviderError:
            raise
        except Exception:
            raise PolicyCrossEncoderProviderError(
                "The CPU cross-encoder scoring call failed"
            ) from None

    return CrossEncoderReranker(score_pairs)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
