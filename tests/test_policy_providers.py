import sys
from types import ModuleType, SimpleNamespace

import pytest

from careflow.policy_models import PolicySearchResult
from careflow.policy_providers import (
    EmbeddingProviderConfig,
    PolicyCrossEncoderProviderError,
    PolicyEmbeddingProviderError,
    PolicyProviderConfigurationError,
    create_cpu_cross_encoder_reranker,
    create_litellm_embedder,
    load_embedding_provider_config,
)


def test_embedding_configuration_uses_environment_without_exposing_secrets():
    secret = "synthetic-test-api-key"
    config = load_embedding_provider_config(
        {
            "CAREFLOW_EMBEDDING_MODEL": "provider/embed-model",
            "CAREFLOW_EMBEDDING_DIMENSION": "3",
            "CAREFLOW_EMBEDDING_API_KEY": secret,
            "CAREFLOW_EMBEDDING_API_BASE": "https://provider.invalid/v1",
        }
    )

    assert config.model_id == "provider/embed-model"
    assert config.dimension == 3
    assert config.api_key == secret
    assert secret not in repr(config)


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"CAREFLOW_EMBEDDING_MODEL": "provider/model"},
        {
            "CAREFLOW_EMBEDDING_MODEL": "provider/model",
            "CAREFLOW_EMBEDDING_DIMENSION": "not-a-number",
        },
        {
            "CAREFLOW_EMBEDDING_MODEL": "provider/model",
            "CAREFLOW_EMBEDDING_DIMENSION": "0",
        },
    ],
)
def test_embedding_configuration_rejects_missing_or_invalid_settings(environment):
    with pytest.raises(PolicyProviderConfigurationError):
        load_embedding_provider_config(environment)


def test_litellm_embedder_passes_config_and_returns_vectors(monkeypatch):
    calls = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
                SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
            ]
        )

    import litellm

    monkeypatch.setattr(litellm, "embedding", fake_embedding)
    config = EmbeddingProviderConfig(
        model_id="provider/embed-model",
        dimension=3,
        api_key="synthetic-test-api-key",
        api_base="https://provider.invalid/v1",
    )

    vectors = create_litellm_embedder(config)(["first synthetic text", "second"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert calls[0]["model"] == config.model_id
    assert calls[0]["api_key"] == config.api_key
    assert calls[0]["api_base"] == config.api_base


def test_litellm_embedder_sanitizes_provider_failures(monkeypatch):
    secret_like_error = "synthetic-error-including-private-provider-data"

    def fake_embedding(**_kwargs):
        raise RuntimeError(secret_like_error)

    import litellm

    monkeypatch.setattr(litellm, "embedding", fake_embedding)
    embed = create_litellm_embedder(
        EmbeddingProviderConfig(model_id="provider/model", dimension=2)
    )

    with pytest.raises(PolicyEmbeddingProviderError) as error:
        embed(["synthetic input"])
    assert secret_like_error not in str(error.value)


def test_cpu_cross_encoder_is_injected_and_scores_without_network(monkeypatch):
    seen = {}

    class FakeCrossEncoder:
        def __init__(self, model_name, *, device):
            seen["model_name"] = model_name
            seen["device"] = device

        def predict(self, pairs, **kwargs):
            seen["pairs"] = pairs
            seen["predict_kwargs"] = kwargs
            return [0.1, 0.9]

    fake_module = ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    reranker = create_cpu_cross_encoder_reranker(
        environment={"CAREFLOW_CROSS_ENCODER_MODEL": "synthetic/test-cross-encoder"}
    )
    results = (
        PolicySearchResult(
            chunk_id="one",
            source_id="SYN-POLICY-TEST-101",
            title="Fictional policy",
            section_name="Section",
            text="synthetic first passage",
        ),
        PolicySearchResult(
            chunk_id="two",
            source_id="SYN-POLICY-TEST-102",
            title="Fictional policy",
            section_name="Section",
            text="synthetic second passage",
        ),
    )

    ranked = reranker.rerank("synthetic question", results, top_k=2)

    assert seen["model_name"] == "synthetic/test-cross-encoder"
    assert seen["device"] == "cpu"
    assert seen["predict_kwargs"] == {
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }
    assert ranked[0].chunk_id == "two"
    assert ranked[0].reranker_score == 0.9


def test_cpu_cross_encoder_load_and_score_errors_are_sanitized(monkeypatch):
    secret_like_error = "synthetic-model-path-private-error"

    class BrokenCrossEncoder:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(secret_like_error)

    fake_module = ModuleType("sentence_transformers")
    fake_module.CrossEncoder = BrokenCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    with pytest.raises(PolicyCrossEncoderProviderError) as error:
        create_cpu_cross_encoder_reranker("synthetic/broken-model")
    assert secret_like_error not in str(error.value)


def test_cpu_cross_encoder_dependency_error_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(PolicyProviderConfigurationError, match="requirements-m4"):
        create_cpu_cross_encoder_reranker("synthetic/test-model")
