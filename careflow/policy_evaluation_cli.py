"""Reproducible CLI for the versioned M4 synthetic policy retrieval evaluation."""

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

from .langfuse_evaluation import (
    LangfuseLoggingResult,
    create_langfuse_evaluation_logger_from_environment,
)
from .policy_evaluation import (
    RetrievalEvaluationReport,
    load_policy_evaluation_dataset,
    run_policy_retrieval_evaluation,
)
from .policy_index import LocalPolicyIndex
from .policy_ingestion import PARSER_VERSION, load_policy_documents
from .policy_providers import (
    create_cpu_cross_encoder_reranker,
    create_litellm_embedder,
    load_embedding_provider_config,
)
from .policy_retrieval import PolicyHybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_M4_INDEX_VERSION = "v1"
M4_PERSIST_DIRECTORY = (
    PROJECT_ROOT / "data" / f"qdrant_policy_index_m4_{DEFAULT_M4_INDEX_VERSION}"
)
M4_COLLECTION_NAME = f"synthetic_policy_handbook_m4_{DEFAULT_M4_INDEX_VERSION}"


def versioned_m4_index_target(index_version: str) -> tuple[Path, str]:
    if not isinstance(index_version, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}",
        index_version,
    ):
        raise ValueError("index_version must be a short alphanumeric M4 version")
    return (
        PROJECT_ROOT / "data" / f"qdrant_policy_index_m4_{index_version}",
        f"synthetic_policy_handbook_m4_{index_version}",
    )


def run_acceptance_evaluation(
    *,
    log_langfuse: bool = False,
    index_version: str = DEFAULT_M4_INDEX_VERSION,
) -> tuple[RetrievalEvaluationReport, LangfuseLoggingResult | None]:
    """Index the synthetic PDFs in the dedicated M4 collection and evaluate."""
    persist_directory, collection_name = versioned_m4_index_target(index_version)
    embedding_config = load_embedding_provider_config()
    embedder = create_litellm_embedder(embedding_config)
    reranker = create_cpu_cross_encoder_reranker()
    documents = load_policy_documents()
    dataset = load_policy_evaluation_dataset()

    with LocalPolicyIndex(
        embedder,
        embedding_model_id=embedding_config.model_id,
        embedding_dimension=embedding_config.dimension,
        persist_directory=persist_directory,
        collection_name=collection_name,
        parser_version=PARSER_VERSION,
    ) as index:
        index.index_documents(documents, skip_if_unchanged=True)
        retriever = PolicyHybridRetriever(index, reranker=reranker)
        report = run_policy_retrieval_evaluation(
            retriever,
            dataset,
            reranker=reranker,
        )

    logging_result = None
    if log_langfuse:
        logger = create_langfuse_evaluation_logger_from_environment()
        logging_result = logger.log_retrieval_report_nonfatal(report)
    return report, logging_result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run dense, BM25, hybrid RRF, and CPU-reranked retrieval against "
            "the synthetic M4 policy PDFs."
        )
    )
    parser.add_argument(
        "--log-langfuse",
        action="store_true",
        help="Create an aggregate-only Langfuse trace after evaluation",
    )
    parser.add_argument(
        "--index-version",
        default=DEFAULT_M4_INDEX_VERSION,
        help=(
            "Dedicated M4 collection version; use a new version for changed "
            "corpus or embedding settings"
        ),
    )
    args = parser.parse_args(argv)

    report, logging_result = run_acceptance_evaluation(
        log_langfuse=args.log_langfuse,
        index_version=args.index_version,
    )
    output = {"retrieval_report": report.model_dump(mode="json")}
    if logging_result is not None:
        output["langfuse"] = logging_result.model_dump(mode="json")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
