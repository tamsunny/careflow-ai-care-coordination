import json
import shutil

import pytest
from pypdf import PdfWriter

from careflow.policy_ingestion import (
    DEFAULT_MANIFEST_PATH,
    PARSER_VERSION,
    PolicyIngestionError,
    load_policy_documents,
)


def _write_manifest(directory, *, file_name, title="Fictional Test Policy"):
    manifest = {
        "schema_version": 1,
        "synthetic_data": True,
        "declaration": (
            "This is fictional synthetic policy content for parser tests only."
        ),
        "documents": [
            {
                "source_id": "SYN-POLICY-TEST-901",
                "file": file_name,
                "title": title,
                "category": "test_policy",
                "version": "1.0",
                "effective_date": "2026-01-15",
                "synthetic_data": True,
            }
        ],
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_default_manifest_loads_fourteen_synthetic_page_aware_documents():
    documents = load_policy_documents()

    assert len(documents) == 14
    assert all(document.synthetic_data is True for document in documents)
    assert {document.category for document in documents} >= {
        "referral",
        "pre_authorization",
        "copay_schedule",
        "clinical_escalation",
        "intake_sop",
        "specialty_handbook",
    }
    assert all(document.version == "1.0" for document in documents)
    assert all(document.effective_date is not None for document in documents)
    assert all(
        section.pages
        and all(page.page_number >= 1 and page.text for page in section.pages)
        for document in documents
        for section in document.sections
    )
    assert PARSER_VERSION == "pypdf-section-v2"
    cardiology = next(
        document
        for document in documents
        if document.source_id == "SYN-POLICY-SPEC-102"
    )
    copay_schedule = next(
        document
        for document in documents
        if document.source_id == "SYN-POLICY-COPAY-102"
    )
    assert cardiology.specialty == "cardiology"
    assert cardiology.plan_ids == ()
    assert copay_schedule.specialty is None
    assert copay_schedule.plan_ids == ("SYN-PLAN-12", "SYN-PLAN-24")


def test_pdf_metadata_must_match_manifest(tmp_path):
    source_pdf = DEFAULT_MANIFEST_PATH.parent / "referral_intake_routing.pdf"
    shutil.copyfile(source_pdf, tmp_path / "policy.pdf")
    manifest_path = _write_manifest(
        tmp_path,
        file_name="policy.pdf",
        title="A Different Fictional Title",
    )

    with pytest.raises(PolicyIngestionError, match="does not match manifest"):
        load_policy_documents(manifest_path)


def test_malformed_pdf_is_reported_as_an_ingestion_error(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"not a PDF")
    manifest_path = _write_manifest(tmp_path, file_name="broken.pdf")

    with pytest.raises(PolicyIngestionError, match="malformed or unreadable"):
        load_policy_documents(manifest_path)


def test_empty_pdf_is_rejected_without_extractable_sections(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with (tmp_path / "empty.pdf").open("wb") as stream:
        writer.write(stream)
    manifest_path = _write_manifest(tmp_path, file_name="empty.pdf")

    with pytest.raises(PolicyIngestionError, match="no extractable sections"):
        load_policy_documents(manifest_path)


def test_manifest_rejects_non_synthetic_corpus(tmp_path):
    manifest_path = _write_manifest(tmp_path, file_name="missing.pdf")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["synthetic_data"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PolicyIngestionError, match="does not declare synthetic data"):
        load_policy_documents(manifest_path)


def test_manifest_rejects_path_traversal(tmp_path):
    manifest_path = _write_manifest(tmp_path, file_name="../outside.pdf")

    with pytest.raises(PolicyIngestionError, match="manifest is invalid"):
        load_policy_documents(manifest_path)
