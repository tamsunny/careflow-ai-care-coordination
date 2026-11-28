"""Load and validate the synthetic policy PDF corpus."""

import argparse
import json
import re
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pypdf import PdfReader

from .policy_models import PolicyDocument, PolicyPageText, PolicySection

PARSER_VERSION = "pypdf-section-v2"
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "policies" / "manifest.json"
)


class PolicyIngestionError(RuntimeError):
    """A policy manifest or source PDF could not be safely ingested."""


class PolicyManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_id: str = Field(
        min_length=16,
        max_length=80,
        pattern=r"^SYN-POLICY-[A-Z0-9-]+$",
    )
    file: str = Field(min_length=5, max_length=120, pattern=r"^[A-Za-z0-9_.-]+\.pdf$")
    title: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=80)
    specialty: str | None = Field(default=None, min_length=1, max_length=80)
    plan_ids: tuple[str, ...] = ()
    version: str = Field(min_length=1, max_length=40)
    effective_date: str = Field(min_length=10, max_length=10)
    synthetic_data: Literal[True]

    @model_validator(mode="after")
    def validate_effective_date(self) -> "PolicyManifestEntry":
        try:
            date.fromisoformat(self.effective_date)
        except ValueError as exc:
            raise ValueError("effective_date must be an ISO calendar date") from exc
        if Path(self.file).name != self.file:
            raise ValueError("Policy PDF file must be a basename")
        if len(self.plan_ids) != len(set(self.plan_ids)):
            raise ValueError("Synthetic plan IDs must be unique")
        if any(
            not re.fullmatch(r"SYN-PLAN-[A-Z0-9-]+", plan_id)
            for plan_id in self.plan_ids
        ):
            raise ValueError("Plan IDs must use the synthetic SYN-PLAN prefix")
        return self


class PolicyManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[1]
    synthetic_data: Literal[True]
    declaration: str = Field(min_length=20, max_length=500)
    documents: tuple[PolicyManifestEntry, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def require_unique_sources_and_files(self) -> "PolicyManifest":
        source_ids = [document.source_id for document in self.documents]
        filenames = [document.file.casefold() for document in self.documents]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Policy source IDs must be unique")
        if len(filenames) != len(set(filenames)):
            raise ValueError("Policy PDF filenames must be unique")
        return self


def load_policy_documents(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> tuple[PolicyDocument, ...]:
    """Validate the manifest and extract page-aware documents from its PDFs."""
    manifest_file = Path(manifest_path).resolve()
    try:
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyIngestionError(
            "The synthetic policy manifest could not be read"
        ) from exc

    try:
        manifest = PolicyManifest.model_validate(manifest_data)
    except ValidationError as exc:
        raise PolicyIngestionError(
            "The policy manifest is invalid or does not declare synthetic data"
        ) from exc

    policy_directory = manifest_file.parent.resolve()
    return tuple(
        _load_pdf_document(entry, policy_directory)
        for entry in manifest.documents
    )


def _load_pdf_document(
    entry: PolicyManifestEntry,
    policy_directory: Path,
) -> PolicyDocument:
    pdf_path = (policy_directory / entry.file).resolve()
    if pdf_path.parent != policy_directory:
        raise PolicyIngestionError("Policy PDF paths must remain inside the corpus directory")

    try:
        reader = PdfReader(str(pdf_path), strict=True)
        if reader.is_encrypted:
            raise PolicyIngestionError(
                f"Synthetic policy PDF {entry.file!r} must not be encrypted"
            )
        if not reader.pages:
            raise PolicyIngestionError(
                f"Synthetic policy PDF {entry.file!r} contains no pages"
            )
        sections: OrderedDict[str, list[PolicyPageText]] = OrderedDict()
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if not isinstance(page_text, str) or not page_text.strip():
                continue
            _validate_embedded_metadata(page_text, entry, page_number)
            for section_name, section_text in _extract_page_sections(page_text):
                sections.setdefault(section_name, []).append(
                    PolicyPageText(
                        page_number=page_number,
                        text=section_text,
                    )
                )
    except PolicyIngestionError:
        raise
    except Exception as exc:
        raise PolicyIngestionError(
            f"Synthetic policy PDF {entry.file!r} is malformed or unreadable"
        ) from exc

    if not sections:
        raise PolicyIngestionError(
            f"Synthetic policy PDF {entry.file!r} contains no extractable sections"
        )

    return PolicyDocument(
        source_id=entry.source_id,
        title=entry.title,
        category=entry.category,
        specialty=entry.specialty,
        plan_ids=entry.plan_ids,
        version=entry.version,
        effective_date=entry.effective_date,
        synthetic_data=True,
        sections=tuple(
            PolicySection(
                name=name,
                text=" ".join(page.text for page in pages),
                pages=tuple(pages),
            )
            for name, pages in sections.items()
        ),
    )


def _validate_embedded_metadata(
    page_text: str,
    entry: PolicyManifestEntry,
    page_number: int,
) -> None:
    if "FICTIONAL SYNTHETIC POLICY" not in page_text:
        raise PolicyIngestionError(
            f"Policy PDF {entry.file!r}, page {page_number}, lacks its synthetic-data notice"
        )
    expected_headers = {
        "SOURCE ID": entry.source_id,
        "TITLE": entry.title,
        "CATEGORY": entry.category,
        "SPECIALTY": entry.specialty or "NONE",
        "PLAN IDS": ",".join(entry.plan_ids) if entry.plan_ids else "NONE",
        "VERSION": entry.version,
        "EFFECTIVE DATE": entry.effective_date,
    }
    for label, expected in expected_headers.items():
        if f"{label}: {expected}" not in page_text:
            raise PolicyIngestionError(
                f"Policy PDF {entry.file!r}, page {page_number}, does not match "
                f"manifest field {label!r}"
            )


def _extract_page_sections(page_text: str) -> tuple[tuple[str, str], ...]:
    sections: list[tuple[str, str]] = []
    current_name: str | None = None
    current_lines: list[str] = []
    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        if line.startswith("SECTION:"):
            if current_name is not None:
                _append_section(sections, current_name, current_lines)
            current_name = line.removeprefix("SECTION:").strip()
            current_lines = []
        elif (
            line
            and not line.startswith("FICTIONAL SYNTHETIC POLICY")
            and not line.startswith(
                (
                    "SOURCE ID:",
                    "TITLE:",
                    "CATEGORY:",
                    "SPECIALTY:",
                    "PLAN IDS:",
                    "VERSION:",
                    "EFFECTIVE DATE:",
                )
            )
        ):
            if current_name is None:
                raise PolicyIngestionError(
                    "Extracted policy text appears before a SECTION heading"
                )
            current_lines.append(line)
    if current_name is not None:
        _append_section(sections, current_name, current_lines)
    return tuple(sections)


def _append_section(
    sections: list[tuple[str, str]],
    section_name: str,
    lines: list[str],
) -> None:
    text = " ".join(lines).strip()
    if not section_name or not text:
        raise PolicyIngestionError("Policy sections require a name and extractable text")
    sections.append((section_name, text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and extract the fictional CareFlow policy PDF corpus."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the synthetic policy manifest JSON",
    )
    args = parser.parse_args(argv)
    documents = load_policy_documents(args.manifest)
    page_count = sum(
        len(section.pages)
        for document in documents
        for section in document.sections
    )
    print(
        f"Validated {len(documents)} synthetic policy PDFs, "
        f"{sum(len(document.sections) for document in documents)} sections, "
        f"and {page_count} page sections."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
