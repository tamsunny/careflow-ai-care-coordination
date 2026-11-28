"""Typed synthetic policy documents and retrieval results."""

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PolicyPageText(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    page_number: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=50_000)


class PolicySection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=50_000)
    pages: tuple[PolicyPageText, ...] = ()

    @model_validator(mode="after")
    def require_unique_page_numbers(self) -> "PolicySection":
        page_numbers = [page.page_number for page in self.pages]
        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError("Page numbers must be unique within a policy section")
        return self


class PolicyDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_id: str = Field(
        min_length=16,
        max_length=80,
        pattern=r"^SYN-POLICY-[A-Z0-9-]+$",
    )
    title: str = Field(min_length=1, max_length=200)
    sections: tuple[PolicySection, ...] = Field(min_length=1)
    category: str = Field(default="general", min_length=1, max_length=80)
    specialty: str | None = Field(default=None, min_length=1, max_length=80)
    plan_ids: tuple[str, ...] = ()
    version: str = Field(default="1.0", min_length=1, max_length=40)
    effective_date: date | None = None
    synthetic_data: Literal[True] = True

    @model_validator(mode="after")
    def require_unique_section_names(self) -> "PolicyDocument":
        names = [section.name.casefold() for section in self.sections]
        if len(names) != len(set(names)):
            raise ValueError("Section names must be unique within a policy document")
        if len(self.plan_ids) != len(set(self.plan_ids)):
            raise ValueError("Synthetic plan IDs must be unique")
        if any(
            not re.fullmatch(r"SYN-PLAN-[A-Z0-9-]+", plan_id)
            for plan_id in self.plan_ids
        ):
            raise ValueError("Plan IDs must use the synthetic SYN-PLAN prefix")
        return self


class PolicyChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    title: str
    section_name: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    category: str = "general"
    specialty: str | None = None
    plan_ids: tuple[str, ...] = ()
    version: str = "1.0"
    effective_date: date | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    synthetic_data: Literal[True] = True

    @model_validator(mode="after")
    def validate_page_range(self) -> "PolicyChunk":
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must not precede page_start")
        return self


class PolicySearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    source_id: str
    title: str
    section_name: str
    text: str
    score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None
    category: str | None = "general"
    specialty: str | None = None
    plan_ids: tuple[str, ...] = ()
    version: str | None = "1.0"
    effective_date: date | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    synthetic_data: Literal[True] = True

    @model_validator(mode="before")
    @classmethod
    def retain_legacy_dense_score(cls, value):
        if (
            isinstance(value, dict)
            and "dense_score" not in value
            and value.get("score") is not None
        ):
            return {**value, "dense_score": value["score"]}
        return value


class PolicySearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_id: str | None = None
    section_name: str | None = None
    category: str | None = None
    specialty: str | None = None
    plan_id: str | None = None
    version: str | None = None

    @model_validator(mode="after")
    def validate_filter_values(self) -> "PolicySearchFilters":
        if self.plan_id is not None and not re.fullmatch(
            r"SYN-PLAN-[A-Z0-9-]+", self.plan_id
        ):
            raise ValueError("plan_id must use the synthetic SYN-PLAN prefix")
        for field_name in (
            "source_id",
            "section_name",
            "category",
            "specialty",
            "version",
        ):
            value = getattr(self, field_name)
            if value is not None and not value:
                raise ValueError(f"{field_name} must not be empty")
        return self


class PolicyIndexResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    documents_indexed: int = Field(ge=1)
    chunks_indexed: int = Field(ge=1)
    stale_chunks_removed: int = Field(ge=0)
