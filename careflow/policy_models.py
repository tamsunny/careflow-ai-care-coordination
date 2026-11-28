"""Typed synthetic policy documents and retrieval results."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PolicySection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=50_000)


class PolicyDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_id: str = Field(
        min_length=16,
        max_length=80,
        pattern=r"^SYN-POLICY-[A-Z0-9-]+$",
    )
    title: str = Field(min_length=1, max_length=200)
    sections: tuple[PolicySection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_section_names(self) -> "PolicyDocument":
        names = [section.name.casefold() for section in self.sections]
        if len(names) != len(set(names)):
            raise ValueError("Section names must be unique within a policy document")
        return self


class PolicyChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    title: str
    section_name: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)


class PolicySearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    source_id: str
    title: str
    section_name: str
    text: str
    score: float


class PolicyIndexResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    documents_indexed: int = Field(ge=1)
    chunks_indexed: int = Field(ge=1)
    stale_chunks_removed: int = Field(ge=0)
