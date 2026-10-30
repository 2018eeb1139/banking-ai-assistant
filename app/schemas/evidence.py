from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceType(str, Enum):
    POLICY = "policy"
    ACCOUNT = "account"


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    source_id: str = Field(min_length=1)
    document_version: str | None = None
    section: str | None = None
    account_field: str | None = None
    evidence: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_details(self) -> "Citation":
        if self.source_type == SourceType.POLICY:
            if not self.document_version or not self.section:
                raise ValueError(
                    "policy citations require a document version and section"
                )
        elif not self.account_field:
            raise ValueError("account citations require an account field")

        return self


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)