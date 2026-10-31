from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceType(str, Enum):
    POLICY = "policy"
    ACCOUNT = "account"
    SERVICE_REQUEST = "service_request"


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
            if self.account_field is not None:
                raise ValueError("policy citations cannot include an account field")

        elif self.source_type == SourceType.ACCOUNT:
            if not self.account_field:
                raise ValueError("account citations require an account field")
            if self.document_version is not None or self.section is not None:
                raise ValueError(
                    "account citations cannot include policy document details"
                )

        elif (
            self.document_version is not None
            or self.section is not None
            or self.account_field is not None
        ):
            raise ValueError(
                "service-request citations cannot include policy or account details"
            )

        return self


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)