from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.evidence import Citation, Claim


class Outcome(str, Enum):
    ANSWER = "answer"
    DECLINE_WITH_REASON = "decline_with_reason"
    REFUSE = "refuse"
    RAISE_REQUEST = "raise_request"
    NEEDS_REVIEW = "needs_review"


class ServiceRequestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    queue: str = Field(min_length=1)
    sla_hours: int = Field(gt=0)


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Outcome
    customer_id: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    message: str = Field(min_length=1)
    claims: list[Claim]
    citations: list[Citation]
    reason_code: str | None = None
    service_request: ServiceRequestResult | None = None
    missing_information: list[str] = Field(default_factory=list)
    next_action: str | None = None

    @model_validator(mode="after")
    def validate_response_consistency(self) -> "AgentResponse":
        if self.outcome == Outcome.RAISE_REQUEST and self.service_request is None:
            raise ValueError(
                "raise_request outcome requires a successful service request result"
            )

        if self.outcome != Outcome.RAISE_REQUEST and self.service_request is not None:
            raise ValueError(
                "service request result is only valid for raise_request outcome"
            )

        if (
            self.outcome == Outcome.DECLINE_WITH_REASON
            and not self.reason_code
        ):
            raise ValueError("decline_with_reason outcome requires a reason code")

        for claim in self.claims:
            for citation in claim.citations:
                if citation not in self.citations:
                    raise ValueError(
                        "each claim citation must also appear in response citations"
                    )

        return self