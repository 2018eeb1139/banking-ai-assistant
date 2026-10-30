from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    customer_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("customer_id", "question")
    @classmethod
    def strip_and_reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value