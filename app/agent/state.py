from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from app.policy.rules import LimitDecision
from app.schemas.response import AgentResponse
from app.retrieval.hybrid_search import SearchResult


class Intent(str, Enum):
    ANNUAL_FEE = "annual_fee"
    PRODUCT_RATE = "product_rate"
    FOREIGN_CURRENCY_MARKUP = "foreign_currency_markup"
    LIMIT_INCREASE = "limit_increase"
    TRANSACTION_DISPUTE = "transaction_dispute"
    CARD_BLOCK = "card_block"
    STATEMENT_REQUEST = "statement_request"
    STANDING_INSTRUCTION = "standing_instruction"
    COMPLAINT = "complaint"
    ACCOUNT_QUERY = "account_query"
    UNSUPPORTED = "unsupported"
    GENERAL = "general"


class IntentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    proposed_limit: int | None = Field(default=None, gt=0)


class AgentState(TypedDict, total=False):
    request_id: str
    customer_id: str
    question: str

    intent: str
    proposed_limit: int | None

    retrieved_policy: list[SearchResult]
    evidence_sufficient: bool
    missing_information: list[str]

    account_lookup_attempted: bool
    account_state: dict[str, Any]
    policy_decision: LimitDecision | None

    service_request: dict[str, str | int]
    workflow_status: str
    final_response: AgentResponse
    error_code: str | None