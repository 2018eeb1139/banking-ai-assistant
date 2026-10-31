import pytest
from pydantic import ValidationError

from app.retrieval.hybrid_search import SearchResult
from app.retrieval.ingestion import PolicyChunk
from app.schemas.evidence import Citation, Claim, SourceType
from app.schemas.response import AgentResponse, Outcome
from app.validation.citation_validator import (
    CitationValidationError,
    format_account_evidence,
    validate_agent_response_citations,
    validate_citation,
)


def policy_result() -> SearchResult:
    chunk = PolicyChunk(
        chunk_id="POL-FEE-007:1",
        document_id="POL-FEE-007",
        title="Fees and Charges Schedule",
        version="5.1",
        effective_date="01-Apr-2026",
        owner="Product",
        section="Fees and Charges Schedule",
        text="Foreign currency markup is 3.5%.",
        is_authoritative=True,
    )
    return SearchResult(chunk=chunk, score=0.1)


def test_accepts_retrieved_policy_quote() -> None:
    citation = Citation(
        source_type=SourceType.POLICY,
        source_id="POL-FEE-007",
        document_version="5.1",
        section="Fees and Charges Schedule",
        evidence="Foreign currency markup is 3.5%.",
    )

    validate_citation(citation, [policy_result()], "CUST-2000", None)


def test_rejects_policy_quote_not_in_retrieved_chunk() -> None:
    citation = Citation(
        source_type=SourceType.POLICY,
        source_id="POL-FEE-007",
        document_version="5.1",
        section="Fees and Charges Schedule",
        evidence="Foreign currency markup is 4%.",
    )

    with pytest.raises(CitationValidationError):
        validate_citation(citation, [policy_result()], "CUST-2000", None)


def test_accepts_exact_account_field_evidence() -> None:
    account = {"card_year_spend": 162000}
    citation = Citation(
        source_type=SourceType.ACCOUNT,
        source_id="CUST-2000",
        account_field="card_year_spend",
        evidence=format_account_evidence(account["card_year_spend"]),
    )

    validate_citation(citation, [], "CUST-2000", account)


def test_rejects_account_citation_for_another_customer() -> None:
    citation = Citation(
        source_type=SourceType.ACCOUNT,
        source_id="CUST-2001",
        account_field="card_year_spend",
        evidence="162000",
    )

    with pytest.raises(CitationValidationError):
        validate_citation(
            citation,
            [],
            "CUST-2000",
            {"card_year_spend": 162000},
        )


def test_rejects_internal_score_citation() -> None:
    citation = Citation(
        source_type=SourceType.ACCOUNT,
        source_id="CUST-2000",
        account_field="behavioural_score",
        evidence="640",
    )

    with pytest.raises(CitationValidationError):
        validate_citation(
            citation,
            [],
            "CUST-2000",
            {"behavioural_score": 640},
        )


def test_response_schema_rejects_claim_citation_missing_from_response_list() -> None:
    citation = Citation(
        source_type=SourceType.POLICY,
        source_id="POL-FEE-007",
        document_version="5.1",
        section="Fees and Charges Schedule",
        evidence="Foreign currency markup is 3.5%.",
    )

    with pytest.raises(ValidationError):
        AgentResponse(
            outcome=Outcome.ANSWER,
            customer_id="CUST-2000",
            intent="foreign_currency_markup",
            message="The markup is 3.5%.",
            claims=[
                Claim(
                    text="The markup is 3.5%.",
                    citations=[citation],
                )
            ],
            citations=[],
        )