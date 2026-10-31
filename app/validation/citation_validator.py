import json

from app.retrieval.hybrid_search import SearchResult
from app.schemas.evidence import Citation, SourceType
from app.schemas.response import AgentResponse


class CitationValidationError(ValueError):
    """Raised when a response citation does not match supplied evidence."""


ALLOWED_ACCOUNT_CITATION_FIELDS = {
    "card_product",
    "current_limit",
    "statement_balance",
    "minimum_due",
    "due_date",
    "card_year_spend",
    "open_dispute",
    "kyc_last_done",
    "last_limit_increase_date",
    "months_repayment_history",
    "payment_over_30dpd_last_12_months",
    "debt_burden_ratio_after_increase_percent",
}


def format_account_evidence(value: object) -> str:
    """Return the exact account value format expected in a citation."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def validate_citation(
    citation: Citation,
    retrieved_policy: list[SearchResult],
    customer_id: str,
    account_state: dict[str, object] | None,
    service_request: dict[str, str | int] | None = None,
) -> None:
    if citation.source_type == SourceType.POLICY:
        for result in retrieved_policy:
            chunk = result.chunk
            if (
                chunk.is_authoritative
                and chunk.document_id == citation.source_id
                and chunk.version == citation.document_version
                and chunk.section == citation.section
                and _normalize_whitespace(citation.evidence)
                in _normalize_whitespace(chunk.text)
            ):
                return

        raise CitationValidationError(
            "Policy citation does not match retrieved authoritative evidence"
        )

    if citation.source_type == SourceType.ACCOUNT:
        if account_state is None:
            raise CitationValidationError(
                "Account citation supplied without retrieved account state"
            )
        if citation.source_id != customer_id:
            raise CitationValidationError(
                "Account citation customer does not match the request"
            )

        field = citation.account_field
        if field not in ALLOWED_ACCOUNT_CITATION_FIELDS:
            raise CitationValidationError(
                "Account citation refers to a disallowed field"
            )
        if field not in account_state:
            raise CitationValidationError(
                "Account citation field was not returned by the account tool"
            )

        expected_evidence = format_account_evidence(account_state[field])
        if citation.evidence != expected_evidence:
            raise CitationValidationError(
                "Account citation evidence does not match the account tool result"
            )
        return

    if citation.source_type == SourceType.SERVICE_REQUEST:
        if service_request is None:
            raise CitationValidationError(
                "Service-request citation supplied without a tool result"
            )

        expected_id = service_request.get("request_id")
        if citation.source_id != expected_id:
            raise CitationValidationError(
                "Service-request citation ID does not match the tool result"
            )

        expected_evidence = json.dumps(
            service_request,
            ensure_ascii=False,
            sort_keys=True,
        )
        if citation.evidence != expected_evidence:
            raise CitationValidationError(
                "Service-request citation does not match the tool result"
            )
        return

    
    raise CitationValidationError("Unsupported citation source type")


def validate_agent_response_citations(
    response: AgentResponse,
    retrieved_policy: list[SearchResult],
    account_state: dict[str, object] | None,
) -> None:
    for claim in response.claims:
        for citation in claim.citations:
            if citation not in response.citations:
                raise CitationValidationError(
                    "Claim citation is missing from response citations"
                )

    for citation in response.citations:
        validate_citation(
            citation=citation,
            retrieved_policy=retrieved_policy,
            customer_id=response.customer_id,
            account_state=account_state,
            service_request=(
                response.service_request.model_dump()
                if response.service_request is not None
                else None
            ),
        )