from typing import Any

from app.agent.response import generate_agent_response
from app.agent.state import Intent
from app.schemas.response import Outcome
from app.retrieval.hybrid_search import SearchResult
from app.retrieval.ingestion import PolicyChunk


class FakeStructuredModel:
    def __init__(
        self,
        schema: type,
        outputs: list[object],
        include_raw: bool,
    ) -> None:
        self.schema = schema
        self.outputs = outputs
        self.include_raw = include_raw
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> object:
        self.calls.append(prompt)
        if not self.outputs:
            raise AssertionError("Unexpected additional structured model call")
        output = self.outputs.pop(0)
        if self.include_raw:
            return {"parsed": output, "raw": None, "parsing_error": None}
        return output


class FakeChatModel:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = outputs
        self.structured_models: list[FakeStructuredModel] = []

    def with_structured_output(
        self,
        schema: type,
        include_raw: bool = False,
    ) -> FakeStructuredModel:
        model = FakeStructuredModel(schema, self.outputs, include_raw)
        self.structured_models.append(model)
        return model


def fee_policy_result() -> SearchResult:
    chunk = PolicyChunk(
        chunk_id="POL-FEE-007:1",
        document_id="POL-FEE-007",
        title="Fees and Charges Schedule",
        version="5.1",
        effective_date="01-Apr-2026",
        owner="Product",
        section="Fees and Charges Schedule",
        text=(
            "| Foreign currency markup | 3.5% |\n\n"
            "The foreign currency markup is 3.5%."
        ),
        is_authoritative=True,
    )
    return SearchResult(chunk=chunk, score=0.1)


def test_answer_uses_application_created_retrieved_citation() -> None:
    model = FakeChatModel(
        [
            {
                "claims": [
                    {
                        "text": "The foreign currency markup is 3.5%.",
                        "evidence_ids": ["policy:POL-FEE-007:1"],
                    }
                ]
            }
        ]
    )
    state: dict[str, Any] = {
        "customer_id": "CUST-2000",
        "question": "What is the foreign currency markup?",
        "intent": Intent.FOREIGN_CURRENCY_MARKUP.value,
        "workflow_status": "ready",
        "retrieved_policy": [fee_policy_result()],
    }

    response = generate_agent_response(state, model)  # type: ignore[arg-type]

    assert response.outcome == Outcome.ANSWER
    assert len(response.claims) == 1
    assert response.citations[0].source_id == "POL-FEE-007"
    assert response.citations[0].document_version == "5.1"


def test_invalid_evidence_id_is_retried_then_returns_needs_review() -> None:
    model = FakeChatModel(
        [
            {
                "claims": [
                    {
                        "text": "The markup is 4%.",
                        "evidence_ids": ["policy:invented"],
                    }
                ]
            },
            {
                "claims": [
                    {
                        "text": "The markup is 4%.",
                        "evidence_ids": ["policy:invented"],
                    }
                ]
            },
        ]
    )
    state: dict[str, Any] = {
        "customer_id": "CUST-2000",
        "question": "What is the foreign currency markup?",
        "intent": Intent.FOREIGN_CURRENCY_MARKUP.value,
        "workflow_status": "ready",
        "retrieved_policy": [fee_policy_result()],
    }

    response = generate_agent_response(state, model)  # type: ignore[arg-type]

    assert response.outcome == Outcome.NEEDS_REVIEW
    assert len(model.structured_models[0].calls) == 2
    assert "previous draft was invalid" in model.structured_models[0].calls[1]


def test_successful_tool_result_is_reported_with_actual_id() -> None:
    state: dict[str, Any] = {
        "customer_id": "CUST-2000",
        "question": "Someone used my card.",
        "intent": Intent.TRANSACTION_DISPUTE.value,
        "workflow_status": "ready",
        "service_request": {
            "request_id": "SR-123",
            "queue": "DISPUTES",
            "sla_hours": 16,
        },
    }

    response = generate_agent_response(state, FakeChatModel([]))  # type: ignore[arg-type]

    assert response.outcome == Outcome.RAISE_REQUEST
    assert response.service_request is not None
    assert response.service_request.request_id == "SR-123"
    assert response.citations[0].source_id == "SR-123"


def test_limit_service_request_says_referred_not_approved() -> None:
    state: dict[str, Any] = {
        "customer_id": "CUST-2000",
        "question": "Can you increase my limit?",
        "intent": Intent.LIMIT_INCREASE.value,
        "workflow_status": "needs_review",
        "missing_information": ["assessed monthly net income"],
        "service_request": {
            "request_id": "SR-456",
            "queue": "CREDIT_OPS",
            "sla_hours": 24,
        },
    }

    response = generate_agent_response(state, FakeChatModel([]))  # type: ignore[arg-type]

    assert response.outcome == Outcome.RAISE_REQUEST
    assert "referred for review" in response.message
    assert "approved" not in response.message.lower()


class EmptyRetriever:
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return []


def test_graph_returns_response_when_classification_is_invalid(
) -> None:
    from app.agent.graph import run_agent_workflow
    from app.schemas.request import AgentRequest

    model = FakeChatModel([{}])
    state = AgentRequest(
        customer_id="CUST-2000",
        question="Please do something.",
    )

    result = run_agent_workflow(
        state,
        EmptyRetriever(),  # type: ignore[arg-type]
        model,  # type: ignore[arg-type]
    )

    assert result["final_response"].outcome == Outcome.NEEDS_REVIEW
    assert result["error_code"] == "invalid_intent_classification"
