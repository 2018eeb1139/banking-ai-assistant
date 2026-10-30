import pytest
from pydantic import ValidationError

from app.schemas.evidence import Citation, SourceType
from app.schemas.request import AgentRequest
from app.schemas.response import AgentResponse, Outcome, ServiceRequestResult



def test_request_generates_id_and_trims_fields() -> None:
    request = AgentRequest(customer_id=" CUST-2000 ", question=" Fees? ")

    assert request.request_id is not None
    assert request.customer_id == "CUST-2000"
    assert request.question == "Fees?"


def test_request_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        AgentRequest(customer_id="CUST-2000", question="  ")


def test_policy_citation_requires_version_and_section() -> None:
    with pytest.raises(ValidationError):
        Citation(
            source_type=SourceType.POLICY,
            source_id="POL-CRD-001",
            evidence="The policy requires six statement cycles.",
        )


def test_raise_request_requires_tool_result() -> None:
    with pytest.raises(ValidationError):
        AgentResponse(
            outcome=Outcome.RAISE_REQUEST,
            customer_id="CUST-2000",
            intent="dispute",
            message="Your request was raised.",
            claims=[],
            citations=[],
        )


def test_raise_request_accepts_tool_result() -> None:
    response = AgentResponse(
        outcome=Outcome.RAISE_REQUEST,
        customer_id="CUST-2000",
        intent="dispute",
        message="Your request was submitted.",
        claims=[],
        citations=[],
        service_request=ServiceRequestResult(
            request_id="SR-example",
            queue="DISPUTES",
            sla_hours=48,
        ),
    )

    assert response.service_request is not None
    assert response.service_request.queue == "DISPUTES"