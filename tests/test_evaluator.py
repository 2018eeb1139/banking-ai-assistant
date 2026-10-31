from pathlib import Path
from typing import Any

from evaluation.evaluator import (
    EvaluationCase,
    load_evaluation_cases,
    run_evaluation,
)
from app.schemas.response import AgentResponse, Outcome


ROOT = Path(__file__).resolve().parents[1]


def test_loads_supplied_seed_and_at_least_fifteen_additional_cases() -> None:
    cases = load_evaluation_cases(
        ROOT / "data" / "eval_seed.csv",
        ROOT / "evaluation" / "eval_additional.csv",
    )

    assert len(cases) == 27
    assert sum(case.case_id.startswith("EVX-") for case in cases) >= 15
    assert cases[0].case_id == "EV-001"


def test_evaluator_checks_outcome_intent_and_account_usage() -> None:
    case = EvaluationCase(
        case_id="EV-TEST",
        customer_id="CUST-2000",
        question="Unsupported question",
        expected_outcome="refuse",
        expected_intent="unsupported",
        expected_account_access=False,
    )
    response = AgentResponse(
        outcome=Outcome.REFUSE,
        customer_id="CUST-2000",
        intent="unsupported",
        message="I cannot answer that from the available information.",
        claims=[],
        citations=[],
    )

    report = run_evaluation(
        [case],
        lambda _: {
            "final_response": response,
            "intent": "unsupported",
            "account_lookup_attempted": False,
            "retrieved_policy": [],
        },
    )

    assert report["passed_cases"] == 1
    checks = report["results"][0]["checks"]
    assert checks["outcome"] is True
    assert checks["intent"] is True
    assert checks["account_usage"] is True
    assert checks["citations"] is True


def test_evaluator_checks_error_handling_and_request_queue() -> None:
    case = EvaluationCase(
        case_id="EV-ERROR",
        customer_id="CUST-UNKNOWN",
        question="What is my card product?",
        expected_outcome="needs_review",
        expected_intent="account_query",
        expected_account_access=True,
        expected_error_code="customer_not_found",
    )
    response = AgentResponse(
        outcome=Outcome.NEEDS_REVIEW,
        customer_id="CUST-UNKNOWN",
        intent="account_query",
        message="This request needs review.",
        claims=[],
        citations=[],
    )

    report = run_evaluation(
        [case],
        lambda _: {
            "final_response": response,
            "intent": "account_query",
            "account_lookup_attempted": True,
            "error_code": "customer_not_found",
            "retrieved_policy": [],
        },
    )

    checks = report["results"][0]["checks"]
    assert checks["error_handling"] is True
    assert checks["account_usage"] is True


def test_runner_records_unexpected_case_error_and_continues() -> None:
    cases = [
        EvaluationCase(
            case_id="EV-ERROR-1",
            customer_id="CUST-2000",
            question="First",
            expected_outcome="answer",
            expected_intent="general",
        ),
        EvaluationCase(
            case_id="EV-ERROR-2",
            customer_id="CUST-2000",
            question="Second",
            expected_outcome="answer",
            expected_intent="general",
        ),
    ]

    def execute(case: EvaluationCase) -> dict[str, Any]:
        if case.case_id == "EV-ERROR-1":
            raise RuntimeError("private diagnostic detail")
        response = AgentResponse(
            outcome=Outcome.ANSWER,
            customer_id=case.customer_id,
            intent="general",
            message="A grounded response.",
            claims=[],
            citations=[],
        )
        return {
            "final_response": response,
            "intent": "general",
            "retrieved_policy": [],
        }

    report = run_evaluation(cases, execute)

    assert report["case_count"] == 2
    assert report["results"][0]["error_code"] == "runtime_exception:RuntimeError"
    assert "private diagnostic detail" not in str(report)
