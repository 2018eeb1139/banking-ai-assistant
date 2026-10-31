import pytest

from app.tools import account
from app.tools.account import CustomerNotFound, get_account_state
from app.tools.service_request import (
    InvalidCategory,
    raise_service_request,
)


def test_account_lookup_returns_requested_customer(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(account.time, "sleep", sleep_calls.append)

    result = get_account_state("CUST-2000")

    assert result["customer_id"] == "CUST-2000"
    assert sleep_calls == [0.4]


def test_account_lookup_raises_for_unknown_customer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(account.time, "sleep", lambda _: None)

    with pytest.raises(CustomerNotFound):
        get_account_state("CUST-UNKNOWN")


@pytest.mark.parametrize(
    ("category", "expected_queue", "expected_sla"),
    [
        ("Card block or replacement", "CARDS_URGENT", 4),
        ("Transaction dispute", "DISPUTES", 16),
        ("Statement or certificate request", "SELFSERVE", 8),
        ("Limit change", "CREDIT_OPS", 24),
        ("Standing instruction change", "PAYMENTS", 16),
        ("Complaint", "COMPLAINTS", 40),
        ("GENERAL", "GENERAL", 8),
    ],
)
def test_service_request_routes_supported_categories(
    category: str,
    expected_queue: str,
    expected_sla: int,
) -> None:
    result = raise_service_request(
        customer_id="CUST-2000",
        category=category,
        summary="Customer requests assistance.",
    )

    assert result["request_id"].startswith("SR-")
    assert result["queue"] == expected_queue
    assert result["sla_hours"] == expected_sla


def test_service_request_ids_are_unique() -> None:
    first = raise_service_request("CUST-2000", "Transaction dispute", "First")
    second = raise_service_request("CUST-2000", "Transaction dispute", "Second")

    assert first["request_id"] != second["request_id"]


def test_service_request_rejects_unknown_category() -> None:
    with pytest.raises(InvalidCategory):
        raise_service_request("CUST-2000", "Urgent help", "Customer requests help.")


def test_service_request_rejects_blank_summary() -> None:
    with pytest.raises(ValueError):
        raise_service_request("CUST-2000", "Transaction dispute", "  ")