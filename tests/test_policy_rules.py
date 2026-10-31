from datetime import date

from app.policy.rules import (
    LimitDecisionStatus,
    evaluate_limit_increase,
)


TODAY = date(2026, 10, 31)


def complete_account() -> dict[str, object]:
    return {
        "customer_id": "CUST-TEST",
        "current_limit": 100_000,
        "vulnerable_flag": False,
        "open_dispute": False,
        "last_limit_increase_date": None,
        "kyc_last_done": "2020-01-01",
        "kyc_risk_band": "low",
        "debt_burden_ratio_after_increase_percent": 50,
        "months_repayment_history": 12,
        "payment_over_30dpd_last_12_months": False,
        "avg_utilisation_3m": 0.50,
        "behavioural_score": 650,
        "assessed_monthly_net_income": 30_000,
        "bureau_trade_lines": 3,
    }


def test_vulnerable_customer_is_declined_without_invented_reason_code() -> None:
    account = complete_account()
    account["vulnerable_flag"] = True

    result = evaluate_limit_increase(account, 120_000, TODAY)

    assert result.status == LimitDecisionStatus.DECLINED
    assert result.reason_code is None
    assert result.compliance_review_required


def test_open_dispute_is_mandatory_decline() -> None:
    account = complete_account()
    account["open_dispute"] = True

    result = evaluate_limit_increase(account, 120_000, TODAY)

    assert result.status == LimitDecisionStatus.DECLINED
    assert result.reason_code == "AA04"


def test_recent_limit_increase_is_mandatory_decline() -> None:
    account = complete_account()
    account["last_limit_increase_date"] = "2026-07-01"

    result = evaluate_limit_increase(account, 120_000, TODAY)

    assert result.status == LimitDecisionStatus.DECLINED
    assert result.reason_code == "AA05"


def test_kyc_overdue_more_than_90_days_is_declined() -> None:
    account = complete_account()
    account["kyc_risk_band"] = "high"
    account["kyc_last_done"] = "2023-01-01"

    result = evaluate_limit_increase(account, 120_000, TODAY)

    assert result.status == LimitDecisionStatus.DECLINED
    assert result.reason_code == "AA06"


def test_debt_burden_over_55_percent_is_declined() -> None:
    account = complete_account()
    account["debt_burden_ratio_after_increase_percent"] = 56

    result = evaluate_limit_increase(account, 120_000, TODAY)

    assert result.status == LimitDecisionStatus.DECLINED
    assert result.reason_code == "AA02"


def test_three_month_seeded_thin_file_is_referred_and_capped() -> None:
    account = complete_account()
    account["months_repayment_history"] = 3

    result = evaluate_limit_increase(account, 120_000, TODAY)

    assert result.status == LimitDecisionStatus.REFER_TO_CREDIT_OFFICER
    assert result.max_permitted_limit == 125_000
    assert result.requires_credit_officer


def test_missing_income_causes_review_not_estimation() -> None:
    account = complete_account()
    del account["assessed_monthly_net_income"]

    result = evaluate_limit_increase(account, 120_000, TODAY)

    assert result.status == LimitDecisionStatus.NEEDS_REVIEW
    assert "assessed monthly net income" in result.missing_information


def test_passing_criteria_is_not_an_automatic_approval() -> None:
    result = evaluate_limit_increase(complete_account(), 120_000, TODAY)

    assert result.status == LimitDecisionStatus.ELIGIBLE_FOR_MANUAL_REVIEW
    assert result.max_permitted_limit == 120_000