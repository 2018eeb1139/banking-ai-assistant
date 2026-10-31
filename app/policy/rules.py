from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from math import floor
from typing import Mapping


class LimitDecisionStatus(str, Enum):
    DECLINED = "declined"
    NOT_ELIGIBLE = "not_eligible"
    NEEDS_REVIEW = "needs_review"
    REFER_TO_CREDIT_OFFICER = "refer_to_credit_officer"
    ELIGIBLE_FOR_MANUAL_REVIEW = "eligible_for_manual_review"


@dataclass(frozen=True)
class LimitDecision:
    status: LimitDecisionStatus
    reason_code: str | None = None
    max_permitted_limit: int | None = None
    requires_credit_officer: bool = False
    compliance_review_required: bool = False
    missing_information: list[str] = field(default_factory=list)
    policy_flags: list[str] = field(default_factory=list)


def _number(account: Mapping[str, object], key: str) -> float | None:
    value = account.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        # Handles leap-day anniversaries in non-leap years.
        return value.replace(month=2, day=28, year=value.year + years)


def _subtract_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 - months
    year, month_index = divmod(month_index, 12)
    month = month_index + 1

    # Clamp to the last day of the target month.
    next_month = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    last_day = (next_month - date.resolution).day
    return date(year, month, min(value.day, last_day))


def _kyc_overdue_more_than_90_days(
    account: Mapping[str, object],
    today: date,
) -> bool | None:
    last_done = _parse_date(account.get("kyc_last_done"))
    risk_band = account.get("kyc_risk_band")

    if last_done is None or not isinstance(risk_band, str):
        return None

    refresh_years = {
        "high": 2,
        "medium": 8,
        "low": 10,
    }.get(risk_band.lower())

    if refresh_years is None:
        return None

    due_date = _add_years(last_done, refresh_years)
    return (today - due_date).days > 90


def evaluate_limit_increase(
    account: Mapping[str, object],
    proposed_limit: int | None,
    today: date | None = None,
) -> LimitDecision:
    """Evaluate policy eligibility; this function never approves an increase."""
    today = today or date.today()

    if proposed_limit is not None and (
        isinstance(proposed_limit, bool)
        or not isinstance(proposed_limit, int)
        or proposed_limit <= 0
    ):
        raise ValueError("proposed_limit must be a positive integer")

    current_limit = _number(account, "current_limit")
    missing: list[str] = []
    flags: list[str] = []
    decline_reasons: list[tuple[str, str | None]] = []
    compliance_review_required = False

    # Mandatory declines are checked before ordinary eligibility.
    vulnerable = account.get("vulnerable_flag")
    if vulnerable is True:
        decline_reasons.append(("vulnerable_customer", None))
        compliance_review_required = True
    elif vulnerable is not False:
        missing.append("vulnerable customer status")

    if account.get("open_dispute") is True:
        decline_reasons.append(("open_dispute", "AA04"))
    elif account.get("open_dispute") is not False:
        missing.append("open dispute status")

    if "last_limit_increase_date" not in account:
        missing.append("date of the most recent limit increase")
    elif account["last_limit_increase_date"] is not None:
        last_increase = _parse_date(account["last_limit_increase_date"])
        if last_increase is None:
            missing.append("valid date of the most recent limit increase")
        elif last_increase > _subtract_months(today, 6):
            decline_reasons.append(("recent_limit_increase", "AA05"))

    kyc_overdue = _kyc_overdue_more_than_90_days(account, today)
    if kyc_overdue is True:
        decline_reasons.append(("overdue_kyc", "AA06"))
    elif kyc_overdue is None:
        missing.append("valid KYC completion date and risk band")

    debt_burden_ratio = _number(
        account,
        "debt_burden_ratio_after_increase_percent",
    )
    if proposed_limit is not None and debt_burden_ratio is not None:
        if debt_burden_ratio > 55:
            decline_reasons.append(("debt_burden_ratio_over_limit", "AA02"))
    else:
        missing.append("debt-burden ratio after the proposed increase")

    if decline_reasons:
        flags.extend(reason for reason, _ in decline_reasons)
        reason_code = next(
            (code for _, code in decline_reasons if code is not None),
            None,
        )
        return LimitDecision(
            status=LimitDecisionStatus.DECLINED,
            reason_code=reason_code,
            requires_credit_officer=compliance_review_required,
            compliance_review_required=compliance_review_required,
            missing_information=missing,
            policy_flags=flags,
        )

    if current_limit is None or current_limit <= 0:
        missing.append("valid current credit limit")

    repayment_months = _number(account, "months_repayment_history")
    if repayment_months is None:
        missing.append("repayment history duration")
    elif repayment_months < 6:
        # The assignment explicitly identifies the three-month seeded case
        # as thin-file; do not make an automated decision for it.
        cap = floor(current_limit * 1.25) if current_limit is not None else None
        return LimitDecision(
            status=LimitDecisionStatus.REFER_TO_CREDIT_OFFICER,
            max_permitted_limit=cap,
            requires_credit_officer=True,
            policy_flags=["thin_file"],
        )

    if proposed_limit is None:
        missing.append("requested limit amount")

    payment_over_30_days = account.get(
        "payment_over_30dpd_last_12_months"
    )
    if payment_over_30_days is True:
        return LimitDecision(
            status=LimitDecisionStatus.NOT_ELIGIBLE,
            reason_code="AA01",
            policy_flags=["recent_missed_payment"],
        )
    if payment_over_30_days is not False:
        missing.append("payment history for the last 12 months")

    utilisation = _number(account, "avg_utilisation_3m")
    behavioural_score = _number(account, "behavioural_score")
    assessed_income = _number(account, "assessed_monthly_net_income")
    bureau_trade_lines = _number(account, "bureau_trade_lines")

    if utilisation is None:
        missing.append("average utilisation over the last three cycles")
    if behavioural_score is None:
        missing.append("eligibility assessment evidence")
    if assessed_income is None or assessed_income <= 0:
        missing.append("assessed monthly net income")
    if bureau_trade_lines is None:
        missing.append("bureau trade-line evidence")

    if missing:
        return LimitDecision(
            status=LimitDecisionStatus.NEEDS_REVIEW,
            missing_information=sorted(set(missing)),
            policy_flags=["insufficient_evidence"],
        )

    if utilisation < 0.40:
        return LimitDecision(
            status=LimitDecisionStatus.NOT_ELIGIBLE,
            compliance_review_required=True,
            policy_flags=["utilisation_eligibility_not_met"],
        )

    if behavioural_score < 620:
        # Do not expose the internal score; the supplied approved reasons
        # do not provide a customer-facing mapping for this criterion.
        return LimitDecision(
            status=LimitDecisionStatus.NOT_ELIGIBLE,
            compliance_review_required=True,
            policy_flags=["eligibility_criterion_not_met"],
        )

    maximum_limit = min(
        int(current_limit * 2),
        floor(assessed_income * 4),
    )

    if bureau_trade_lines < 2:
        thin_file_cap = floor(current_limit * 1.25)
        return LimitDecision(
            status=LimitDecisionStatus.REFER_TO_CREDIT_OFFICER,
            max_permitted_limit=min(maximum_limit, thin_file_cap),
            requires_credit_officer=True,
            policy_flags=["thin_file"],
        )

    if proposed_limit > maximum_limit:
        return LimitDecision(
            status=LimitDecisionStatus.NOT_ELIGIBLE,
            max_permitted_limit=maximum_limit,
            compliance_review_required=True,
            policy_flags=["requested_limit_exceeds_policy_cap"],
        )

    return LimitDecision(
        status=LimitDecisionStatus.ELIGIBLE_FOR_MANUAL_REVIEW,
        max_permitted_limit=maximum_limit,
        policy_flags=["eligible_for_human_review_only"],
    )