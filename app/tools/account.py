import json
import random
import time
from typing import Any

from app.config import settings


class CustomerNotFound(LookupError):
    """Raised when the supplied customer ID is not in the account data."""


class UpstreamTimeout(TimeoutError):
    """Raised when the simulated account service times out."""


def get_account_state(customer_id: str) -> dict[str, Any]:
    time.sleep(0.4)

    if random.random() < 1 / 12:
        raise UpstreamTimeout("Account service timed out")

    accounts_path = settings.data_dir / "accounts.json"
    with accounts_path.open(encoding="utf-8") as account_file:
        accounts = json.load(account_file)

    if not isinstance(accounts, dict):
        raise ValueError("accounts.json must contain a customer-ID mapping")

    account = accounts.get(customer_id)
    if account is None:
        raise CustomerNotFound(f"Customer not found: {customer_id}")
    if not isinstance(account, dict):
        raise ValueError(f"Account record is invalid for customer: {customer_id}")
    if account.get("customer_id") != customer_id:
        raise ValueError(f"Account record ID does not match: {customer_id}")

    return account