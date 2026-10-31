from uuid import uuid4


class InvalidCategory(ValueError):
    """Raised when a service-request category is not supported."""


WORKING_DAY_HOURS = 8

ROUTES: dict[str, tuple[str, int]] = {
    "Card block or replacement": ("CARDS_URGENT", 4),
    "Transaction dispute": ("DISPUTES", 2 * WORKING_DAY_HOURS),
    "Statement or certificate request": ("SELFSERVE", WORKING_DAY_HOURS),
    "Limit change": ("CREDIT_OPS", 3 * WORKING_DAY_HOURS),
    "Standing instruction change": ("PAYMENTS", 2 * WORKING_DAY_HOURS),
    "Complaint": ("COMPLAINTS", 5 * WORKING_DAY_HOURS),
    "GENERAL": ("GENERAL", WORKING_DAY_HOURS),
}


def raise_service_request(
    customer_id: str,
    category: str,
    summary: str,
) -> dict[str, str | int]:
    if not customer_id.strip():
        raise ValueError("customer_id must not be blank")
    if not summary.strip():
        raise ValueError("summary must not be blank")

    route = ROUTES.get(category)
    if route is None:
        accepted = ", ".join(ROUTES)
        raise InvalidCategory(
            f"Unsupported category {category!r}. Expected one of: {accepted}"
        )

    queue, sla_hours = route
    return {
        "request_id": f"SR-{uuid4().hex.upper()}",
        "queue": queue,
        "sla_hours": sla_hours,
    }