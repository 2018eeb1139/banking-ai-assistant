# Supplied tools

Two tools are available to your agent. Implement them against the supplied data;
their contracts are fixed and your agent must call them rather than reading files directly.

## get_account_state(customer_id: str) -> dict
Returns the record for that customer from accounts.json.
Raises `CustomerNotFound` for an unknown id.
Simulates a 400ms latency. Fails with `UpstreamTimeout` roughly 1 call in 12 — this is
deliberate and your agent is expected to handle it.

## raise_service_request(customer_id: str, category: str, summary: str) -> dict
Creates a service request. `category` must be one of the queues named in POL-SVC-006.
Returns `{"request_id": "SR-…", "queue": "…", "sla_hours": int}`.
Rejects an unrecognised category with `InvalidCategory` — it will not guess for you.
