# Service Request Routing
Document ID: POL-SVC-006 | Version 1.9 | Effective 10-Mar-2026 | Owner: Service Operations

Requests are routed by intent category. Categories and their owning queues:
- Card block or replacement -> CARDS_URGENT, SLA 4 hours
- Transaction dispute -> DISPUTES, SLA 2 working days
- Statement or certificate request -> SELFSERVE, SLA 1 working day
- Limit change -> CREDIT_OPS, SLA 3 working days
- Standing instruction change -> PAYMENTS, SLA 2 working days
- Complaint -> COMPLAINTS, SLA 5 working days, escalation review at day 3

Any request that cannot be categorised with confidence is routed to GENERAL and
must be reviewed by a human within 1 working day. Guessing a category is a defect.
