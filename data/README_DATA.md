# Supplied data — GenAI / LLM Engineer project

| File | What it is |
| --- | --- |
| policy_corpus/ | 12 governed policy documents, markdown. This is the only grounding your agent may use. |
| accounts.json | 10 synthetic customers and the account state your tools return. |
| tools.md | The contracts for the two supplied tools. Implement them; do not change the contracts. |
| eval_seed.csv | 12 seeded evaluation cases with the expected behaviour. Extend to at least 15 of your own. |

The corpus is deliberately imperfect. It contains at least one contradiction between
documents, at least one question it does not answer, and content that should not be
treated as policy. Finding those is part of the exercise.

All data is synthetic. No real customer information is present.
