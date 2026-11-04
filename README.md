# Banking AI Assistant

A demonstration banking customer-service assistant built with Groq, LangGraph,
LangChain hybrid retrieval, local FAISS, and deterministic credit-limit rules.
The assistant grounds customer-facing answers in the supplied policy corpus,
uses synthetic account records when needed, and validates policy citations.

## Features

- Classifies customer requests into supported intents.
- Retrieves policy documents using BM25 and FAISS vector search, with reciprocal
  rank fusion.
- Keeps non-authoritative draft content from being treated as policy evidence.
- Looks up account information only for account-dependent requests.
- Evaluates credit-limit requests with deterministic rules; it does not
  automatically approve a limit increase.
- Routes supported service requests to a queue and returns a request ID and SLA.
- Returns a structured response with citations, refusal, decline, or review
  information.
- Exposes the workflow through a FastAPI endpoint.

See [architecture.md](./architecture.md) for a workflow diagram.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- A Groq API key
- Internet access for Groq requests and initial embedding-model download

## Setup

Install the project and development dependencies:

```bash
uv sync
```

Create a local environment file:

```bash
cp .env.example .env
```

Edit `.env` and set your Groq API key:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3.8-27b
DATA_DIR=data
```

Keep `.env` private and never commit your API key. The `.env.example` file is a
safe template with no secret value. The model name can be changed to another
model available to your Groq account.

## Run the API

Start the local development server:

```bash
uv run uvicorn app.main:app --reload
```

At startup the app loads the policy corpus, initializes local retrieval, and
creates the chat model. The API listens at `http://127.0.0.1:8000`.

### Health check

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{ "status": "ok" }
```

### Send a customer question

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"CUST-2000","question":"What is the foreign currency markup?"}'
```

The request accepts:

| Field         | Required | Description                                       |
| ------------- | -------- | ------------------------------------------------- |
| `customer_id` | Yes      | Synthetic customer ID, up to 64 characters        |
| `question`    | Yes      | Customer question, up to 4,000 characters         |
| `request_id`  | No       | Optional UUID; generated automatically if omitted |

The response follows the `AgentResponse` schema. It contains an `outcome`,
`message`, `intent`, claims and citations, and—when applicable—a service request,
reason code, or review information.

Interactive API documentation is available at
[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs).

If Groq rate-limits a request, `/chat` returns HTTP `503` with a retry message.
Wait and retry; repeated calls can continue to hit the account's rate or usage
limit.

## Run the Streamlit chat UI

Start the FastAPI backend in one terminal:

```bash
uv run uvicorn app.main:app --reload
```

In another terminal, launch the Streamlit interface:

```bash
uv run streamlit run streamlit_app.py
```

The UI connects to `http://127.0.0.1:8000` by default. To use a different
backend URL, set `BANKING_API_URL` before starting Streamlit:

```bash
BANKING_API_URL=http://127.0.0.1:8001 uv run streamlit run streamlit_app.py
```

Choose a supplied synthetic customer ID in the sidebar, then ask a question.
Conversation history is kept in the Streamlit session and can be cleared from
the sidebar.

### Trace a request

The Streamlit UI shows the generated request ID above each assistant response.
Use it to follow the request across both terminal windows:

- The Streamlit terminal logs request submission, the API status, and response
  rendering.
- The FastAPI terminal logs API receipt, intent classification, policy
  retrieval, evidence checks, workflow routing, response generation, citation
  validation, and the final outcome.

For example, asking `What is the foreign currency markup?` should produce
backend events for intent classification, retrieval of `POL-FEE-007`, evidence
assessment, response drafting, citation validation, and an `answer` outcome.
Logs include request IDs and document IDs, but do not print the question,
account values, or API key.

## Run tests

Run the local test suite without calling Groq:

```bash
uv run python -m pytest -q
```

## Run the live evaluation

Check the evaluation datasets without calling Groq:

```bash
uv run evaluation/evaluator.py --validate-data
```

Run the full evaluation:

```bash
uv run evaluation/evaluator.py
```

The live evaluation invokes the Groq model for each case and may take several
minutes or be interrupted by API rate limits. Select a smaller run while
debugging:

```bash
# Run the first three cases
uv run evaluation/evaluator.py --limit 3

# Run selected cases
uv run evaluation/evaluator.py --case-id EV-001 --case-id EVX-015
```

The latest evaluation report is written to
[`evaluation/results/latest.json`](./evaluation/results/latest.json). A case
passes only if all applicable checks pass. An API rate-limit error is a run-time
failure, not evidence that the underlying case behavior is incorrect.

## Project layout

```text
app/
  agent/        LangGraph workflow and response generation
  api/          FastAPI routes
  policy/       Deterministic credit-limit policy rules
  retrieval/    Policy ingestion and hybrid search
  schemas/      Request, response, claim, and citation schemas
  tools/        Local account lookup and service-request tools
  validation/   Citation validation
data/
  policy_corpus/ Supplied policy documents
  accounts.json  Synthetic account records
  eval_seed.csv  Supplied evaluation cases
evaluation/
  eval_additional.csv  Additional evaluation cases
  evaluator.py         Live evaluation runner
  results/             Generated evaluation reports
tests/                  Unit and workflow tests
```

The supplied data is described in [data/README_DATA.md](./data/README_DATA.md).
Do not overwrite or modify the original `data/eval_seed.csv` assignment cases.

## Current validation status

Last checked on 2026-10-31:

- `uv run python -m pytest -q`: **54 passed**.
- Evaluation data validation: **27 cases**.
- Latest complete live run: **26/27 cases passed**.
- The remaining failure is EV-003. The supplied policy requires declining a
  vulnerable customer's limit-increase request, but does not provide an
  approved customer-facing reason code for that situation. The assistant
  returns `needs_review` rather than inventing a decline reason; the evaluation
  expects `decline_with_reason`.
