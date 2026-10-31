import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from groq import RateLimitError

from app.schemas.request import AgentRequest
from app.schemas.response import AgentResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=AgentResponse)
def chat(payload: AgentRequest, request: Request) -> AgentResponse:
    graph: Any = request.app.state.agent_graph

    try:
        state = graph.invoke(
            {
                "request_id": str(payload.request_id),
                "customer_id": payload.customer_id,
                "question": payload.question,
            }
        )
    except RateLimitError as exc:
        logger.warning(
            "Groq rate limit reached for request_id=%s",
            payload.request_id,
        )
        raise HTTPException(
            status_code=503,
            detail="The AI service is rate-limited. Please try again shortly.",
        ) from exc

    response = state.get("final_response")
    if not isinstance(response, AgentResponse):
        logger.error(
            "Agent did not produce a valid response for request_id=%s",
            payload.request_id,
        )
        raise HTTPException(
            status_code=500,
            detail="The agent could not produce a valid response.",
        )

    return response