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
    request_id = str(payload.request_id)
    logger.info(
        "request_id=%s step=api_received",
        request_id,
    )

    try:
        state = graph.invoke(
            {
                "request_id": request_id,
                "customer_id": payload.customer_id,
                "question": payload.question,
            }
        )
    except RateLimitError as exc:
        logger.warning(
            "request_id=%s step=api_failed reason=groq_rate_limit",
            request_id,
        )
        raise HTTPException(
            status_code=503,
            detail="The AI service is rate-limited. Please try again shortly.",
        ) from exc
    except Exception as exc:
        logger.error(
            "request_id=%s step=api_failed error_type=%s",
            request_id,
            type(exc).__name__,
        )
        raise

    response = state.get("final_response")
    if not isinstance(response, AgentResponse):
        logger.error(
            "request_id=%s step=api_failed reason=invalid_agent_response",
            request_id,
        )
        raise HTTPException(
            status_code=500,
            detail="The agent could not produce a valid response.",
        )

    logger.info(
        "request_id=%s step=api_response outcome=%s citations=%d",
        request_id,
        response.outcome.value,
        len(response.citations),
    )
    return response