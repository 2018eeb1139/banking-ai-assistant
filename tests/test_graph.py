import httpx
import pytest
from groq import BadRequestError

from app.agent.graph import _invoke_intent_classifier
from app.agent.state import AgentState


class FakeClassifier:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = outputs
        self.calls = 0

    def invoke(self, prompt: str) -> object:
        self.calls += 1
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def groq_error(code: str) -> BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(
        "Invalid structured output",
        response=response,
        body={"error": {"code": code}},
    )


def test_intent_classifier_retries_invalid_structured_output() -> None:
    classifier = FakeClassifier([groq_error("tool_use_failed"), {"intent": "general"}])

    result = _invoke_intent_classifier(
        classifier,
        "Classify the request",
        AgentState(request_id="request-1"),
    )

    assert result == {"intent": "general"}
    assert classifier.calls == 2


def test_intent_classifier_does_not_retry_other_bad_requests() -> None:
    error = groq_error("invalid_request_error")
    classifier = FakeClassifier([error])

    with pytest.raises(BadRequestError) as raised:
        _invoke_intent_classifier(
            classifier,
            "Classify the request",
            AgentState(request_id="request-1"),
        )

    assert raised.value is error
    assert classifier.calls == 1
