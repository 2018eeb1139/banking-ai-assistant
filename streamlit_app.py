import json
import os
from typing import Any

import requests
import streamlit as st
from pydantic import ValidationError

from app.schemas.request import AgentRequest
from app.schemas.response import AgentResponse, Outcome

API_URL = os.getenv("BANKING_API_URL", "http://127.0.0.1:8000").rstrip("/")
CHAT_URL = f"{API_URL}/chat"
REQUEST_TIMEOUT_SECONDS = 120

OUTCOME_LABELS = {
    Outcome.ANSWER: "Answer provided",
    Outcome.DECLINE_WITH_REASON: "Request declined",
    Outcome.REFUSE: "Request refused",
    Outcome.RAISE_REQUEST: "Service request raised",
    Outcome.NEEDS_REVIEW: "Manual review required",
}


def _citation_key(citation: dict[str, Any]) -> str:
    return json.dumps(citation, sort_keys=True, ensure_ascii=False)


def _render_citation(citation: dict[str, Any]) -> None:
    source_type = citation.get("source_type", "unknown")
    source_id = citation.get("source_id", "Unknown source")
    st.markdown(f"**{source_type.replace('_', ' ').title()} - {source_id}**")

    details = []
    if citation.get("document_version"):
        details.append(f"Version: {citation['document_version']}")
    if citation.get("section"):
        details.append(f"Section: {citation['section']}")
    if citation.get("account_field"):
        details.append(f"Account field: {citation['account_field']}")
    if details:
        st.caption(" | ".join(details))
    if citation.get("evidence"):
        st.write(citation["evidence"])


def _render_response(response_data: dict[str, Any]) -> None:
    try:
        response = AgentResponse.model_validate(response_data)
    except ValidationError:
        st.error("The service returned a response in an unexpected format.")
        return

    st.markdown(response.message)
    st.caption(f"Status: **{OUTCOME_LABELS[response.outcome]}**")

    if (
        response.outcome == Outcome.RAISE_REQUEST
        and response.service_request is not None
    ):
        service_request = response.service_request
        st.success("Service request raised successfully.")
        st.write(f"**Request ID:** {service_request.request_id}")
        st.write(f"**Queue:** {service_request.queue}")
        st.write(f"**SLA:** {service_request.sla_hours} hours")

    if response.reason_code:
        with st.expander("Decision Details"):
            st.write(f"**Reason code:** {response.reason_code}")

    if response.missing_information:
        st.markdown("**Additional Information Required**")
        for item in response.missing_information:
            st.markdown(f"- {item}")

    if response.next_action:
        st.markdown("**Next Steps**")
        st.write(response.next_action)

    citation_data = [
        citation.model_dump(mode="json")
        for citation in response.citations
    ]
    unique_citations = {
        _citation_key(citation): citation for citation in citation_data
    }

    with st.expander("Policy Evidence & Citations"):
        if not unique_citations:
            st.write("No supporting citations were returned for this response.")
        else:
            for citation in unique_citations.values():
                _render_citation(citation)

        if response.claims:
            st.markdown("**Claims and supporting citations**")
            for claim in response.claims:
                st.write(claim.text)
                source_ids = list(
                    dict.fromkeys(
                        f"{citation.source_type.value}: {citation.source_id}"
                        for citation in claim.citations
                    )
                )
                st.caption("Supported by: " + ", ".join(source_ids))


def _clear_conversation() -> None:
    st.session_state.messages = []


st.set_page_config(
    page_title="BusinessNEXT AI Assistant",
    page_icon="💬",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("BusinessNEXT AI Assistant")
st.caption("Policy-aware customer service assistant")

with st.sidebar:
    st.subheader("Customer")
    st.text_input(
        "Customer ID",
        key="customer_id",
        max_chars=64,
        placeholder="Enter a customer ID",
    )
    with st.expander("Example customer IDs"):
        st.caption("These examples are present in the supplied synthetic data.")
        st.code("CUST-2000\nCUST-2002\nCUST-2005")

    st.button(
        "Clear Conversation",
        on_click=_clear_conversation,
        use_container_width=True,
    )
    st.caption(
        "Responses use the selected customer's account information when needed "
        "and the supplied business policies."
    )
    st.caption(f"Backend: {API_URL}")

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        if item["role"] == "assistant" and "response" in item:
            _render_response(item["response"])
        else:
            st.markdown(item["content"])

question = st.chat_input(
    "Ask a banking service question",
    max_chars=4000,
)
if question:
    customer_id = st.session_state.get("customer_id", "").strip()
    if not customer_id:
        st.warning("Enter a customer ID in the sidebar before asking a question.")
    else:
        request = AgentRequest(customer_id=customer_id, question=question)
        st.session_state.messages.append(
            {"role": "user", "content": request.question}
        )
        with st.chat_message("user"):
            st.markdown(request.question)

        with st.chat_message("assistant"):
            with st.spinner("Checking policies and preparing a response..."):
                try:
                    api_response = requests.post(
                        CHAT_URL,
                        json=request.model_dump(mode="json"),
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                    api_response.raise_for_status()
                    response = AgentResponse.model_validate(api_response.json())
                except requests.Timeout:
                    error_message = (
                        "The assistant took too long to respond. Please try again."
                    )
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )
                except requests.HTTPError as exc:
                    try:
                        detail = exc.response.json().get("detail")
                    except (AttributeError, ValueError):
                        detail = None
                    error_message = (
                        str(detail)
                        if detail
                        else "The assistant service could not process the request."
                    )
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )
                except requests.RequestException:
                    error_message = (
                        "Could not reach the assistant service. Check that the "
                        "FastAPI backend is running and try again."
                    )
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )
                except ValidationError:
                    error_message = (
                        "The assistant service returned an unexpected response."
                    )
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )
                else:
                    response_data = response.model_dump(mode="json")
                    _render_response(response_data)
                    st.session_state.messages.append(
                        {"role": "assistant", "response": response_data}
                    )
