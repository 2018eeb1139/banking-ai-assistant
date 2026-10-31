from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.agent.graph import build_agent_graph, create_chat_model
from app.api.routes import router
from app.config import settings
from app.retrieval.hybrid_search import PolicyRetriever
from app.retrieval.ingestion import load_policy_corpus


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    chunks = load_policy_corpus(settings.data_dir / "policy_corpus")
    retriever = PolicyRetriever(chunks)
    chat_model = create_chat_model()

    app.state.agent_graph = build_agent_graph(retriever, chat_model)

    yield

    app.state.agent_graph = None


app = FastAPI(
    title="Banking AI Assistant",
    lifespan=lifespan,
)

app.include_router(router)