from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from app.agent.graph import build_agent_graph, create_chat_model
from app.api.routes import router
from app.config import settings
from app.retrieval.hybrid_search import PolicyRetriever
from app.retrieval.ingestion import load_policy_corpus


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.getLogger(__name__).info("step=startup_load_policy")
    chunks = load_policy_corpus(settings.data_dir / "policy_corpus")
    logging.getLogger(__name__).info(
        "step=startup_policy_loaded chunks=%d",
        len(chunks),
    )
    retriever = PolicyRetriever(chunks)
    logging.getLogger(__name__).info("step=startup_retriever_ready")
    chat_model = create_chat_model()

    app.state.agent_graph = build_agent_graph(retriever, chat_model)
    logging.getLogger(__name__).info("step=startup_agent_ready")

    yield

    app.state.agent_graph = None


app = FastAPI(
    title="Banking AI Assistant",
    lifespan=lifespan,
)

app.include_router(router)