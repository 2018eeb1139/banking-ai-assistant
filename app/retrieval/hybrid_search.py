import re
from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.retrieval.ingestion import PolicyChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: PolicyChunk
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def _to_document(chunk: PolicyChunk) -> Document:
    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "version": chunk.version,
            "effective_date": chunk.effective_date,
            "owner": chunk.owner,
            "section": chunk.section,
            "is_authoritative": chunk.is_authoritative,
        },
    )


class PolicyRetriever:
    def __init__(
        self,
        chunks: list[PolicyChunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rrf_constant: int = 60,
        embeddings: Embeddings | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("At least one policy chunk is required")
        if rrf_constant <= 0:
            raise ValueError("rrf_constant must be greater than zero")

        self.rrf_constant = rrf_constant
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(self._chunks_by_id) != len(chunks):
            raise ValueError("Policy chunk IDs must be unique")

        documents = [_to_document(chunk) for chunk in chunks]
        self._bm25 = BM25Retriever.from_documents(
            documents,
            preprocess_func=_tokenize,
            k=len(documents),
        )

        self._embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector_store = FAISS.from_documents(
            documents,
            self._embeddings,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_non_authoritative: bool = False,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # The corpus is small, so retrieve the full rankings before filtering
        # drafts. This avoids drafts crowding authoritative results out.
        bm25_documents = self._bm25.invoke(query)
        vector_documents = self._vector_store.similarity_search(
            query,
            k=len(self._chunks_by_id),
        )

        fused_scores: dict[str, float] = {}

        for ranking in (bm25_documents, vector_documents):
            for rank, document in enumerate(ranking, start=1):
                chunk_id = document.metadata["chunk_id"]
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                    1.0 / (self.rrf_constant + rank)
                )

        ranked_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )

        results: list[SearchResult] = []
        for chunk_id in ranked_chunk_ids:
            chunk = self._chunks_by_id[chunk_id]
            if not include_non_authoritative and not chunk.is_authoritative:
                continue

            results.append(
                SearchResult(chunk=chunk, score=fused_scores[chunk_id])
            )
            if len(results) >= top_k:
                break

        return results