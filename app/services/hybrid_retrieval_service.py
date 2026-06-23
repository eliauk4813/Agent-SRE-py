"""Hybrid retrieval with Milvus semantic search, Elasticsearch keyword search, and RRF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.keyword_index_service import KeywordSearchHit, keyword_index_service
from app.services.query_rewrite_service import query_rewrite_service
from app.services.vector_store_manager import vector_store_manager


@dataclass
class RankedDocument:
    id: str
    document: Document
    score: float


class HybridRetrievalService:
    """Combine semantic and keyword retrieval results using Reciprocal Rank Fusion."""

    def search(self, query: str) -> List[Document]:
        rewrite_result = query_rewrite_service.rewrite(query)
        milvus_docs: List[Document] = []
        es_hits: List[KeywordSearchHit] = []

        for rewritten_query in rewrite_result.queries:
            current_milvus_docs = self._search_milvus(rewritten_query, config.rag_milvus_top_k)
            for doc in current_milvus_docs:
                self._attach_query_metadata(
                    doc,
                    rewrite_result.original_query,
                    rewritten_query,
                    rewrite_result.rewrite_applied,
                    rewrite_result.strategy,
                    rewrite_result.reasons,
                )
            milvus_docs.extend(current_milvus_docs)

            current_es_hits = keyword_index_service.search(rewritten_query, config.rag_es_top_k)
            for hit in current_es_hits:
                hit.metadata["original_query"] = rewrite_result.original_query
                hit.metadata["matched_query"] = rewritten_query
                hit.metadata["query_rewrite_applied"] = rewrite_result.rewrite_applied
                hit.metadata["query_rewrite_strategy"] = rewrite_result.strategy
                hit.metadata["query_rewrite_reasons"] = rewrite_result.reasons
            es_hits.extend(current_es_hits)

        ranked = self._rrf(
            milvus_docs=milvus_docs,
            es_hits=es_hits,
            rrf_k=config.rag_rrf_k,
            milvus_weight=config.rag_milvus_weight,
            es_weight=config.rag_es_weight,
        )
        final_docs = [item.document for item in ranked[: config.rag_final_top_k]]
        logger.info(
            "Hybrid retrieval complete: "
            f"queries={len(rewrite_result.queries)}, "
            f"rewrite_applied={rewrite_result.rewrite_applied}, "
            f"milvus={len(milvus_docs)}, es={len(es_hits)}, final={len(final_docs)}"
        )
        return final_docs

    def _search_milvus(self, query: str, top_k: int) -> List[Document]:
        try:
            docs = vector_store_manager.similarity_search(query, k=top_k)
            logger.info(f"Milvus semantic search returned {len(docs)} chunks")
            return docs
        except Exception as exc:
            logger.error(f"Milvus semantic search failed: {exc}")
            return []

    def _attach_query_metadata(
        self,
        doc: Document,
        original_query: str,
        matched_query: str,
        rewrite_applied: bool,
        rewrite_strategy: str,
        rewrite_reasons: List[str],
    ) -> None:
        doc.metadata["original_query"] = original_query
        doc.metadata["matched_query"] = matched_query
        doc.metadata["query_rewrite_applied"] = rewrite_applied
        doc.metadata["query_rewrite_strategy"] = rewrite_strategy
        doc.metadata["query_rewrite_reasons"] = rewrite_reasons

    def _rrf(
        self,
        milvus_docs: List[Document],
        es_hits: List[KeywordSearchHit],
        rrf_k: int,
        milvus_weight: float,
        es_weight: float,
    ) -> List[RankedDocument]:
        scores: Dict[str, float] = {}
        docs: Dict[str, Document] = {}

        self._add_ranked_documents(
            docs=milvus_docs,
            scores=scores,
            documents_by_id=docs,
            weight=milvus_weight,
            rrf_k=rrf_k,
            source_name="milvus",
        )
        self._add_ranked_documents(
            docs=[hit.to_document() for hit in es_hits],
            scores=scores,
            documents_by_id=docs,
            weight=es_weight,
            rrf_k=rrf_k,
            source_name="elasticsearch",
        )

        return sorted(
            [
                RankedDocument(id=doc_id, document=docs[doc_id], score=score)
                for doc_id, score in scores.items()
            ],
            key=lambda item: item.score,
            reverse=True,
        )

    def _add_ranked_documents(
        self,
        docs: Iterable[Document],
        scores: Dict[str, float],
        documents_by_id: Dict[str, Document],
        weight: float,
        rrf_k: int,
        source_name: str,
    ) -> None:
        for rank, doc in enumerate(docs, start=1):
            doc_id = self._document_id(doc)
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank)
            if doc_id not in documents_by_id:
                doc.metadata["retrieval_sources"] = [source_name]
                doc.metadata["rrf_score"] = scores[doc_id]
                documents_by_id[doc_id] = doc
            else:
                sources = documents_by_id[doc_id].metadata.setdefault("retrieval_sources", [])
                if source_name not in sources:
                    sources.append(source_name)
                documents_by_id[doc_id].metadata["rrf_score"] = scores[doc_id]

    def _document_id(self, doc: Document) -> str:
        metadata = doc.metadata
        if metadata.get("chunk_id"):
            return str(metadata["chunk_id"])
        if metadata.get("pk"):
            return str(metadata["pk"])
        source = metadata.get("_source", "")
        chunk_index = metadata.get("chunk_index", "")
        if source != "" and chunk_index != "":
            return f"{source}:{chunk_index}"
        return ""


hybrid_retrieval_service = HybridRetrievalService()
