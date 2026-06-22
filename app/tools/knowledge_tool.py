"""Knowledge retrieval tool."""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.services.hybrid_retrieval_service import hybrid_retrieval_service


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """Retrieve relevant knowledge chunks for a user question."""
    try:
        logger.info(f"Knowledge retrieval called: query={query!r}")
        docs = hybrid_retrieval_service.search(query)

        if not docs:
            logger.warning("No relevant documents found")
            return "没有找到相关信息。", []

        context = format_docs(docs)
        logger.info(f"Retrieved {len(docs)} relevant documents")
        return context, docs
    except Exception as exc:
        logger.error(f"Knowledge retrieval failed: {exc}")
        return f"检索知识时发生错误: {exc}", []


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents as context text for the LLM."""
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        source = metadata.get("_file_name") or metadata.get("_source", "未知来源")

        headers = []
        for key in ["h1", "h2", "h3"]:
            if metadata.get(key):
                headers.append(str(metadata[key]))
        header_str = " > ".join(headers) if headers else str(metadata.get("header_path", ""))

        formatted = f"【参考资料{i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"
        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
