"""Query rewrite service for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Iterable, List

from loguru import logger
from pydantic import BaseModel, Field

from app.config import config


@dataclass
class QueryRewriteResult:
    """Structured result of query rewriting."""

    original_query: str
    queries: List[str]
    strategy: str
    rewrite_applied: bool
    reasons: List[str] = field(default_factory=list)


class LLMRewriteOutput(BaseModel):
    """Structured output expected from the rewrite model."""

    queries: List[str] = Field(
        description="Retrieval-oriented rewritten queries. Do not include explanations."
    )


class QueryRewriteService:
    """Rewrite user questions into retrieval-friendly queries."""

    _ERROR_CODE_PATTERN = re.compile(r"\b(?:[1-5]\d{2}|OOMKilled|CrashLoopBackOff|ErrImagePull)\b", re.I)
    _LATENCY_PATTERN = re.compile(
        r"(\u6162|\u8d85\u65f6|timeout|latency|\u8017\u65f6|\u5361|\u54cd\u5e94\u6162)",
        re.I,
    )
    _ERROR_PATTERN = re.compile(
        r"(\u62a5\u9519|\u5931\u8d25|\u5f02\u5e38|error|exception|fail|502|503|504|500)",
        re.I,
    )
    _LOG_PATTERN = re.compile(
        r"(\u65e5\u5fd7|log|trace|\u94fe\u8def|\u544a\u8b66|alert|metric|\u6307\u6807)",
        re.I,
    )
    _OPS_PATTERN = re.compile(
        r"(\u670d\u52a1|\u63a5\u53e3|pod|k8s|kubernetes|ingress|nginx|"
        r"\u7f51\u5173|\u6570\u636e\u5e93|redis|mysql|cpu|\u5185\u5b58|"
        r"\u78c1\u76d8|jvm|gc|\u5bb9\u5668|\u90e8\u7f72|\u53d1\u5e03|"
        r"\u91cd\u542f|\u76d1\u63a7)",
        re.I,
    )
    _FOLLOW_UP_PATTERN = re.compile(
        r"^(\u90a3|\u8fd9\u4e2a|\u5b83|\u4e0a\u9762|\u521a\u624d|\u7ee7\u7eed|"
        r"\u600e\u4e48|\u5982\u4f55|\u4e3a\u4ec0\u4e48|\u539f\u56e0|"
        r"\u5904\u7406|\u6062\u590d)",
        re.I,
    )

    def rewrite(self, query: str) -> QueryRewriteResult:
        """Return original query plus optional rewritten variants."""
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return QueryRewriteResult(
                original_query=query,
                queries=[],
                strategy="disabled",
                rewrite_applied=False,
                reasons=["empty_query"],
            )

        if not config.rag_query_rewrite_enabled:
            return QueryRewriteResult(
                original_query=normalized_query,
                queries=[normalized_query],
                strategy="disabled",
                rewrite_applied=False,
                reasons=["config_disabled"],
            )

        should_rewrite, reasons = self._should_rewrite(normalized_query)
        if not should_rewrite:
            return QueryRewriteResult(
                original_query=normalized_query,
                queries=[normalized_query],
                strategy="none",
                rewrite_applied=False,
                reasons=reasons,
            )

        strategy = config.rag_query_rewrite_strategy.strip().lower()
        rewritten_queries = self._rewrite_by_rules(normalized_query)

        if strategy in {"llm", "hybrid"}:
            rewritten_queries.extend(self._rewrite_by_llm(normalized_query, reasons))

        queries = self._deduplicate_queries(
            [normalized_query, *rewritten_queries],
            max_queries=config.rag_query_rewrite_max_queries,
        )

        rewrite_applied = len(queries) > 1
        logger.info(
            "Query rewrite complete: "
            f"strategy={strategy}, applied={rewrite_applied}, "
            f"reasons={reasons}, queries={queries}"
        )
        return QueryRewriteResult(
            original_query=normalized_query,
            queries=queries,
            strategy=strategy,
            rewrite_applied=rewrite_applied,
            reasons=reasons,
        )

    def _should_rewrite(self, query: str) -> tuple[bool, List[str]]:
        reasons: List[str] = []

        if len(query) <= 4:
            reasons.append("too_short")
            return False, reasons

        if self._ERROR_CODE_PATTERN.search(query):
            reasons.append("contains_error_code")
        if self._ERROR_PATTERN.search(query):
            reasons.append("contains_error_signal")
        if self._LATENCY_PATTERN.search(query):
            reasons.append("contains_latency_signal")
        if self._LOG_PATTERN.search(query):
            reasons.append("contains_observability_signal")
        if self._OPS_PATTERN.search(query):
            reasons.append("contains_ops_entity")
        if self._FOLLOW_UP_PATTERN.search(query):
            reasons.append("looks_like_follow_up_or_question")
        if len(query) >= 20:
            reasons.append("long_natural_language_query")

        if not reasons:
            reasons.append("no_rewrite_signal")
            return False, reasons
        return True, reasons

    def _rewrite_by_rules(self, query: str) -> List[str]:
        variants: List[str] = []

        if self._ERROR_PATTERN.search(query):
            variants.append(f"{query} common causes troubleshooting solution")
            variants.append(f"{query} upstream service gateway timeout logs")

        if self._LATENCY_PATTERN.search(query):
            variants.append(f"{query} performance bottleneck slow query resource usage")

        if self._LOG_PATTERN.search(query):
            variants.append(f"{query} logs trace alerts metrics correlation analysis")

        if self._OPS_PATTERN.search(query):
            variants.append(f"{query} sre incident diagnosis root cause recovery steps")

        return variants

    def _rewrite_by_llm(self, query: str, reasons: Iterable[str]) -> List[str]:
        if not config.dashscope_api_key:
            logger.info("Skip LLM query rewrite: dashscope_api_key is empty")
            return []

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_qwq import ChatQwen

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        dedent(
                            """
                            You are a retrieval query rewriter.
                            Rewrite the user's question into search-friendly queries.
                            Requirements:
                            - Preserve the original intent and do not introduce conclusions.
                            - Output 1 to 3 queries.
                            - Cover symptoms, key entities, error codes, logs, metrics, and diagnosis actions.
                            - Do not output explanations.
                            """
                        ).strip(),
                    ),
                    (
                        "user",
                        "User question: {query}\nRewrite signals: {reasons}\nGenerate retrieval queries.",
                    ),
                ]
            )
            llm = ChatQwen(
                model=config.rag_query_rewrite_model,
                api_key=config.dashscope_api_key,
                temperature=config.rag_query_rewrite_temperature,
            )
            chain = prompt | llm.with_structured_output(LLMRewriteOutput)
            result = chain.invoke({"query": query, "reasons": ", ".join(reasons)})
            return [self._normalize_query(item) for item in result.queries if item.strip()]
        except Exception as exc:
            logger.warning(f"LLM query rewrite failed, fallback to rule rewrite: {exc}")
            return []

    def _deduplicate_queries(self, queries: Iterable[str], max_queries: int) -> List[str]:
        seen: set[str] = set()
        deduplicated: List[str] = []

        for query in queries:
            normalized = self._normalize_query(query)
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            deduplicated.append(normalized)
            if len(deduplicated) >= max(1, max_queries):
                break

        return deduplicated

    def _normalize_query(self, query: str) -> str:
        return " ".join(str(query).strip().split())


query_rewrite_service = QueryRewriteService()
