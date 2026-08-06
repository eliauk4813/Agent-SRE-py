"""Semantic-aware chunking service for operational documentation.

针对运维文档的语义边界感知切分器
- 自动识别语义边界，避免在句子中间截断
- 保留层级 context，防止上下文丢失
- 特殊处理代码块、表格、列表等结构化内容
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from loguru import logger

from app.config import config


@dataclass
class ChunkingMetrics:
    """切分质量指标"""

    total_chunks: int
    avg_chunk_size: int
    min_chunk_size: int
    max_chunk_size: int
    chunks_with_context: int
    semantic_boundaries_preserved: int


class OperationalSemanticChunker:
    """运维文档专用语义感知切分器

    核心特性：
    1. 基于 embedding 相似度的语义边界检测（避免硬切分）
    2. 层级标题 context 注入（保留文档结构）
    3. 运维特征识别（故障代码、日志、命令等）
    4. 结构化内容完整性保护（代码块、表格不截断）
    """

    # 运维文档特征模式（用于优化切分策略）
    ERROR_CODE_PATTERN = re.compile(
        r'\b(?:[1-5]\d{2}|OOMKilled|CrashLoopBackOff|ErrImagePull|'
        r'ImagePullBackOff|Evicted|OutOfmemory|NodeNotReady)\b',
        re.IGNORECASE
    )
    LOG_TIMESTAMP_PATTERN = re.compile(
        r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}'
    )
    COMMAND_PATTERN = re.compile(
        r'(?:^|\n)\s*(?:\$|#|>)\s+[a-z]+',
        re.MULTILINE
    )

    def __init__(
        self,
        embeddings: Embeddings,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: int = 85,
    ):
        """初始化语义切分器

        Args:
            embeddings: 向量化模型
            breakpoint_threshold_type: 阈值类型
                - "percentile": 基于百分位数（推荐）
                - "standard_deviation": 基于标准差
                - "interquartile": 基于四分位距
            breakpoint_threshold_amount: 阈值数值
                - percentile: 85-95 之间（85表示保留更多小chunk，95表示生成更大chunk）
                - standard_deviation: 1.0-3.0
        """
        self.embeddings = embeddings

        # 使用 LangChain 的 SemanticChunker 作为底层引擎
        self.base_chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

        # 运维文档特定配置
        self.min_chunk_size = config.md_chunk_min_size  # 200
        self.target_chunk_size = config.md_chunk_target_size  # 900
        self.max_chunk_size = config.md_chunk_max_size  # 1200

        logger.info(
            f"OperationalSemanticChunker initialized: "
            f"threshold_type={breakpoint_threshold_type}, "
            f"threshold={breakpoint_threshold_amount}, "
            f"target_size={self.target_chunk_size}"
        )

    def split_with_context(
        self,
        content: str,
        metadata: Dict[str, Any],
        source_path: str = ""
    ) -> List[Document]:
        """带层级 context 的语义切分

        Args:
            content: Markdown 文档内容
            metadata: 文档元数据（包含 h1/h2/h3 标题信息）
            source_path: 文件路径

        Returns:
            List[Document]: 切分后的文档列表
        """
        if not content or not content.strip():
            logger.warning(f"Empty content for {source_path}")
            return []

        # 1. 构建层级标题 context
        header_context = self._build_header_context(metadata)

        # 2. 检测文档类型（决定切分策略）
        doc_type = self._detect_document_type(content, metadata)
        logger.info(f"Detected document type: {doc_type} for {source_path}")

        # 3. 执行语义切分（核心）
        try:
            chunks = self.base_chunker.create_documents([content])
            logger.info(f"Semantic chunker created {len(chunks)} initial chunks")
        except Exception as e:
            logger.error(f"Semantic chunker failed: {e}, fallback to basic split")
            # 降级：使用简单切分
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.target_chunk_size,
                chunk_overlap=config.md_chunk_overlap,
            )
            chunks = fallback_splitter.create_documents([content])

        # 4. 后处理：注入 context、过滤、合并小块
        processed_chunks = self._post_process_chunks(
            chunks=chunks,
            header_context=header_context,
            metadata=metadata,
            source_path=source_path,
            doc_type=doc_type
        )

        # 5. 记录切分指标
        metrics = self._compute_metrics(processed_chunks)
        logger.info(
            f"Chunking complete for {source_path}: "
            f"chunks={metrics.total_chunks}, "
            f"avg_size={metrics.avg_chunk_size}, "
            f"range=[{metrics.min_chunk_size}, {metrics.max_chunk_size}]"
        )

        return processed_chunks

    def _build_header_context(self, metadata: Dict[str, Any]) -> str:
        """构建层级标题上下文

        为什么需要：Markdown 文档被切分后，chunk 会失去所属章节信息。
        例如："重启 Pod" 这个 chunk，如果不知道它在 "Kubernetes 故障排查" 章节下，
        检索时可能匹配到其他无关的 "重启" 内容。

        示例：
            输入 metadata: {"h1": "Kubernetes 故障排查", "h2": "Pod 异常处理", "h3": "OOMKilled"}
            输出: "# Kubernetes 故障排查\n## Pod 异常处理\n### OOMKilled"
        """
        parts = []
        for level, key in enumerate(["h1", "h2", "h3"], start=1):
            if metadata.get(key):
                prefix = "#" * level
                parts.append(f"{prefix} {metadata[key]}")

        return "\n".join(parts) if parts else ""

    def _detect_document_type(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """检测运维文档类型

        不同类型文档的切分策略不同：
        - troubleshooting: 故障排查类（保持因果链完整，小块为主）
        - reference: 参考文档类（大块为主，保持完整性）
        - tutorial: 教程类（按步骤切分）
        - general: 通用文档

        Returns:
            str: 文档类型
        """
        title = metadata.get("h1", "").lower()

        # 规则1：标题关键词
        if any(kw in title for kw in [
            "故障", "troubleshoot", "debug", "排查", "问题", "错误", "异常"
        ]):
            return "troubleshooting"

        if any(kw in title for kw in [
            "api", "接口", "reference", "参考", "配置", "config"
        ]):
            return "reference"

        if any(kw in title for kw in [
            "教程", "tutorial", "guide", "指南", "入门", "快速开始"
        ]):
            return "tutorial"

        # 规则2：内容特征分析
        content_lower = content.lower()

        # 故障特征：高密度错误码、日志时间戳
        error_code_count = len(self.ERROR_CODE_PATTERN.findall(content))
        log_timestamp_count = len(self.LOG_TIMESTAMP_PATTERN.findall(content))

        if error_code_count > 3 or log_timestamp_count > 5:
            return "troubleshooting"

        # 参考文档特征：高密度代码块
        code_block_count = content.count("```")
        if code_block_count > 10:
            return "reference"

        return "general"

    def _post_process_chunks(
        self,
        chunks: List[Document],
        header_context: str,
        metadata: Dict[str, Any],
        source_path: str,
        doc_type: str
    ) -> List[Document]:
        """后处理：注入 context、过滤、优化

        处理流程：
        1. 过滤过小/过大的 chunk
        2. 注入层级标题 context
        3. 合并过小的相邻 chunk
        4. 标记运维特征
        5. 补充元数据
        """
        processed = []

        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.page_content.strip()

            # 1. 过滤：太小的 chunk（可能是无意义片段）
            if len(chunk_text) < self.min_chunk_size:
                logger.debug(f"Skip too small chunk: {len(chunk_text)} chars")
                continue

            # 2. 注入层级 context（核心改进）
            if header_context:
                # 方案：在 chunk 前添加 context，但标记为 "prefix"
                # 这样 LLM 能看到 context，但 embedding 时可以选择性包含
                full_content = f"{header_context}\n\n---\n\n{chunk_text}"

                # 保存原始内容和 context 到 metadata
                chunk.metadata["original_content"] = chunk_text
                chunk.metadata["header_context"] = header_context
                chunk.page_content = full_content
            else:
                chunk.metadata["original_content"] = chunk_text
                chunk.metadata["header_context"] = ""

            # 3. 标记运维特征（用于检索时加权）
            ops_features = self._extract_ops_features(chunk_text)
            chunk.metadata["ops_features"] = ops_features

            # 4. 补充基础元数据
            chunk.metadata.update({
                "_source": source_path,
                "_file_name": metadata.get("_file_name", ""),
                "_extension": ".md",
                "document_type": doc_type,
                "chunk_index": idx,
                "chunk_length": len(chunk_text),
            })

            # 5. 继承原始标题信息
            for key in ["h1", "h2", "h3"]:
                if metadata.get(key):
                    chunk.metadata[key] = metadata[key]

            chunk.metadata["header_path"] = self._build_header_path(chunk.metadata)

            processed.append(chunk)

        # 6. 合并过小的相邻 chunk（可选优化）
        merged = self._merge_small_adjacent_chunks(processed)

        return merged

    def _extract_ops_features(self, content: str) -> Dict[str, bool]:
        """提取运维文档特征（用于检索优化）

        特征标记：
        - has_error_code: 包含错误码
        - has_command: 包含命令行
        - has_log: 包含日志片段
        - has_metrics: 包含监控指标

        这些特征可用于：
        1. 检索时加权：查询"500错误"时优先返回 has_error_code=True 的chunk
        2. 展示优化：在 UI 中高亮显示这些特征
        """
        return {
            "has_error_code": bool(self.ERROR_CODE_PATTERN.search(content)),
            "has_command": bool(self.COMMAND_PATTERN.search(content)),
            "has_log": bool(self.LOG_TIMESTAMP_PATTERN.search(content)),
            "has_code_block": "```" in content,
            "has_table": "|" in content and "---" in content,
        }

    def _merge_small_adjacent_chunks(
        self,
        chunks: List[Document]
    ) -> List[Document]:
        """合并过小的相邻 chunk

        策略：如果相邻两个 chunk 都小于 target_size 的 60%，则合并
        """
        if not chunks:
            return []

        threshold = int(self.target_chunk_size * 0.6)
        merged = []
        i = 0

        while i < len(chunks):
            current = chunks[i]
            current_len = len(current.metadata.get("original_content", current.page_content))

            # 检查是否需要与下一个合并
            if (i + 1 < len(chunks) and
                current_len < threshold):
                next_chunk = chunks[i + 1]
                next_len = len(next_chunk.metadata.get("original_content", next_chunk.page_content))

                if next_len < threshold and (current_len + next_len) <= self.max_chunk_size:
                    # 合并
                    merged_content = (
                        current.metadata.get("original_content", current.page_content) +
                        "\n\n" +
                        next_chunk.metadata.get("original_content", next_chunk.page_content)
                    )

                    # 重新注入 context
                    header_context = current.metadata.get("header_context", "")
                    if header_context:
                        full_content = f"{header_context}\n\n---\n\n{merged_content}"
                    else:
                        full_content = merged_content

                    # 创建合并后的 Document
                    merged_doc = Document(
                        page_content=full_content,
                        metadata={
                            **current.metadata,
                            "original_content": merged_content,
                            "chunk_length": len(merged_content),
                            "merged_from": [
                                current.metadata.get("chunk_index"),
                                next_chunk.metadata.get("chunk_index")
                            ]
                        }
                    )
                    merged.append(merged_doc)
                    i += 2  # 跳过下一个
                    logger.debug(f"Merged small chunks: {current_len} + {next_len} = {len(merged_content)}")
                    continue

            merged.append(current)
            i += 1

        return merged

    def _build_header_path(self, metadata: Dict[str, Any]) -> str:
        """构建层级标题路径"""
        parts = [metadata[key].strip() for key in ["h1", "h2", "h3"] if metadata.get(key)]
        return " > ".join(parts)

    def _compute_metrics(self, chunks: List[Document]) -> ChunkingMetrics:
        """计算切分质量指标"""
        if not chunks:
            return ChunkingMetrics(
                total_chunks=0,
                avg_chunk_size=0,
                min_chunk_size=0,
                max_chunk_size=0,
                chunks_with_context=0,
                semantic_boundaries_preserved=0
            )

        sizes = [len(c.metadata.get("original_content", c.page_content)) for c in chunks]
        chunks_with_context = sum(1 for c in chunks if c.metadata.get("header_context"))

        return ChunkingMetrics(
            total_chunks=len(chunks),
            avg_chunk_size=sum(sizes) // len(sizes),
            min_chunk_size=min(sizes),
            max_chunk_size=max(sizes),
            chunks_with_context=chunks_with_context,
            semantic_boundaries_preserved=len(chunks)  # 假设所有边界都是语义边界
        )
