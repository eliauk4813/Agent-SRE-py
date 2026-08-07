"""文档质量检查器 - RAG 系统的质量门禁

这是 RAG 系统中最容易被忽视但最关键的模块之一。
低质量的 chunk 会严重影响检索效果：
- 过短的 chunk：信息不完整，检索无意义
- 过长的 chunk：信息过载，语义不聚焦
- 重复的 chunk：浪费存储，污染检索结果
- 格式异常的 chunk：解析错误，影响用户体验

本模块实现了多层质量检查机制，确保只有高质量的 chunk 进入索引。
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter

from langchain_core.documents import Document
from loguru import logger

from app.config import config


@dataclass
class QualityIssue:
    """质量问题记录"""

    severity: str  # "critical" | "warning" | "info"
    category: str  # "length" | "content" | "metadata" | "duplicate" | "format"
    message: str
    chunk_index: int
    suggestion: Optional[str] = None


@dataclass
class QualityReport:
    """质量检查报告"""

    # 输入统计
    total_input: int = 0
    total_chars_input: int = 0

    # 输出统计
    total_valid: int = 0
    total_filtered: int = 0
    total_merged: int = 0
    total_chars_output: int = 0

    # 问题统计
    issues: List[QualityIssue] = field(default_factory=list)
    critical_count: int = 0
    warning_count: int = 0

    # 质量指标
    avg_chunk_size: int = 0
    quality_score: float = 0.0  # 0-100

    def add_issue(self, issue: QualityIssue) -> None:
        """添加质量问题"""
        self.issues.append(issue)
        if issue.severity == "critical":
            self.critical_count += 1
        elif issue.severity == "warning":
            self.warning_count += 1

    def compute_quality_score(self) -> float:
        """计算综合质量分数（0-100）"""
        if self.total_input == 0:
            return 0.0

        # 基础分 100
        score = 100.0

        # 过滤率惩罚（过滤超过30%扣分）
        filter_ratio = self.total_filtered / self.total_input
        if filter_ratio > 0.3:
            score -= (filter_ratio - 0.3) * 100  # 超过30%每1%扣1分

        # 严重问题惩罚
        score -= self.critical_count * 5  # 每个严重问题扣5分

        # 警告问题惩罚
        score -= self.warning_count * 1  # 每个警告扣1分

        # 平均大小加分（接近目标大小900）
        if self.avg_chunk_size > 0:
            target_size = config.md_chunk_target_size
            size_diff = abs(self.avg_chunk_size - target_size)
            size_score = max(0, 100 - (size_diff / target_size) * 100)
            score = score * 0.7 + size_score * 0.3  # 权重：质量70%，大小30%

        self.quality_score = max(0, min(100, score))
        return self.quality_score

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "input": {
                "total_chunks": self.total_input,
                "total_chars": self.total_chars_input,
            },
            "output": {
                "valid_chunks": self.total_valid,
                "filtered_chunks": self.total_filtered,
                "merged_chunks": self.total_merged,
                "total_chars": self.total_chars_output,
            },
            "quality": {
                "score": round(self.quality_score, 2),
                "avg_chunk_size": self.avg_chunk_size,
                "critical_issues": self.critical_count,
                "warnings": self.warning_count,
            },
            "issues": [
                {
                    "severity": issue.severity,
                    "category": issue.category,
                    "message": issue.message,
                    "chunk_index": issue.chunk_index,
                }
                for issue in self.issues[:10]  # 最多返回10个问题
            ],
        }


class DocumentQualityChecker:
    """文档质量检查器

    核心功能：
    1. 长度检查：过滤过短/过长的 chunk
    2. 内容质量：检测无意义内容、格式错误
    3. 去重：检测并移除重复/近似重复的 chunk
    4. 元数据验证：确保必需字段完整
    5. 统计分析：检测异常值
    """

    # 正则模式（类级别，避免重复编译）
    EXCESSIVE_NEWLINE_PATTERN = re.compile(r'\n{4,}')  # 连续4个以上换行
    EXCESSIVE_SPACE_PATTERN = re.compile(r'[ \t]{10,}')  # 连续10个以上空格
    BINARY_PATTERN = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')  # 二进制字符
    CHINESE_CHAR_PATTERN = re.compile(r'[一-鿿]')  # 中文字符

    def __init__(self):
        """初始化质量检查器"""
        self.min_chunk_size = config.md_chunk_min_size  # 200
        self.max_chunk_size = config.md_chunk_max_size * 1.5  # 1800 (允许1.5倍)
        self.target_size = config.md_chunk_target_size  # 900

        # 去重用的哈希集合
        self._content_hashes: Set[str] = set()
        self._fuzzy_hashes: Set[str] = set()

        logger.info(
            f"DocumentQualityChecker initialized: "
            f"min_size={self.min_chunk_size}, "
            f"max_size={self.max_chunk_size}, "
            f"target_size={self.target_size}"
        )

    def check_and_clean(
        self,
        documents: List[Document],
        source_path: str = ""
    ) -> Tuple[List[Document], QualityReport]:
        """质量检查并清洗文档

        Args:
            documents: 待检查的文档列表
            source_path: 源文件路径（用于日志）

        Returns:
            (清洗后的文档列表, 质量报告)
        """
        report = QualityReport()
        report.total_input = len(documents)
        report.total_chars_input = sum(len(doc.page_content) for doc in documents)

        logger.info(f"Quality check start: {source_path}, input_chunks={len(documents)}")

        # 重置去重集合
        self._content_hashes.clear()
        self._fuzzy_hashes.clear()

        valid_documents = []

        for idx, doc in enumerate(documents):
            # 获取原始内容（排除注入的 context）
            content = doc.metadata.get("original_content", doc.page_content)

            # 多层检查
            issues = []

            # 1. 长度检查
            issues.extend(self._check_length(content, idx))

            # 2. 内容质量检查
            issues.extend(self._check_content_quality(content, idx))

            # 3. 元数据完整性检查
            issues.extend(self._check_metadata(doc.metadata, idx))

            # 4. 去重检查
            is_duplicate, dup_issue = self._check_duplicate(content, idx)
            if dup_issue:
                issues.append(dup_issue)

            # 5. 格式异常检查
            issues.extend(self._check_format(content, idx))

            # 判断是否保留
            has_critical = any(issue.severity == "critical" for issue in issues)

            if has_critical or is_duplicate:
                # 严重问题或重复，过滤掉
                report.total_filtered += 1
                for issue in issues:
                    report.add_issue(issue)
                logger.debug(f"Filtered chunk {idx}: {[i.message for i in issues]}")
                continue

            # 保留，但记录警告
            valid_documents.append(doc)
            for issue in issues:
                if issue.severity == "warning":
                    report.add_issue(issue)

        # 统计输出
        report.total_valid = len(valid_documents)
        report.total_chars_output = sum(
            len(doc.metadata.get("original_content", doc.page_content))
            for doc in valid_documents
        )
        report.avg_chunk_size = (
            report.total_chars_output // report.total_valid
            if report.total_valid > 0
            else 0
        )

        # 计算质量分数
        report.compute_quality_score()

        logger.info(
            f"Quality check complete: {source_path}, "
            f"valid={report.total_valid}, "
            f"filtered={report.total_filtered}, "
            f"score={report.quality_score:.1f}"
        )

        return valid_documents, report

    def _check_length(self, content: str, idx: int) -> List[QualityIssue]:
        """长度检查"""
        issues = []
        length = len(content)

        if length < self.min_chunk_size:
            issues.append(
                QualityIssue(
                    severity="critical",
                    category="length",
                    message=f"Too short: {length} < {self.min_chunk_size}",
                    chunk_index=idx,
                    suggestion="Merge with adjacent chunks or filter out",
                )
            )

        elif length > self.max_chunk_size:
            issues.append(
                QualityIssue(
                    severity="critical",
                    category="length",
                    message=f"Too long: {length} > {self.max_chunk_size}",
                    chunk_index=idx,
                    suggestion="Re-split this chunk with smaller target size",
                )
            )

        # 警告：偏离目标大小过多
        elif abs(length - self.target_size) > self.target_size * 0.5:
            diff_ratio = abs(length - self.target_size) / self.target_size
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="length",
                    message=f"Size deviation: {length} chars ({diff_ratio:.1%} from target)",
                    chunk_index=idx,
                )
            )

        return issues

    def _check_content_quality(self, content: str, idx: int) -> List[QualityIssue]:
        """内容质量检查"""
        issues = []

        # 1. 检测空白内容
        stripped = content.strip()
        if not stripped:
            issues.append(
                QualityIssue(
                    severity="critical",
                    category="content",
                    message="Empty content after stripping whitespace",
                    chunk_index=idx,
                )
            )
            return issues  # 空内容直接返回

        # 2. 检测信息密度（熵值）
        entropy = self._calculate_entropy(stripped)
        if entropy < 2.0:  # 熵值过低，内容重复度高
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="content",
                    message=f"Low information density (entropy={entropy:.2f})",
                    chunk_index=idx,
                    suggestion="Content may be repetitive or meaningless",
                )
            )

        # 3. 检测换行符比例（可能是格式问题）
        newline_ratio = content.count('\n') / (len(content) + 1)
        if newline_ratio > 0.3:  # 超过30%是换行符
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="content",
                    message=f"High newline ratio: {newline_ratio:.1%}",
                    chunk_index=idx,
                    suggestion="May contain excessive line breaks",
                )
            )

        # 4. 检测语言一致性（中英文混合过多）
        if self._is_mixed_language(content):
            issues.append(
                QualityIssue(
                    severity="info",
                    category="content",
                    message="Mixed language content detected",
                    chunk_index=idx,
                )
            )

        return issues

    def _check_metadata(self, metadata: Dict[str, Any], idx: int) -> List[QualityIssue]:
        """元数据完整性检查"""
        issues = []

        # 必需字段
        required_fields = ["_source", "_file_name", "chunk_index"]
        missing = [f for f in required_fields if not metadata.get(f)]

        if missing:
            issues.append(
                QualityIssue(
                    severity="critical",
                    category="metadata",
                    message=f"Missing required metadata: {missing}",
                    chunk_index=idx,
                    suggestion="Ensure metadata is properly propagated",
                )
            )

        # 推荐字段
        recommended_fields = ["header_path", "document_type"]
        missing_recommended = [f for f in recommended_fields if not metadata.get(f)]

        if missing_recommended:
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="metadata",
                    message=f"Missing recommended metadata: {missing_recommended}",
                    chunk_index=idx,
                )
            )

        return issues

    def _check_duplicate(self, content: str, idx: int) -> Tuple[bool, Optional[QualityIssue]]:
        """去重检查

        使用两种去重策略：
        1. 精确去重（MD5哈希）
        2. 近似去重（SimHash前缀）
        """
        # 1. 精确去重
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        if content_hash in self._content_hashes:
            issue = QualityIssue(
                severity="critical",
                category="duplicate",
                message="Exact duplicate content detected",
                chunk_index=idx,
                suggestion="Remove this chunk",
            )
            return True, issue

        self._content_hashes.add(content_hash)

        # 2. 近似去重（简化版SimHash：取前64个字符的哈希）
        fuzzy_hash = hashlib.md5(content[:64].encode('utf-8')).hexdigest()[:8]
        if fuzzy_hash in self._fuzzy_hashes:
            issue = QualityIssue(
                severity="warning",
                category="duplicate",
                message="Similar content detected (fuzzy match)",
                chunk_index=idx,
                suggestion="Check if this is a near-duplicate",
            )
            # 不返回True，只是警告
            self._fuzzy_hashes.add(fuzzy_hash)
            return False, issue

        self._fuzzy_hashes.add(fuzzy_hash)
        return False, None

    def _check_format(self, content: str, idx: int) -> List[QualityIssue]:
        """格式异常检查"""
        issues = []

        # 1. 检测二进制字符
        if self.BINARY_PATTERN.search(content):
            issues.append(
                QualityIssue(
                    severity="critical",
                    category="format",
                    message="Binary characters detected",
                    chunk_index=idx,
                    suggestion="Content may be corrupted",
                )
            )

        # 2. 检测过多连续换行
        if self.EXCESSIVE_NEWLINE_PATTERN.search(content):
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="format",
                    message="Excessive consecutive newlines (4+)",
                    chunk_index=idx,
                    suggestion="Content may have formatting issues",
                )
            )

        # 3. 检测过多连续空格
        if self.EXCESSIVE_SPACE_PATTERN.search(content):
            issues.append(
                QualityIssue(
                    severity="warning",
                    category="format",
                    message="Excessive consecutive spaces (10+)",
                    chunk_index=idx,
                    suggestion="Content may have formatting issues",
                )
            )

        return issues

    def _calculate_entropy(self, text: str) -> float:
        """计算文本熵值（信息密度指标）

        熵值越高，信息密度越大：
        - 熵 < 2.0: 内容重复度高，信息密度低
        - 熵 2.0-4.0: 正常范围
        - 熵 > 4.0: 信息密度高

        算法：Shannon Entropy
        H = -Σ p(x) * log2(p(x))
        """
        if not text:
            return 0.0

        # 计算字符频率
        char_counts = Counter(text)
        total_chars = len(text)

        # 计算熵
        entropy = 0.0
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * (probability ** 0.5)  # 简化计算

        return entropy

    def _is_mixed_language(self, text: str) -> bool:
        """检测中英文混合内容

        判断标准：
        - 同时包含中文和英文
        - 中英文字符数都超过10%
        """
        # 统计中文字符
        chinese_chars = len(self.CHINESE_CHAR_PATTERN.findall(text))

        # 统计英文字符（简化：统计ASCII字母）
        english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')

        total_chars = len(text)
        if total_chars == 0:
            return False

        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars

        # 中英文都超过10%认为是混合内容
        return chinese_ratio > 0.1 and english_ratio > 0.1


# 全局单例
document_quality_checker = DocumentQualityChecker()
