"""Markdown document splitting service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from app.config import config


HEADER_KEYS = ("h1", "h2", "h3")
CODE_FENCE_RE = re.compile(r"^```")
ORDERED_LIST_RE = re.compile(r"^\s*\d+\.\s+")
UNORDERED_LIST_RE = re.compile(r"^\s*[-*+]\s+")
QUOTE_RE = re.compile(r"^\s*>")
TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:-]+(?:\|[\s:-]+)+\|?\s*$")


@dataclass
class MarkdownBlock:
    """A semantic Markdown block within a section."""

    block_type: str
    content: str


class DocumentSplitterService:
    """Split Markdown documents into retrieval-friendly chunks."""

    def __init__(self) -> None:
        self.target_chunk_size = config.md_chunk_target_size
        self.max_chunk_size = config.md_chunk_max_size
        self.chunk_overlap = config.md_chunk_overlap
        self.min_chunk_size = config.md_chunk_min_size

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            "Document splitter initialized for Markdown: "
            f"target={self.target_chunk_size}, max={self.max_chunk_size}, "
            f"overlap={self.chunk_overlap}, min={self.min_chunk_size}"
        )

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """Split a Markdown document into chunks."""
        if file_path and not file_path.endswith(".md"):
            raise ValueError(f"Only Markdown files are supported: {file_path}")
        return self.split_markdown(content, file_path)

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """Split Markdown content by structure first, length second."""
        normalized = self._normalize_content(content)
        if not normalized:
            logger.warning(f"Markdown content is empty: {file_path}")
            return []

        source_path = file_path
        file_name = Path(file_path).name if file_path else ""
        final_docs: List[Document] = []

        try:
            sections = self.markdown_splitter.split_text(normalized)
            for section_index, section_doc in enumerate(sections):
                blocks = self._split_markdown_section_blocks(section_doc)
                assembled_docs = self._assemble_blocks_to_chunks(
                    blocks=blocks,
                    section_doc=section_doc,
                    source_path=source_path,
                    file_name=file_name,
                    section_index=section_index,
                )
                final_docs.extend(assembled_docs)

            for chunk_index, doc in enumerate(final_docs):
                doc.metadata["chunk_index"] = chunk_index

            logger.info(
                f"Markdown split complete: {file_path or '<memory>'} -> {len(final_docs)} chunks"
            )
            return final_docs
        except Exception as exc:
            logger.error(f"Markdown split failed: {file_path}, error: {exc}")
            raise

    def _normalize_content(self, content: str) -> str:
        """Normalize line endings and trim outer whitespace."""
        if not content or not content.strip():
            return ""
        return content.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _split_markdown_section_blocks(self, section_doc: Document) -> List[MarkdownBlock]:
        """Split a section into semantic Markdown blocks."""
        lines = section_doc.page_content.split("\n")
        blocks: List[MarkdownBlock] = []
        index = 0

        while index < len(lines):
            line = lines[index]

            if not line.strip():
                index += 1
                continue

            if CODE_FENCE_RE.match(line.strip()):
                block_lines = [line]
                index += 1
                while index < len(lines):
                    block_lines.append(lines[index])
                    if CODE_FENCE_RE.match(lines[index].strip()):
                        index += 1
                        break
                    index += 1
                blocks.append(MarkdownBlock("code", "\n".join(block_lines).strip()))
                continue

            if self._is_table_start(lines, index):
                block_lines = [line]
                index += 1
                while index < len(lines) and TABLE_RE.match(lines[index]):
                    block_lines.append(lines[index])
                    index += 1
                blocks.append(MarkdownBlock("table", "\n".join(block_lines).strip()))
                continue

            if self._is_list_line(line):
                block_lines = [line]
                index += 1
                while index < len(lines):
                    next_line = lines[index]
                    if not next_line.strip():
                        lookahead = index + 1
                        if lookahead < len(lines) and self._is_list_line(lines[lookahead]):
                            block_lines.append(next_line)
                            index += 1
                            continue
                        break
                    if self._is_list_line(next_line) or next_line.startswith("  ") or next_line.startswith("\t"):
                        block_lines.append(next_line)
                        index += 1
                        continue
                    break
                blocks.append(MarkdownBlock("list", "\n".join(block_lines).strip()))
                continue

            if QUOTE_RE.match(line):
                block_lines = [line]
                index += 1
                while index < len(lines) and lines[index].strip():
                    if not QUOTE_RE.match(lines[index]):
                        break
                    block_lines.append(lines[index])
                    index += 1
                blocks.append(MarkdownBlock("quote", "\n".join(block_lines).strip()))
                continue

            block_lines = [line]
            index += 1
            while index < len(lines):
                next_line = lines[index]
                if not next_line.strip():
                    break
                if (
                    CODE_FENCE_RE.match(next_line.strip())
                    or self._is_table_start(lines, index)
                    or self._is_list_line(next_line)
                    or QUOTE_RE.match(next_line)
                ):
                    break
                block_lines.append(next_line)
                index += 1
            blocks.append(MarkdownBlock("paragraph", "\n".join(block_lines).strip()))

        return blocks

    def _assemble_blocks_to_chunks(
        self,
        blocks: List[MarkdownBlock],
        section_doc: Document,
        source_path: str,
        file_name: str,
        section_index: int,
    ) -> List[Document]:
        """Assemble semantic blocks into final chunks."""
        docs: List[Document] = []
        current_blocks: List[MarkdownBlock] = []

        for block in blocks:
            if self._should_split_block(block):
                docs.extend(
                    self._flush_current_blocks(
                        current_blocks, section_doc.metadata, source_path, file_name, section_index
                    )
                )
                current_blocks = []
                docs.extend(
                    self._split_oversized_block(
                        block, section_doc.metadata, source_path, file_name, section_index
                    )
                )
                continue

            candidate_blocks = current_blocks + [block]
            candidate_size = self._blocks_size(candidate_blocks)

            if candidate_size <= self.target_chunk_size:
                current_blocks = candidate_blocks
                continue

            if current_blocks and self._blocks_size(current_blocks) >= self.min_chunk_size:
                docs.extend(
                    self._flush_current_blocks(
                        current_blocks, section_doc.metadata, source_path, file_name, section_index
                    )
                )
                current_blocks = [block]
                continue

            current_blocks = candidate_blocks
            if candidate_size > self.max_chunk_size:
                docs.extend(
                    self._flush_current_blocks(
                        current_blocks, section_doc.metadata, source_path, file_name, section_index
                    )
                )
                current_blocks = []

        docs.extend(
            self._flush_current_blocks(
                current_blocks, section_doc.metadata, source_path, file_name, section_index
            )
        )
        return docs

    def _flush_current_blocks(
        self,
        blocks: List[MarkdownBlock],
        metadata: Dict[str, Any],
        source_path: str,
        file_name: str,
        section_index: int,
    ) -> List[Document]:
        """Emit a chunk from accumulated blocks."""
        if not blocks:
            return []

        content = "\n\n".join(block.content for block in blocks if block.content.strip()).strip()
        if not content:
            return []

        doc_metadata = self._build_chunk_metadata(
            metadata=metadata,
            source_path=source_path,
            file_name=file_name,
            section_index=section_index,
            block_types=sorted({block.block_type for block in blocks}),
        )
        return [Document(page_content=content, metadata=doc_metadata)]

    def _split_oversized_block(
        self,
        block: MarkdownBlock,
        metadata: Dict[str, Any],
        source_path: str,
        file_name: str,
        section_index: int,
    ) -> List[Document]:
        """Split an oversized block with a recursive fallback splitter."""
        fallback_docs = self.fallback_splitter.create_documents([block.content])
        split_docs: List[Document] = []
        for doc in fallback_docs:
            split_docs.append(
                Document(
                    page_content=doc.page_content.strip(),
                    metadata=self._build_chunk_metadata(
                        metadata=metadata,
                        source_path=source_path,
                        file_name=file_name,
                        section_index=section_index,
                        block_types=[block.block_type],
                    ),
                )
            )
        return [doc for doc in split_docs if doc.page_content]

    def _build_chunk_metadata(
        self,
        metadata: Dict[str, Any],
        source_path: str,
        file_name: str,
        section_index: int,
        block_types: List[str],
    ) -> Dict[str, Any]:
        """Build metadata for the final chunk."""
        chunk_metadata: Dict[str, Any] = {
            "_source": source_path,
            "_extension": ".md",
            "_file_name": file_name,
            "section_index": section_index,
            "block_types": block_types,
        }
        for key in HEADER_KEYS:
            value = metadata.get(key)
            if value:
                chunk_metadata[key] = value
        chunk_metadata["header_path"] = self._build_header_path(chunk_metadata)
        return chunk_metadata

    def _build_header_path(self, metadata: Dict[str, Any]) -> str:
        """Build a hierarchical header path from metadata."""
        parts = [metadata[key].strip() for key in HEADER_KEYS if metadata.get(key)]
        return " > ".join(parts)

    def _blocks_size(self, blocks: List[MarkdownBlock]) -> int:
        """Calculate the assembled size of blocks."""
        if not blocks:
            return 0
        return len("\n\n".join(block.content for block in blocks))

    def _should_split_block(self, block: MarkdownBlock) -> bool:
        """Decide whether a block needs fallback splitting."""
        block_size = len(block.content)
        if block.block_type in {"code", "table"}:
            return block_size > int(self.max_chunk_size * 1.5)
        return block_size > self.max_chunk_size

    def _is_table_start(self, lines: List[str], index: int) -> bool:
        """Detect whether the current line begins a Markdown table."""
        if index + 1 >= len(lines):
            return False
        return bool(TABLE_RE.match(lines[index]) and TABLE_SEPARATOR_RE.match(lines[index + 1]))

    def _is_list_line(self, line: str) -> bool:
        """Detect unordered or ordered Markdown list lines."""
        return bool(ORDERED_LIST_RE.match(line) or UNORDERED_LIST_RE.match(line))


document_splitter_service = DocumentSplitterService()
