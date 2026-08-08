"""
RAG评估数据集生成器

从现有运维文档中自动生成评估问答对，无需大量人工标注。

生成策略：
1. 简单事实问题：从文档中提取关键信息生成问答
2. 故障排查问题：基于troubleshooting文档生成场景问题
3. 配置查询问题：关于配置参数、命令的问题
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from app.config import config


@dataclass
class EvaluationSample:
    """单个评估样本"""

    question: str  # 用户问题
    ground_truth: str  # 标准答案
    contexts: List[str]  # 相关文档片段
    metadata: dict  # 元数据（文档来源、难度等）


class EvaluationDatasetGenerator:
    """评估数据集生成器

    从运维文档自动生成评估问答对，减少人工标注成本。
    """

    # 问题生成提示模板
    QUESTION_GENERATION_PROMPT = """你是一个专业的运维工程师。基于以下文档内容，生成{num_questions}个高质量的运维问题和答案。

文档内容：
{document_content}

要求：
1. 问题必须能够从文档内容中直接回答
2. 涵盖不同难度：简单事实查询、故障排查、配置查询
3. 问题要具体、实用，符合真实运维场景
4. 每个问题提供标准答案（ground truth）

请以JSON格式返回，格式如下：
{{
  "questions": [
    {{
      "question": "具体问题",
      "answer": "标准答案",
      "difficulty": "easy/medium/hard",
      "category": "fact/troubleshooting/configuration"
    }}
  ]
}}

只返回JSON，不要其他内容。"""

    def __init__(self):
        """初始化生成器"""
        self.llm = self._create_llm()
        logger.info("EvaluationDatasetGenerator initialized")

    def _create_llm(self) -> ChatOpenAI:
        """创建LLM实例"""
        return ChatOpenAI(
            model=config.dashscope_model,
            api_key=config.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.3,  # 降低温度，保证输出稳定
        )

    def generate_from_document(
        self,
        document_content: str,
        num_questions: int = 3,
        document_source: str = "",
    ) -> List[EvaluationSample]:
        """从单个文档生成评估样本

        Args:
            document_content: 文档内容
            num_questions: 生成问题数量
            document_source: 文档来源

        Returns:
            评估样本列表
        """
        logger.info(f"Generating {num_questions} questions from document: {document_source}")

        # 限制文档长度（避免超出LLM上下文）
        if len(document_content) > 3000:
            document_content = document_content[:3000] + "..."

        # 构建提示
        prompt = ChatPromptTemplate.from_template(self.QUESTION_GENERATION_PROMPT)
        messages = prompt.format_messages(
            num_questions=num_questions,
            document_content=document_content
        )

        try:
            # 调用LLM生成问题
            response = self.llm.invoke(messages)
            content = response.content

            # 提取JSON（可能包含markdown代码块）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # 解析JSON
            result = json.loads(content)

            # 构建评估样本
            samples = []
            for item in result.get("questions", []):
                sample = EvaluationSample(
                    question=item["question"],
                    ground_truth=item["answer"],
                    contexts=[document_content],  # 使用原始文档作为上下文
                    metadata={
                        "source": document_source,
                        "difficulty": item.get("difficulty", "medium"),
                        "category": item.get("category", "fact"),
                        "generated": True,
                    }
                )
                samples.append(sample)

            logger.info(f"Generated {len(samples)} samples from {document_source}")
            return samples

        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            return []

    def generate_from_directory(
        self,
        docs_dir: Path,
        num_questions_per_doc: int = 3,
        max_docs: int = 30,
    ) -> List[EvaluationSample]:
        """从文档目录批量生成评估样本

        Args:
            docs_dir: 文档目录路径
            num_questions_per_doc: 每个文档生成的问题数
            max_docs: 最多处理的文档数

        Returns:
            所有评估样本列表
        """
        all_samples = []

        # 查找markdown文档
        md_files = list(docs_dir.glob("**/*.md"))[:max_docs]
        logger.info(f"Found {len(md_files)} markdown files in {docs_dir}")

        for md_file in md_files:
            try:
                # 读取文档内容
                content = md_file.read_text(encoding="utf-8")

                # 生成样本
                samples = self.generate_from_document(
                    document_content=content,
                    num_questions=num_questions_per_doc,
                    document_source=str(md_file.relative_to(docs_dir))
                )

                all_samples.extend(samples)

            except Exception as e:
                logger.warning(f"Failed to process {md_file}: {e}")
                continue

        logger.info(f"Generated total {len(all_samples)} samples from {len(md_files)} documents")
        return all_samples

    def create_golden_dataset(self) -> List[EvaluationSample]:
        """创建黄金数据集（人工标注的核心场景）

        这些是手工精心设计的评估用例，覆盖关键场景。
        """
        golden_samples = [
            # 场景1: Kubernetes Pod OOMKilled
            EvaluationSample(
                question="Kubernetes Pod出现OOMKilled错误时应该如何排查？",
                ground_truth="1. 使用kubectl describe pod查看资源限制和实际使用情况；2. 检查内存limit是否设置过低；3. 使用kubectl top pod查看实时内存使用；4. 分析应用日志查找内存泄漏线索；5. 必要时调整memory limit或优化应用。",
                contexts=[
                    "当Kubernetes Pod出现OOMKilled错误时，表示容器因内存使用超过限制而被系统终止。常见原因包括：内存limit设置过低、应用存在内存泄漏、突发流量导致内存激增。排查方法：使用kubectl describe pod查看资源限制和实际使用情况。"
                ],
                metadata={
                    "source": "golden",
                    "difficulty": "medium",
                    "category": "troubleshooting",
                    "scenario": "k8s_oom"
                }
            ),

            # 场景2: Elasticsearch查询优化
            EvaluationSample(
                question="如何优化Elasticsearch的查询性能？",
                ground_truth="1. 使用filter代替query减少评分计算；2. 合理设置分片数量；3. 启用query cache；4. 避免深度分页；5. 使用routing减少搜索范围；6. 优化mapping减少字段数量。",
                contexts=[
                    "Elasticsearch查询优化技巧：使用filter context代替query context可以避免评分计算，提升性能。合理设置分片数量，避免过多或过少。启用query cache缓存常用查询。避免深度分页，使用scroll或search_after。"
                ],
                metadata={
                    "source": "golden",
                    "difficulty": "hard",
                    "category": "configuration",
                    "scenario": "es_optimization"
                }
            ),

            # 场景3: 日志查询基础
            EvaluationSample(
                question="如何查看最近100行的系统日志？",
                ground_truth="使用命令：tail -n 100 /var/log/syslog 或 journalctl -n 100",
                contexts=[
                    "查看系统日志的常用命令：tail -n 100 /var/log/syslog 显示最后100行日志。使用journalctl -n 100查看systemd日志。添加-f参数可以实时跟踪日志。"
                ],
                metadata={
                    "source": "golden",
                    "difficulty": "easy",
                    "category": "fact",
                    "scenario": "basic_command"
                }
            ),

            # 场景4: Docker容器重启
            EvaluationSample(
                question="Docker容器频繁重启应该如何排查？",
                ground_truth="1. 使用docker logs查看容器日志；2. 检查docker inspect查看退出码；3. 查看restart policy是否设置为always；4. 检查健康检查配置是否合理；5. 查看资源限制是否触发OOM。",
                contexts=[
                    "Docker容器频繁重启排查步骤：首先使用docker logs <container_id>查看容器日志，了解崩溃原因。使用docker inspect查看容器详细信息，包括退出码。检查restart policy设置。查看健康检查配置。"
                ],
                metadata={
                    "source": "golden",
                    "difficulty": "medium",
                    "category": "troubleshooting",
                    "scenario": "docker_restart"
                }
            ),

            # 场景5: 磁盘空间不足
            EvaluationSample(
                question="服务器磁盘空间不足时如何快速定位占用空间的文件？",
                ground_truth="使用命令：du -sh /* | sort -rh | head -10 查看根目录下各目录的空间占用，或使用 ncdu 交互式查看磁盘使用情况。",
                contexts=[
                    "快速定位磁盘空间占用：使用du -sh /*显示各目录大小，配合sort -rh排序，head -10显示前10个。或使用ncdu工具交互式浏览。常见占用大户：日志文件、临时文件、Docker镜像。"
                ],
                metadata={
                    "source": "golden",
                    "difficulty": "easy",
                    "category": "troubleshooting",
                    "scenario": "disk_space"
                }
            ),
        ]

        logger.info(f"Created golden dataset with {len(golden_samples)} samples")
        return golden_samples

    def save_dataset(self, samples: List[EvaluationSample], output_path: Path) -> None:
        """保存评估数据集到JSON文件

        Args:
            samples: 评估样本列表
            output_path: 输出文件路径
        """
        data = []
        for sample in samples:
            data.append({
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "contexts": sample.contexts,
                "metadata": sample.metadata,
            })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(samples)} samples to {output_path}")

    def load_dataset(self, input_path: Path) -> List[EvaluationSample]:
        """从JSON文件加载评估数据集

        Args:
            input_path: 输入文件路径

        Returns:
            评估样本列表
        """
        data = json.loads(input_path.read_text(encoding="utf-8"))

        samples = []
        for item in data:
            sample = EvaluationSample(
                question=item["question"],
                ground_truth=item["ground_truth"],
                contexts=item["contexts"],
                metadata=item.get("metadata", {})
            )
            samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from {input_path}")
        return samples


# 全局单例
evaluation_dataset_generator = EvaluationDatasetGenerator()
