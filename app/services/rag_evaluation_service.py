"""
RAG评估服务

基于RAGAs框架实现三层评估体系：
1. 检索层：Context Precision, Context Recall, MRR
2. 生成层：Faithfulness, Answer Relevancy
3. 系统层：Response Time, Token Cost

评估标准（业界阈值）：
- Context Precision: > 0.7 (优秀 > 0.85)
- Context Recall: > 0.8 (优秀 > 0.9)
- Faithfulness: > 0.8 (优秀 > 0.9)
- Answer Relevancy: > 0.75 (优秀 > 0.85)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import asyncio

from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

from app.services.evaluation_dataset_generator import EvaluationSample
from app.services.hybrid_retrieval_service import hybrid_retrieval_service


@dataclass
class RetrievalMetrics:
    """检索层指标"""

    context_precision: float  # 检索精度（检索的文档中有多少是相关的）
    context_recall: float  # 检索召回（相关文档中有多少被检索到）
    mrr: float  # Mean Reciprocal Rank（第一个相关文档的排名倒数）
    avg_retrieved_docs: float  # 平均检索文档数
    avg_retrieval_time_ms: float  # 平均检索耗时


@dataclass
class GenerationMetrics:
    """生成层指标"""

    faithfulness: float  # 忠实度（生成内容是否忠实于检索文档）
    answer_relevancy: float  # 答案相关性（答案与问题的相关程度）
    avg_answer_length: float  # 平均答案长度
    avg_generation_time_ms: float  # 平均生成耗时


@dataclass
class SystemMetrics:
    """系统层指标"""

    avg_total_time_ms: float  # 平均总耗时（检索+生成）
    avg_token_cost: float  # 平均Token消耗
    success_rate: float  # 成功率
    error_count: int  # 错误数量


@dataclass
class EvaluationReport:
    """完整评估报告"""

    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    system_metrics: SystemMetrics
    total_samples: int
    passed_samples: int
    overall_score: float  # 综合评分（0-100）
    recommendations: List[str]  # 改进建议


class RAGEvaluationService:
    """RAG评估服务

    实现三层评估体系，提供量化指标和改进建议。
    """

    # 业界标准阈值
    THRESHOLDS = {
        "context_precision": {
            "good": 0.85,
            "acceptable": 0.70,
        },
        "context_recall": {
            "good": 0.90,
            "acceptable": 0.80,
        },
        "faithfulness": {
            "good": 0.90,
            "acceptable": 0.80,
        },
        "answer_relevancy": {
            "good": 0.85,
            "acceptable": 0.75,
        },
        "response_time_ms": {
            "good": 2000,
            "acceptable": 3000,
        },
    }

    def __init__(self):
        logger.info("RAGEvaluationService initialized")

    async def evaluate_dataset(
        self,
        samples: List[EvaluationSample],
        rag_pipeline_func=None,
    ) -> EvaluationReport:
        """评估完整数据集

        Args:
            samples: 评估样本列表
            rag_pipeline_func: RAG Pipeline函数（question -> answer）
                              如果为None，则只评估检索层

        Returns:
            评估报告
        """
        logger.info(f"Starting evaluation on {len(samples)} samples")

        # 准备RAGAs数据集格式
        questions = []
        ground_truths = []
        contexts_list = []
        answers = []

        retrieval_times = []
        generation_times = []
        total_times = []
        token_costs = []
        error_count = 0

        for i, sample in enumerate(samples, 1):
            logger.info(f"Evaluating sample {i}/{len(samples)}: {sample.question[:50]}...")

            try:
                # 1. 检索阶段
                start_time = time.time()
                retrieved_docs = hybrid_retrieval_service.search(sample.question)
                retrieval_time = (time.time() - start_time) * 1000
                retrieval_times.append(retrieval_time)

                # 提取检索内容
                retrieved_contexts = [doc.page_content for doc in retrieved_docs]

                # 2. 生成阶段（如果提供了Pipeline函数）
                if rag_pipeline_func:
                    start_time = time.time()
                    answer = await rag_pipeline_func(sample.question, retrieved_docs)
                    generation_time = (time.time() - start_time) * 1000
                    generation_times.append(generation_time)
                    total_times.append(retrieval_time + generation_time)

                    # 估算Token消耗（简化计算）
                    token_cost = len(sample.question) + sum(len(ctx) for ctx in retrieved_contexts[:3]) + len(answer)
                    token_costs.append(token_cost / 4)  # 粗略估算

                    answers.append(answer)
                else:
                    # 只评估检索层，使用ground truth作为答案
                    answers.append(sample.ground_truth)
                    generation_times.append(0)
                    total_times.append(retrieval_time)
                    token_costs.append(0)

                # 收集数据
                questions.append(sample.question)
                ground_truths.append(sample.ground_truth)
                contexts_list.append(retrieved_contexts)

            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                error_count += 1
                # 添加占位数据以保持索引一致
                questions.append(sample.question)
                ground_truths.append(sample.ground_truth)
                contexts_list.append([])
                answers.append("")
                retrieval_times.append(0)
                generation_times.append(0)
                total_times.append(0)
                token_costs.append(0)

        # 构建RAGAs数据集
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(dataset_dict)

        # 使用RAGAs评估
        logger.info("Running RAGAs metrics evaluation...")
        try:
            ragas_result = evaluate(
                dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy,
                ]
            )

            # 提取指标
            retrieval_metrics = RetrievalMetrics(
                context_precision=ragas_result["context_precision"],
                context_recall=ragas_result["context_recall"],
                mrr=self._calculate_mrr(contexts_list, ground_truths),
                avg_retrieved_docs=sum(len(ctx) for ctx in contexts_list) / len(contexts_list),
                avg_retrieval_time_ms=sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
            )

            generation_metrics = GenerationMetrics(
                faithfulness=ragas_result["faithfulness"],
                answer_relevancy=ragas_result["answer_relevancy"],
                avg_answer_length=sum(len(ans) for ans in answers) / len(answers),
                avg_generation_time_ms=sum(generation_times) / len(generation_times) if generation_times else 0,
            )

            system_metrics = SystemMetrics(
                avg_total_time_ms=sum(total_times) / len(total_times) if total_times else 0,
                avg_token_cost=sum(token_costs) / len(token_costs) if token_costs else 0,
                success_rate=(len(samples) - error_count) / len(samples),
                error_count=error_count,
            )

        except Exception as e:
            logger.error(f"RAGAs evaluation failed: {e}")
            # 返回默认值
            retrieval_metrics = RetrievalMetrics(
                context_precision=0.0,
                context_recall=0.0,
                mrr=0.0,
                avg_retrieved_docs=0.0,
                avg_retrieval_time_ms=0.0,
            )
            generation_metrics = GenerationMetrics(
                faithfulness=0.0,
                answer_relevancy=0.0,
                avg_answer_length=0.0,
                avg_generation_time_ms=0.0,
            )
            system_metrics = SystemMetrics(
                avg_total_time_ms=0.0,
                avg_token_cost=0.0,
                success_rate=0.0,
                error_count=len(samples),
            )

        # 计算综合评分和改进建议
        overall_score = self._calculate_overall_score(
            retrieval_metrics,
            generation_metrics,
            system_metrics
        )

        recommendations = self._generate_recommendations(
            retrieval_metrics,
            generation_metrics,
            system_metrics
        )

        # 生成报告
        report = EvaluationReport(
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            system_metrics=system_metrics,
            total_samples=len(samples),
            passed_samples=len(samples) - error_count,
            overall_score=overall_score,
            recommendations=recommendations,
        )

        logger.info(f"Evaluation complete. Overall score: {overall_score:.1f}/100")
        return report

    def _calculate_mrr(
        self,
        contexts_list: List[List[str]],
        ground_truths: List[str]
    ) -> float:
        """计算Mean Reciprocal Rank

        MRR衡量第一个相关文档的排名：
        - 第1个相关：1/1 = 1.0
        - 第2个相关：1/2 = 0.5
        - 第3个相关：1/3 = 0.33
        """
        reciprocal_ranks = []

        for contexts, ground_truth in zip(contexts_list, ground_truths):
            # 简化：检查ground truth是否出现在contexts中
            for rank, context in enumerate(contexts, start=1):
                if self._is_relevant(context, ground_truth):
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                # 没有相关文档
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    def _is_relevant(self, context: str, ground_truth: str) -> bool:
        """简单判断context是否与ground_truth相关

        实际应用中可以使用更复杂的相似度计算
        """
        # 简化：检查关键词重叠
        context_words = set(context.lower().split())
        truth_words = set(ground_truth.lower().split())
        overlap = len(context_words & truth_words) / max(len(truth_words), 1)
        return overlap > 0.3

    def _calculate_overall_score(
        self,
        retrieval: RetrievalMetrics,
        generation: GenerationMetrics,
        system: SystemMetrics,
    ) -> float:
        """计算综合评分（0-100）

        权重分配：
        - 检索层：40%（Precision 20% + Recall 20%）
        - 生成层：40%（Faithfulness 20% + Relevancy 20%）
        - 系统层：20%（Success Rate 10% + Response Time 10%）
        """
        # 检索层得分
        retrieval_score = (
            retrieval.context_precision * 20 +
            retrieval.context_recall * 20
        )

        # 生成层得分
        generation_score = (
            generation.faithfulness * 20 +
            generation.answer_relevancy * 20
        )

        # 系统层得分
        # Response Time: < 2s得满分，> 3s得0分
        time_score = max(0, min(1, (3000 - system.avg_total_time_ms) / 1000)) * 10
        system_score = system.success_rate * 10 + time_score

        # 综合得分
        overall = retrieval_score + generation_score + system_score
        return min(100, max(0, overall))

    def _generate_recommendations(
        self,
        retrieval: RetrievalMetrics,
        generation: GenerationMetrics,
        system: SystemMetrics,
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 检索层建议
        if retrieval.context_precision < self.THRESHOLDS["context_precision"]["acceptable"]:
            recommendations.append(
                f"[检索精度] Context Precision={retrieval.context_precision:.2f} 偏低，"
                f"建议：1) 优化Query Rewrite策略；2) 调整RRF权重；3) 提升文档质量"
            )

        if retrieval.context_recall < self.THRESHOLDS["context_recall"]["acceptable"]:
            recommendations.append(
                f"[检索召回] Context Recall={retrieval.context_recall:.2f} 偏低，"
                f"建议：1) 增加检索top_k；2) 优化chunk切分策略；3) 检查索引覆盖率"
            )

        # 生成层建议
        if generation.faithfulness < self.THRESHOLDS["faithfulness"]["acceptable"]:
            recommendations.append(
                f"[生成忠实度] Faithfulness={generation.faithfulness:.2f} 偏低，"
                f"建议：1) 优化系统提示词；2) 降低LLM temperature；3) 加强上下文约束"
            )

        if generation.answer_relevancy < self.THRESHOLDS["answer_relevancy"]["acceptable"]:
            recommendations.append(
                f"[答案相关性] Answer Relevancy={generation.answer_relevancy:.2f} 偏低，"
                f"建议：1) 优化检索相关性；2) 改进问题理解；3) 调整生成策略"
            )

        # 系统层建议
        if system.avg_total_time_ms > self.THRESHOLDS["response_time_ms"]["acceptable"]:
            recommendations.append(
                f"[响应时间] 平均{system.avg_total_time_ms:.0f}ms 超过阈值，"
                f"建议：1) 优化索引算法；2) 减少检索top_k；3) 使用缓存机制"
            )

        if system.success_rate < 0.95:
            recommendations.append(
                f"[系统稳定性] 成功率{system.success_rate:.1%} 偏低（{system.error_count}次错误），"
                f"建议：1) 检查异常日志；2) 加强错误处理；3) 增加重试机制"
            )

        # 如果所有指标都良好
        if not recommendations:
            recommendations.append("✅ 所有指标均达到优秀水平，系统运行良好！")

        return recommendations

    def print_report(self, report: EvaluationReport) -> None:
        """打印评估报告"""
        print("\n" + "=" * 80)
        print("RAG系统评估报告")
        print("=" * 80)
        print()

        print(f"[总览]")
        print(f"  样本总数: {report.total_samples}")
        print(f"  成功样本: {report.passed_samples}")
        print(f"  综合评分: {report.overall_score:.1f}/100")
        print()

        print("=" * 80)
        print("[检索层指标]")
        print("=" * 80)
        r = report.retrieval_metrics
        self._print_metric("Context Precision", r.context_precision, self.THRESHOLDS["context_precision"])
        self._print_metric("Context Recall", r.context_recall, self.THRESHOLDS["context_recall"])
        print(f"  MRR (Mean Reciprocal Rank): {r.mrr:.3f}")
        print(f"  平均检索文档数: {r.avg_retrieved_docs:.1f}")
        print(f"  平均检索耗时: {r.avg_retrieval_time_ms:.1f} ms")
        print()

        print("=" * 80)
        print("[生成层指标]")
        print("=" * 80)
        g = report.generation_metrics
        self._print_metric("Faithfulness", g.faithfulness, self.THRESHOLDS["faithfulness"])
        self._print_metric("Answer Relevancy", g.answer_relevancy, self.THRESHOLDS["answer_relevancy"])
        print(f"  平均答案长度: {g.avg_answer_length:.0f} 字符")
        print(f"  平均生成耗时: {g.avg_generation_time_ms:.1f} ms")
        print()

        print("=" * 80)
        print("[系统层指标]")
        print("=" * 80)
        s = report.system_metrics
        print(f"  平均总耗时: {s.avg_total_time_ms:.1f} ms")
        print(f"  平均Token消耗: {s.avg_token_cost:.0f}")
        print(f"  成功率: {s.success_rate:.1%}")
        print(f"  错误数量: {s.error_count}")
        print()

        print("=" * 80)
        print("[改进建议]")
        print("=" * 80)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
        print()

        print("=" * 80)

    def _print_metric(self, name: str, value: float, thresholds: Dict[str, float]) -> None:
        """打印单个指标"""
        if value >= thresholds["good"]:
            status = "[优秀]"
        elif value >= thresholds["acceptable"]:
            status = "[合格]"
        else:
            status = "[需改进]"

        print(f"  {name}: {value:.3f} {status}")


# 全局单例
rag_evaluation_service = RAGEvaluationService()
