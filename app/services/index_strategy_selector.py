"""
向量索引策略自适应选择器

根据数据规模和性能需求，自动选择最优的Milvus索引类型和参数。

索引类型对比：
- FLAT: 暴力搜索，100%精度，适合小规模（<1万）
- IVF_FLAT: 倒排索引，95-98%精度，适合中规模（1万-50万）
- HNSW: 图索引，97-99%精度，适合高性能场景（1万-100万）
- IVF_PQ: 压缩索引，90-95%精度，适合大规模（50万-1000万）
"""

from dataclasses import dataclass
from typing import Dict, Any, Literal
from loguru import logger


IndexType = Literal["FLAT", "IVF_FLAT", "HNSW", "IVF_PQ"]
MetricType = Literal["L2", "IP", "COSINE"]


@dataclass
class IndexRecommendation:
    """索引推荐结果"""

    index_type: IndexType
    metric_type: MetricType
    params: Dict[str, Any]
    reason: str
    estimated_memory_mb: float
    estimated_query_time_ms: float
    expected_recall: float


class IndexStrategySelector:
    """向量索引策略选择器

    根据数据规模自动选择最优索引类型和参数。
    """

    # 数据规模阈值
    SMALL_SCALE = 10_000       # < 1万：小规模
    MEDIUM_SCALE = 500_000     # 1万-50万：中规模
    LARGE_SCALE = 5_000_000    # 50万-500万：大规模

    # 向量维度（根据embedding模型）
    VECTOR_DIM = 1024

    # 每个向量的字节数（float32）
    BYTES_PER_VECTOR = VECTOR_DIM * 4

    def __init__(self):
        logger.info("IndexStrategySelector initialized")

    def select_index(
        self,
        vector_count: int,
        preferred_index_type: str = "auto",
        metric_type: MetricType = "L2",
    ) -> IndexRecommendation:
        """根据数据规模选择最优索引

        Args:
            vector_count: 向量数量
            preferred_index_type: 偏好的索引类型（"auto"为自动选择）
            metric_type: 距离度量类型

        Returns:
            IndexRecommendation: 索引推荐结果
        """
        # 如果指定了非auto类型，直接使用
        if preferred_index_type != "auto":
            return self._build_recommendation_for_type(
                index_type=preferred_index_type,  # type: ignore[arg-type]
                vector_count=vector_count,
                metric_type=metric_type,
            )

        # 自动选择策略
        if vector_count < self.SMALL_SCALE:
            return self._recommend_flat(vector_count, metric_type)
        elif vector_count < self.MEDIUM_SCALE:
            return self._recommend_ivf_flat(vector_count, metric_type)
        elif vector_count < self.LARGE_SCALE:
            return self._recommend_hnsw(vector_count, metric_type)
        else:
            return self._recommend_ivf_pq(vector_count, metric_type)

    def _recommend_flat(
        self,
        vector_count: int,
        metric_type: MetricType,
    ) -> IndexRecommendation:
        """推荐FLAT索引（暴力搜索）

        优点：
        - 100%召回率（精确搜索）
        - 无需训练，即建即用
        - 实现简单，无超参数

        缺点：
        - O(n)查询复杂度，数据量大时慢
        - 内存占用随数据量线性增长
        """
        memory_mb = (vector_count * self.BYTES_PER_VECTOR) / (1024 * 1024)

        # FLAT查询时间估算：每1000个向量约1ms
        query_time_ms = vector_count / 1000.0

        return IndexRecommendation(
            index_type="FLAT",
            metric_type=metric_type,
            params={},  # FLAT无需参数
            reason=f"数据规模小（{vector_count:,}向量），使用FLAT精确搜索",
            estimated_memory_mb=memory_mb,
            estimated_query_time_ms=query_time_ms,
            expected_recall=1.0,  # 100%召回
        )

    def _recommend_ivf_flat(
        self,
        vector_count: int,
        metric_type: MetricType,
    ) -> IndexRecommendation:
        """推荐IVF_FLAT索引（倒排文件索引）

        原理：
        - 使用k-means将向量聚类成nlist个簇
        - 查询时只搜索nprobe个最近的簇
        - 减少搜索空间，提升速度

        优点：
        - 平衡速度和精度（95-98%召回）
        - 内存占用适中
        - 参数调优空间大

        缺点：
        - 需要训练（k-means聚类）
        - nlist设置影响性能
        """
        # nlist推荐值：sqrt(N) 到 4*sqrt(N)
        # 经验值：每个簇100-1000个向量
        optimal_nlist = max(128, min(16384, int((vector_count / 500) ** 0.5) * 64))

        # 向下取整到2的幂次，便于分布式计算
        nlist = 2 ** int(optimal_nlist.bit_length() - 1)

        # nprobe推荐值：nlist的5-10%
        nprobe = max(8, min(256, nlist // 16))

        memory_mb = (vector_count * self.BYTES_PER_VECTOR) / (1024 * 1024)

        # IVF_FLAT查询时间估算：搜索nprobe个簇
        vectors_per_cluster = vector_count / nlist
        query_time_ms = (nprobe * vectors_per_cluster) / 1000.0

        return IndexRecommendation(
            index_type="IVF_FLAT",
            metric_type=metric_type,
            params={
                "nlist": nlist,  # 聚类中心数
                "nprobe": nprobe,  # 查询时搜索的簇数
            },
            reason=(
                f"数据规模中等（{vector_count:,}向量），"
                f"使用IVF_FLAT平衡速度和精度（nlist={nlist}, nprobe={nprobe}）"
            ),
            estimated_memory_mb=memory_mb,
            estimated_query_time_ms=query_time_ms,
            expected_recall=0.97,  # 97%召回
        )

    def _recommend_hnsw(
        self,
        vector_count: int,
        metric_type: MetricType,
    ) -> IndexRecommendation:
        """推荐HNSW索引（分层导航小世界图）

        原理：
        - 构建多层图结构，每层稀疏程度递增
        - 查询时从顶层快速定位，底层精细搜索
        - 近似最近邻搜索的SOTA算法

        优点：
        - 查询速度极快（对数复杂度）
        - 高召回率（97-99%）
        - 适合实时查询场景

        缺点：
        - 内存占用较高（1.5-2倍向量数据）
        - 构建时间长
        - 不支持增量更新（需重建）
        """
        # M: 每个节点的最大连接数（推荐16-64）
        # 数据量越大，M可以适当增大
        if vector_count < 100_000:
            M = 16
        elif vector_count < 500_000:
            M = 32
        else:
            M = 48

        # efConstruction: 构建时的搜索深度（推荐100-500）
        # 越大构建时间越长，但索引质量越好
        efConstruction = 200

        # ef: 查询时的搜索深度（运行时可调整）
        ef = 64

        # HNSW内存占用约为原始数据的1.5-2倍
        base_memory_mb = (vector_count * self.BYTES_PER_VECTOR) / (1024 * 1024)
        memory_mb = base_memory_mb * 1.8

        # HNSW查询时间估算：对数复杂度
        import math
        query_time_ms = math.log(vector_count, 2) * 0.1

        return IndexRecommendation(
            index_type="HNSW",
            metric_type=metric_type,
            params={
                "M": M,  # 每个节点的最大连接数
                "efConstruction": efConstruction,  # 构建时的搜索深度
                "ef": ef,  # 查询时的搜索深度
            },
            reason=(
                f"数据规模较大（{vector_count:,}向量），"
                f"使用HNSW实现高速查询（M={M}, ef={ef}）"
            ),
            estimated_memory_mb=memory_mb,
            estimated_query_time_ms=query_time_ms,
            expected_recall=0.98,  # 98%召回
        )

    def _recommend_ivf_pq(
        self,
        vector_count: int,
        metric_type: MetricType,
    ) -> IndexRecommendation:
        """推荐IVF_PQ索引（倒排文件+乘积量化）

        原理：
        - IVF: 先聚类减少搜索空间
        - PQ: 将向量切分成子向量，分别量化压缩
        - 大幅降低内存占用（5-10倍压缩）

        优点：
        - 极低内存占用（20-30%原始数据）
        - 支持超大规模数据
        - 查询速度快

        缺点：
        - 召回率较低（90-95%）
        - 参数调优复杂
        - 训练时间长
        """
        # nlist推荐值：大规模数据使用更多簇
        optimal_nlist = max(1024, min(65536, int((vector_count / 1000) ** 0.5) * 128))
        nlist = 2 ** int(optimal_nlist.bit_length() - 1)

        nprobe = max(16, min(512, nlist // 32))

        # m: 子向量数量（必须能被向量维度整除）
        # 推荐：维度 / m ∈ [4, 64]
        # 1024维 -> m=16 (每个子向量64维)
        m = 16

        # nbits: 每个子向量的量化位数（推荐8）
        nbits = 8

        # PQ压缩后内存占用约为原始的20-30%
        base_memory_mb = (vector_count * self.BYTES_PER_VECTOR) / (1024 * 1024)
        memory_mb = base_memory_mb * 0.25

        # IVF_PQ查询时间类似IVF_FLAT
        vectors_per_cluster = vector_count / nlist
        query_time_ms = (nprobe * vectors_per_cluster) / 800.0  # PQ稍快

        return IndexRecommendation(
            index_type="IVF_PQ",
            metric_type=metric_type,
            params={
                "nlist": nlist,
                "nprobe": nprobe,
                "m": m,  # 子向量数量
                "nbits": nbits,  # 量化位数
            },
            reason=(
                f"数据规模大（{vector_count:,}向量），"
                f"使用IVF_PQ压缩存储（nlist={nlist}, m={m}）"
            ),
            estimated_memory_mb=memory_mb,
            estimated_query_time_ms=query_time_ms,
            expected_recall=0.92,  # 92%召回
        )

    def _build_recommendation_for_type(
        self,
        index_type: IndexType,
        vector_count: int,
        metric_type: MetricType,
    ) -> IndexRecommendation:
        """为指定索引类型构建推荐"""
        if index_type == "FLAT":
            return self._recommend_flat(vector_count, metric_type)
        elif index_type == "IVF_FLAT":
            return self._recommend_ivf_flat(vector_count, metric_type)
        elif index_type == "HNSW":
            return self._recommend_hnsw(vector_count, metric_type)
        elif index_type == "IVF_PQ":
            return self._recommend_ivf_pq(vector_count, metric_type)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def get_search_params(self, recommendation: IndexRecommendation) -> Dict[str, Any]:
        """获取查询时的搜索参数

        Args:
            recommendation: 索引推荐结果

        Returns:
            搜索参数字典
        """
        if recommendation.index_type == "FLAT":
            return {}

        elif recommendation.index_type == "IVF_FLAT":
            return {
                "nprobe": recommendation.params.get("nprobe", 16)
            }

        elif recommendation.index_type == "HNSW":
            return {
                "ef": recommendation.params.get("ef", 64)
            }

        elif recommendation.index_type == "IVF_PQ":
            return {
                "nprobe": recommendation.params.get("nprobe", 16)
            }

        return {}


# 全局单例
index_strategy_selector = IndexStrategySelector()
