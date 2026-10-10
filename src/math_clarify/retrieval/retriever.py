"""公式稀疏检索与双通道排名融合的独立实现。"""

from collections import Counter
from dataclasses import dataclass
import math

from math_clarify.retrieval.normalizer import extract_formula_tokens


@dataclass(frozen=True)
class FormulaDocument:
    """公式检索通道所需的最小文档结构。"""

    document_id: str
    content: str


def rank_formula_documents(
    query: str,
    documents: list[FormulaDocument],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    """用 BM25 对规范化公式 token 排名。

    返回所有文档，零分文档排在末尾，方便与 dense 全排名做 RRF。
    """
    if not documents:
        return []

    query_tokens = extract_formula_tokens(query)
    document_tokens = [extract_formula_tokens(doc.content) for doc in documents]
    lengths = [len(tokens) for tokens in document_tokens]
    average_length = sum(lengths) / len(lengths) if lengths else 1.0

    document_frequency: Counter[str] = Counter()
    for tokens in document_tokens:
        document_frequency.update(set(tokens))

    query_counts = Counter(query_tokens)
    ranked = []
    total_documents = len(documents)
    for order, (document, tokens) in enumerate(zip(documents, document_tokens)):
        term_counts = Counter(tokens)
        score = 0.0
        for token, query_count in query_counts.items():
            frequency = term_counts[token]
            if frequency == 0:
                continue
            df = document_frequency[token]
            inverse_document_frequency = math.log(
                1 + (total_documents - df + 0.5) / (df + 0.5)
            )
            length_factor = 1 - b + b * lengths[order] / max(average_length, 1.0)
            score += query_count * inverse_document_frequency * (
                frequency * (k1 + 1) / (frequency + k1 * length_factor)
            )
        ranked.append(
            {
                "document_id": document.document_id,
                "score": score,
                "original_order": order,
            }
        )

    ranked.sort(
        key=lambda item: (-item["score"], item["original_order"], item["document_id"])
    )
    return ranked


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    rank_constant: int = 60,
    weights: list[float] | None = None,
) -> list[dict]:
    """融合多个只含文档 ID 的排名，避免直接比较不同通道的分数尺度。"""
    if rank_constant < 1:
        raise ValueError("rank_constant 必须大于等于 1")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights 数量必须和 rankings 数量一致")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + weight / (
                rank_constant + rank
            )
            best_rank[document_id] = min(best_rank.get(document_id, rank), rank)

    ordered = sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id],
            best_rank[document_id],
            document_id,
        ),
    )
    return [
        {
            "document_id": document_id,
            "score": scores[document_id],
            "best_channel_rank": best_rank[document_id],
        }
        for document_id in ordered
    ]


def dual_channel_rank(
    query: str,
    documents: list[FormulaDocument],
    dense_ranking: list[str],
    *,
    top_k: int | None = None,
    dense_weight: float = 1.0,
    formula_weight: float = 1.0,
) -> list[dict]:
    """融合自然语言 dense 排名与规范化公式 BM25 排名。"""
    formula_results = rank_formula_documents(query, documents)
    # 零分文档不向融合器提供虚假的公式信号。
    formula_ranking = [
        item["document_id"] for item in formula_results if item["score"] > 0
    ]
    fused = reciprocal_rank_fusion(
        [dense_ranking, formula_ranking],
        weights=[dense_weight, formula_weight],
    )
    return fused if top_k is None else fused[:top_k]


