"""不调用大模型的可复现公式自然语言扩写原型。"""

import re

from math_clarify.retrieval.normalizer import normalize_formula


def expand_formula_query(text: str) -> str:
    """根据公式结构追加自然语言检索关键词。

    规则扩写的覆盖范围有意保持很小。真实产品可以把人工/LLM 解释通过同一
    接口传入，但离线论文实验应先保留这个无随机性基线。
    """
    normalized = normalize_formula(text)
    hints: list[str] = []

    if "lim" in normalized:
        hints.extend(["极限", "趋近"])
    if "infinity" in normalized:
        hints.extend(["趋向无穷", "自变量不断增大"])
    if "alpha" in normalized:
        hints.extend(["alpha 表示无穷小量", "无穷小趋近于零"])
    if "epsilon" in normalized and "delta" in normalized:
        hints.extend(["函数极限的 epsilon-delta 定义", "任意精度与去心邻域"])
    if "det(" in normalized or normalized.startswith("det"):
        hints.append("矩阵的行列式")
    if "matrix[" in normalized:
        hints.extend(["矩阵", "按行和列排列的元素"])
    if re.search(r"\bab\s*!=\s*ba\b", normalized):
        hints.extend(["矩阵乘法", "交换律"])
    if "lambda" in normalized and "det" in normalized:
        hints.extend(["特征方程", "矩阵的特征值"])
    if "sin" in normalized and "/" in normalized and "lim" in normalized:
        hints.extend(["三角函数", "重要极限"])
    if "^" in normalized:
        hints.append("幂")

    # 按首次出现顺序去重，避免重复关键词稀释原问题。
    unique_hints = list(dict.fromkeys(hints))
    if not unique_hints:
        return f"原问题：{text}。规范化公式：{normalized}。"
    return (
        f"原问题：{text}。规范化公式：{normalized}。"
        f"数学含义关键词：{'；'.join(unique_hints)}。"
    )


