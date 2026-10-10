"""大小写敏感的轻量公式字符级 CRF 分词器。"""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import unicodedata
from typing import Any

import joblib
import sklearn_crfsuite


# 已知数学关键字使用最长匹配进行保护。CRF 负责其余变量边界和上下文角色。
_PROTECTED_LEXEMES = {
    # 向量与空间解析几何中的固定 LaTeX 命令直接走规则，避免让 CRF
    # 为含义确定的符号消耗训练样本。
    r"\overrightarrow": "VECTOR_MARKER",
    r"\operatorname": "FORMAT_COMMAND",
    r"\mathrm": "FORMAT_COMMAND",
    r"\varphi": "VARIABLE",
    r"\gamma": "VARIABLE",
    r"\cdot": "OPERATOR",
    r"\vec": "VECTOR_MARKER",
    r"\pi": "CONSTANT",
    r"\mu": "VARIABLE",
    r"\rightarrow": "ARROW",
    r"\Rightarrow": "RELATION",
    r"\partial": "DERIVATIVE",
    r"\epsilon": "VARIABLE",
    r"\infty": "CONSTANT",
    r"\lambda": "VARIABLE",
    r"\alpha": "VARIABLE",
    r"\delta": "VARIABLE",
    r"\theta": "VARIABLE",
    r"\arcsin": "FUNCTION",
    r"\arccos": "FUNCTION",
    r"\arctan": "FUNCTION",
    r"\sqrt": "ROOT",
    r"\frac": "FRACTION",
    r"\lim": "LIMIT",
    r"\int": "INTEGRAL",
    r"\sum": "SUM",
    r"\prod": "PRODUCT",
    r"\det": "FUNCTION",
    r"\sinh": "FUNCTION",
    r"\cosh": "FUNCTION",
    r"\tanh": "FUNCTION",
    r"\sin": "FUNCTION",
    r"\cos": "FUNCTION",
    r"\tan": "FUNCTION",
    r"\log": "FUNCTION",
    r"\ln": "FUNCTION",
    r"\exp": "FUNCTION",
    r"\to": "ARROW",
    r"\ne": "RELATION",
    "rightarrow": "ARROW",
    "infinity": "CONSTANT",
    "limsup": "LIMIT",
    "liminf": "LIMIT",
    "arcsin": "FUNCTION",
    "arccos": "FUNCTION",
    "arctan": "FUNCTION",
    "epsilon": "VARIABLE",
    "lambda": "VARIABLE",
    "alpha": "VARIABLE",
    "delta": "VARIABLE",
    "theta": "VARIABLE",
    "sqrt": "ROOT",
    "frac": "FRACTION",
    "sinh": "FUNCTION",
    "cosh": "FUNCTION",
    "tanh": "FUNCTION",
    "sin": "FUNCTION",
    "cos": "FUNCTION",
    "tan": "FUNCTION",
    "log": "FUNCTION",
    "lim": "LIMIT",
    "int": "INTEGRAL",
    "sum": "SUM",
    "prod": "PRODUCT",
    "det": "FUNCTION",
    "Prj": "FUNCTION",
    "ln": "FUNCTION",
    "exp": "FUNCTION",
    "->": "ARROW",
    "=>": "RELATION",
    "<=": "RELATION",
    ">=": "RELATION",
    "!=": "RELATION",
    "→": "ARROW",
    "⇒": "RELATION",
    "≠": "RELATION",
    "∞": "CONSTANT",
    "∫": "INTEGRAL",
    "Σ": "SUM",
    "Π": "PRODUCT",
    "{": "DELIMITER",
    "}": "DELIMITER",
    "(": "DELIMITER",
    ")": "DELIMITER",
    "[": "DELIMITER",
    "]": "DELIMITER",
    "|": "DELIMITER",
    ",": "DELIMITER",
    "_": "SCRIPT",
    "^": "SCRIPT",
    "=": "RELATION",
    "<": "RELATION",
    ">": "RELATION",
}


@dataclass(frozen=True)
class FormulaSegmentation:
    """模型对一条公式的离线诊断结果。"""

    formula: str
    tokens: list[dict[str, Any]]
    mean_confidence: float
    minimum_confidence: float
    needs_clarification: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "tokens": self.tokens,
            "mean_confidence": self.mean_confidence,
            "minimum_confidence": self.minimum_confidence,
            "needs_clarification": self.needs_clarification,
            "reasons": self.reasons,
        }


def _char_features(formula: str, index: int) -> dict[str, Any]:
    char = formula[index]
    features: dict[str, Any] = {
        "bias": 1.0,
        "char": char,
        "lower": char.lower(),
        "unicode_category": unicodedata.category(char),
        "is_upper": char.isupper(),
        "is_lower": char.islower(),
        "is_alpha": char.isalpha(),
        "is_digit": char.isdigit(),
        "is_space": char.isspace(),
        "is_backslash": char == "\\",
        "is_delimiter": char in "()[]{}|",
        "is_script": char in "_^",
        "is_relation_char": char in "=<>!→⇒",
        "is_operator_char": char in "+-*/×÷−",
    }
    for offset in (-3, -2, -1, 1, 2, 3):
        position = index + offset
        key = f"char[{offset:+d}]"
        features[key] = formula[position] if 0 <= position < len(formula) else "<PAD>"
    features["prev_bigram"] = formula[max(0, index - 1) : index + 1]
    features["next_bigram"] = formula[index : min(len(formula), index + 2)]
    features["window3"] = formula[max(0, index - 1) : min(len(formula), index + 2)]
    features["BOS"] = index == 0
    features["EOS"] = index == len(formula) - 1
    return features


def formula_to_features(formula: str) -> list[dict[str, Any]]:
    if not isinstance(formula, str):
        raise TypeError("formula 必须是字符串")
    if not formula:
        raise ValueError("formula 不能为空")
    return [_char_features(formula, index) for index in range(len(formula))]


def _protected_spans(formula: str) -> list[tuple[int, int, str]]:
    """按最长匹配找到不重叠的数学关键字。"""
    candidates: list[tuple[int, int, str]] = []
    for text, role in sorted(_PROTECTED_LEXEMES.items(), key=lambda item: -len(item[0])):
        start = 0
        while True:
            position = formula.find(text, start)
            if position < 0:
                break
            candidates.append((position, position + len(text), role))
            start = position + 1

    # 向量标记后的花括号内容具有明确的向量角色。例如
    # ``\overrightarrow{AB}`` 中 AB 是向量名，不应再按连续大写字母
    # 误报为歧义。这里只处理边界完整的简单名称；复杂下标仍交给 CRF。
    vector_name_spans: list[tuple[int, int]] = []
    for command in (r"\overrightarrow", r"\vec"):
        start = 0
        while True:
            command_start = formula.find(command, start)
            if command_start < 0:
                break
            content_start = command_start + len(command)
            if content_start < len(formula) and formula[content_start] == "{":
                content_end = formula.find("}", content_start + 1)
                if content_end > content_start + 1:
                    content = formula[content_start + 1 : content_end]
                    if all(char.isalnum() or char in "_'" for char in content):
                        span = (content_start + 1, content_end)
                        vector_name_spans.append(span)
                        candidates.append((*span, "VECTOR"))
            start = command_start + len(command)

    # 任意连续大写拉丁字母都可能是多字符变量、矩阵乘法或几何线段。
    # 这里保护整个跨度并标记歧义，不根据少数训练样本强行拆开。
    position = 0
    while position < len(formula):
        if "A" <= formula[position] <= "Z":
            end = position + 1
            while end < len(formula) and "A" <= formula[end] <= "Z":
                end += 1
            inside_vector_name = any(
                position >= start and end <= stop
                for start, stop in vector_name_spans
            )
            if end - position >= 2 and not inside_vector_name:
                candidates.append((position, end, "AMBIGUOUS"))
            position = end
        else:
            position += 1
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    occupied: set[int] = set()
    selected: list[tuple[int, int, str]] = []
    for start, end, role in candidates:
        if any(position in occupied for position in range(start, end)):
            continue
        selected.append((start, end, role))
        occupied.update(range(start, end))
    return selected


def _apply_protected_labels(formula: str, labels: list[str]) -> list[str]:
    protected = list(labels)
    spans = _protected_spans(formula)
    for start, end, role in spans:
        for position in range(start, end):
            prefix = "B" if position == start else "I"
            protected[position] = f"{prefix}-{role}"

    # 避免 CRF 把紧跟在 ∫/sin/lim 后面的变量粘进操作符整体。
    for _, end, role in spans:
        if (
            role in {"INTEGRAL", "FUNCTION", "LIMIT"}
            and end < len(formula)
            and formula[end].isalpha()
            and protected[end] == f"I-{role}"
        ):
            protected[end] = "B-VARIABLE"

    # 字母或右括号后的“-数字”是二元减法；箭头、关系符或开头后的负号
    # 仍可作为负数的一部分。这个上下文规则避免把 z-5 合成一个 NUMBER。
    for position, char in enumerate(formula[:-1]):
        if char != "-" or not formula[position + 1].isdigit():
            continue
        previous = position - 1
        while previous >= 0 and formula[previous].isspace():
            previous -= 1
        if previous >= 0 and (formula[previous].isalnum() or formula[previous] in ")]}"):
            protected[position] = "B-OPERATOR"
            if protected[position + 1].startswith("I-NUMBER"):
                protected[position + 1] = "B-NUMBER"
    return protected


def bio_labels_to_tokens(
    formula: str,
    labels: list[str],
    confidences: list[float] | None = None,
) -> list[dict[str, Any]]:
    """把字符 BIO 标签恢复为 token span。"""
    if len(formula) != len(labels):
        raise ValueError("formula 与 labels 长度必须一致")
    if confidences is None:
        confidences = [1.0] * len(formula)

    tokens: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for index, (char, label, confidence) in enumerate(zip(formula, labels, confidences)):
        if label == "O":
            if current is not None:
                current["confidence"] = sum(current.pop("_confidences")) / len(
                    current["text"]
                )
                tokens.append(current)
                current = None
            continue

        prefix, role = label.split("-", 1)
        starts_new = prefix == "B" or current is None or current["role"] != role
        if starts_new:
            if current is not None:
                current["confidence"] = sum(current.pop("_confidences")) / len(
                    current["text"]
                )
                tokens.append(current)
            current = {
                "text": char,
                "role": role,
                "start": index,
                "end": index + 1,
                "_confidences": [confidence],
            }
        else:
            current["text"] += char
            current["end"] = index + 1
            current["_confidences"].append(confidence)

    if current is not None:
        current["confidence"] = sum(current.pop("_confidences")) / len(current["text"])
        tokens.append(current)
    return tokens


class FormulaCRFSegmenter:
    """公式字符级 CRF；已知符号由最长匹配规则保护。"""

    def __init__(self) -> None:
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.08,
            c2=0.08,
            max_iterations=120,
            all_possible_transitions=True,
        )

    def fit(self, records: list[dict]) -> "FormulaCRFSegmenter":
        if not records:
            raise ValueError("训练数据不能为空")
        features = [formula_to_features(record["formula"]) for record in records]
        labels = [record["char_labels"] for record in records]
        self.model.fit(features, labels)
        return self

    def predict_labels(self, formula: str) -> tuple[list[str], list[float]]:
        features = formula_to_features(formula)
        labels = list(self.model.predict_single(features))
        marginals = self.model.predict_marginals_single(features)
        confidences = [float(marginal[label]) for marginal, label in zip(marginals, labels)]

        protected_labels = _apply_protected_labels(formula, labels)
        protected_positions = {
            position
            for start, end, _ in _protected_spans(formula)
            for position in range(start, end)
        }
        for index, (old, new) in enumerate(zip(labels, protected_labels)):
            # 固定词典命中属于确定性判断，不继承 CRF 的低置信度。
            if old != new or index in protected_positions:
                confidences[index] = 1.0
        return protected_labels, confidences

    def segment(self, formula: str, *, confidence_threshold: float = 0.72) -> FormulaSegmentation:
        labels, confidences = self.predict_labels(formula)
        tokens = bio_labels_to_tokens(formula, labels, confidences)
        token_confidences = [token["confidence"] for token in tokens]
        mean_confidence = (
            sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
        )
        minimum_confidence = min(token_confidences, default=0.0)

        reasons: list[str] = []
        if any(token["role"] == "AMBIGUOUS" for token in tokens):
            reasons.append("ambiguous_token")
        if minimum_confidence < confidence_threshold:
            reasons.append("low_token_confidence")
        if not tokens:
            reasons.append("no_math_token")
        return FormulaSegmentation(
            formula=formula,
            tokens=tokens,
            mean_confidence=mean_confidence,
            minimum_confidence=minimum_confidence,
            needs_clarification=bool(reasons),
            reasons=reasons,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "FormulaCRFSegmenter":
        # joblib files can execute code when loaded. Only load models you trust.
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("模型文件不是 FormulaCRFSegmenter")
        return model


def dependency_versions() -> dict[str, str]:
    return {
        "sklearn-crfsuite": version("sklearn-crfsuite"),
        "python-crfsuite": version("python-crfsuite"),
    }


