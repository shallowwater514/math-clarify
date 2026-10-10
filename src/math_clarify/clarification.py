"""把公式歧义转换成可直接回答的中文澄清问题。

该模块完全离线：规则只负责发现“仅凭公式不能唯一确定”的常见情况，
CRF 的低置信度和歧义 token 作为兜底信号。它不尝试替用户猜测数学含义。
"""

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class ClarificationQuestion:
    """一个带候选答案的中文澄清问题。"""

    code: str
    fragment: str
    prompt: str
    options: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "fragment": self.fragment,
            "prompt": self.prompt,
            "options": list(self.options),
        }


@dataclass(frozen=True)
class FormulaClarification:
    """一条公式需要向用户追问的结构化结果。"""

    formula: str
    needs_clarification: bool
    reason_codes: tuple[str, ...]
    questions: tuple[ClarificationQuestion, ...]
    message: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "needs_clarification": self.needs_clarification,
            "reason_codes": list(self.reason_codes),
            "questions": [question.to_dict() for question in self.questions],
            "message": self.message,
        }


def _question(
    code: str,
    fragment: str,
    prompt: str,
    *options: str,
) -> ClarificationQuestion:
    return ClarificationQuestion(code, fragment, prompt, tuple(options))


def _targeted_questions(formula: str) -> list[ClarificationQuestion]:
    """发现无需模型就能定位的数学歧义。"""
    questions: list[ClarificationQuestion] = []

    # 显式向量标记中的 AB 已经有明确角色，不再触发“连续大写字母”反问。
    vector_names = {
        match.group(1)
        for match in re.finditer(r"\\(?:vec|overrightarrow)\{([^{}]+)\}", formula)
    }
    for match in re.finditer(r"(?<![A-Za-z\\])([A-Z]{2,})(?![A-Za-z])", formula):
        fragment = match.group(1)
        if fragment in vector_names:
            continue
        questions.append(
            _question(
                "adjacent_uppercase",
                fragment,
                f"公式里的“{fragment}”具体表示什么？",
                "多个矩阵或变量相乘",
                "几何中的线段或有向线段名称",
                "一个不可拆分的完整变量名",
            )
        )

    for match in re.finditer(r"\|([^|]+)\|", formula):
        content = match.group(1)
        if r"\vec" in content or r"\overrightarrow" in content:
            continue
        if re.fullmatch(r"[a-zα-ω](?:_[A-Za-z0-9{}]+)?", content):
            continue
        questions.append(
            _question(
                "vertical_bar_meaning",
                match.group(0),
                f"“{match.group(0)}”中的竖线采用哪种含义？",
                "绝对值",
                "矩阵行列式",
                "向量的模",
                "集合的元素个数",
            )
        )

    point_call = re.search(
        r"(?<![A-Za-z])([A-Z](?:_\{?\w+\}?)?)\(([^()]*)\)", formula
    )
    if point_call and point_call.group(2).count(",") == 2:
        questions.append(
            _question(
                "point_or_function",
                point_call.group(0),
                f"“{point_call.group(0)}”表示空间点还是函数调用？",
                "三维空间中的点及其坐标",
                "三元函数在这些参数处的调用",
                "其他含义",
            )
        )

    if not re.search(r"\\(?:int|iint|iiint|oint)", formula):
        differential = re.search(r"(?<![A-Za-z\\])(d[A-Za-z])(?![A-Za-z])", formula)
        if differential:
            fragment = differential.group(1)
            questions.append(
                _question(
                    "differential_or_identifier",
                    fragment,
                    f"“{fragment}”在这里是微分整体，还是普通字母/变量组合？",
                    "微分符号",
                    "d与后面变量相乘",
                    "一个不可拆分的变量名",
                )
            )

    inverse_trig = re.search(r"\\(sin|cos|tan)\^\{-?1\}", formula)
    if inverse_trig:
        fragment = inverse_trig.group(0)
        questions.append(
            _question(
                "inverse_trig_or_reciprocal",
                fragment,
                f"“{fragment}”表示反三角函数还是三角函数的倒数？",
                "反三角函数",
                "三角函数值的倒数",
            )
        )

    inverse_function = re.search(r"(?<![A-Za-z])([fgh])\^\{-1\}\s*\(", formula)
    if inverse_function:
        fragment = inverse_function.group(1) + r"^{-1}"
        questions.append(
            _question(
                "inverse_function_or_reciprocal",
                fragment,
                f"“{fragment}”表示反函数还是函数值的倒数？",
                "反函数",
                "原函数值的倒数",
            )
        )

    matrix_inverse = re.search(r"(?<![A-Za-z])([A-Z])\^\{-1\}(?!\s*\()", formula)
    if matrix_inverse:
        fragment = matrix_inverse.group(0)
        questions.append(
            _question(
                "matrix_inverse_or_reciprocal",
                fragment,
                f"“{fragment}”中的大写字母表示矩阵还是普通数值？",
                "矩阵的逆矩阵",
                "普通变量的倒数",
                "其他对象的逆运算",
            )
        )

    implicit = re.search(r"(?<![A-Za-z\\])(xy|yx|uv|vu|ab|ba)(?![A-Za-z])", formula)
    if implicit:
        fragment = implicit.group(1)
        questions.append(
            _question(
                "implicit_product_or_identifier",
                fragment,
                f"“{fragment}”是两个变量相乘，还是一个完整变量名？",
                f"{fragment[0]}与{fragment[1]}相乘",
                "一个不可拆分的变量名",
            )
        )

    if re.search(r"\\overline\{[^{}]+\}", formula):
        fragment = re.search(r"\\overline\{[^{}]+\}", formula).group(0)  # type: ignore[union-attr]
        questions.append(
            _question(
                "overline_meaning",
                fragment,
                f"“{fragment}”上方横线采用哪种含义？",
                "线段长度或几何线段",
                "复数共轭",
                "集合的补集",
                "平均值或其他含义",
            )
        )

    bar = re.search(r"\\bar\s*\{?([A-Za-z])\}?", formula)
    if bar:
        fragment = bar.group(0)
        questions.append(
            _question(
                "bar_meaning",
                fragment,
                f"“{fragment}”上方横线表示复数共轭、平均值还是其他含义？",
                "复数共轭",
                "统计平均值",
                "几何线段或其他含义",
            )
        )

    times = re.search(r"\\times|×", formula)
    if times:
        questions.append(
            _question(
                "times_or_cross_product",
                times.group(0),
                f"“{times.group(0)}”表示普通乘法还是向量叉积？",
                "普通乘法",
                "向量叉积",
                "笛卡尔积或其他含义",
            )
        )

    delta_call = re.search(r"\\delta\s*\(", formula)
    if delta_call:
        questions.append(
            _question(
                "delta_function_meaning",
                r"\delta(·)",
                "这里的δ是狄拉克δ函数，还是名称为δ的普通函数？",
                "狄拉克δ函数",
                "普通函数或变量",
                "其他含义",
            )
        )

    if re.search(r"(?<![A-Za-z\\])e\s*\^", formula):
        questions.append(
            _question(
                "e_constant_or_variable",
                "e",
                "这里的e是自然常数，还是题目定义的普通变量？",
                "自然常数e",
                "普通变量e",
            )
        )

    if re.search(r"(?<![A-Za-z\\])i\s*\^\s*(?:\{?2\}?)", formula):
        questions.append(
            _question(
                "imaginary_unit_or_variable",
                "i",
                "这里的i是虚数单位，还是普通变量/下标？",
                "虚数单位i",
                "普通变量i",
                "序号或下标i",
            )
        )

    bracket = re.search(r"\[[^\[\]]+\]", formula)
    if bracket:
        questions.append(
            _question(
                "square_bracket_meaning",
                bracket.group(0),
                f"“{bracket.group(0)}”中的方括号表示什么？",
                "取整/高斯符号",
                "矩阵或向量",
                "等价类",
                "普通分组符号",
            )
        )

    angle = re.search(r"\\langle.+?\\rangle", formula)
    if angle:
        questions.append(
            _question(
                "angle_bracket_meaning",
                angle.group(0),
                "尖括号里的表达式表示内积还是有序对象？",
                "向量内积",
                "有序对或坐标",
                "其他含义",
            )
        )

    if re.search(r"(?<![\d])0\s*\^\s*(?:\{?0\}?)(?![\d])", formula):
        questions.append(
            _question(
                "zero_power_zero_context",
                "0^0",
                "这里的0的0次幂处于哪种数学语境？",
                "初等代数中的幂运算",
                "极限表达式",
                "组合数学或程序约定",
            )
        )

    sqrt_square = re.search(r"\\sqrt\{([A-Za-z])\^\{?2\}?\}\s*=\s*\1", formula)
    if sqrt_square:
        variable = sqrt_square.group(1)
        questions.append(
            _question(
                "sqrt_square_domain",
                sqrt_square.group(0),
                f"是否已知{variable}≥0？实数范围内通常有√({variable}²)=|{variable}|。",
                f"已知{variable}≥0",
                f"{variable}可取任意实数",
                "使用了其他数域或约定",
            )
        )

    chained_power = re.search(
        r"(?<![A-Za-z])([A-Za-z])\s*\^\s*([A-Za-z])\s*\^\s*([A-Za-z])",
        formula,
    )
    if chained_power:
        fragment = chained_power.group(0)
        questions.append(
            _question(
                "power_associativity",
                fragment,
                f"“{fragment}”应按哪种顺序计算幂？",
                "先计算右侧幂，即a^(b^c)一类",
                "先计算左侧幂，即(a^b)^c一类",
                "原式省略了括号，需要重新输入",
            )
        )

    compact_function = re.search(
        r"(?<![A-Za-z\\])(sin|cos|tan|log|ln)([A-Za-z])(?![A-Za-z])",
        formula,
    )
    if compact_function:
        fragment = compact_function.group(0)
        function_name, argument = compact_function.groups()
        questions.append(
            _question(
                "function_or_identifier",
                fragment,
                f"“{fragment}”表示{function_name}({argument})，还是一个完整变量名？",
                f"函数{function_name}作用于{argument}",
                "一个不可拆分的变量名",
            )
        )

    derivative_operator = re.search(r"(?<![A-Za-z\\])D_\{?([A-Za-z])\}?", formula)
    if derivative_operator:
        fragment = derivative_operator.group(0)
        questions.append(
            _question(
                "derivative_operator_or_identifier",
                fragment,
                f"“{fragment}”是对指定变量求导的算子，还是变量D的下标？",
                "微分/方向导数算子",
                "带下标的普通变量D",
            )
        )

    combination = re.search(r"(?<![A-Za-z\\])C_\{?\w+\}?\^\{?\w+\}?", formula)
    if combination:
        fragment = combination.group(0)
        questions.append(
            _question(
                "combination_or_indexed_variable",
                fragment,
                f"“{fragment}”是组合数记号，还是带上下标的变量C？",
                "组合数",
                "带上下标的变量",
            )
        )

    divided_product = re.search(
        r"(?<![A-Za-z])([A-Za-z])\s*/\s*([A-Za-z]{2})(?![A-Za-z])",
        formula,
    )
    if divided_product:
        fragment = divided_product.group(0)
        denominator = divided_product.group(2)
        questions.append(
            _question(
                "division_grouping",
                fragment,
                f"“{fragment}”的分母是整个{denominator}，还是只包含紧跟除号的第一个量？",
                f"分母是{denominator}整体",
                "按从左到右计算除法和乘法",
                "原式应补充括号",
            )
        )

    return questions


def _segmentation_state(segmentation: Any | None) -> tuple[list[dict[str, Any]], set[str]]:
    if segmentation is None:
        return [], set()
    if hasattr(segmentation, "tokens"):
        tokens = list(segmentation.tokens)
        reasons = set(segmentation.reasons)
        return tokens, reasons
    if isinstance(segmentation, dict):
        return list(segmentation.get("tokens", [])), set(segmentation.get("reasons", []))
    raise TypeError("segmentation 必须是 FormulaSegmentation、字典或 None")


def clarify_formula(
    formula: str,
    segmentation: Any | None = None,
    *,
    max_questions: int = 3,
) -> FormulaClarification:
    """生成最多 ``max_questions`` 个具体中文反问。"""
    if not isinstance(formula, str):
        raise TypeError("formula 必须是字符串")
    if not formula.strip():
        raise ValueError("formula 不能为空")
    if max_questions < 1:
        raise ValueError("max_questions 必须至少为1")

    questions = _targeted_questions(formula)
    tokens, segmentation_reasons = _segmentation_state(segmentation)
    covered_fragments = {question.fragment for question in questions}

    for token in tokens:
        if token.get("role") != "AMBIGUOUS" or token.get("text") in covered_fragments:
            continue
        fragment = str(token.get("text", ""))
        questions.append(
            _question(
                "ambiguous_token",
                fragment,
                f"机器无法唯一判断“{fragment}”的数学角色，它在本题中表示什么？",
                "多个符号的组合",
                "一个完整变量或函数名",
                "几何对象",
                "其他含义",
            )
        )

    if "low_token_confidence" in segmentation_reasons and not questions:
        questions.append(
            _question(
                "low_token_confidence",
                formula,
                "机器对这条公式的符号边界或角色不够确定，请选择最接近的情况。",
                "公式输入正确，请结合题目文字继续判断",
                "公式可能有漏字、错字或括号错误",
                "我可以补充这条公式的题目背景",
            )
        )

    unique: list[ClarificationQuestion] = []
    seen: set[tuple[str, str]] = set()
    for question in questions:
        key = (question.code, question.fragment)
        if key in seen:
            continue
        seen.add(key)
        unique.append(question)

    selected = tuple(unique[:max_questions])
    if selected:
        lines = ["为了避免误解这条公式，请确认："]
        for index, question in enumerate(selected, start=1):
            options = "；".join(
                f"{option_index}.{option}"
                for option_index, option in enumerate(question.options, start=1)
            )
            lines.append(f"{index}. {question.prompt}（{options}）")
        message = "\n".join(lines)
    else:
        message = None

    return FormulaClarification(
        formula=formula,
        needs_clarification=bool(selected),
        reason_codes=tuple(question.code for question in selected),
        questions=selected,
        message=message,
    )


