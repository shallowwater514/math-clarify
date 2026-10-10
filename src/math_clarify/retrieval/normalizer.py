"""最小公式规范化器。

目标不是完整解析 LaTeX，而是先统一教材检索中最常见的等价写法。
所有规则都是确定性的，方便做消融实验和定位误匹配。
"""

import re
import unicodedata


_UNICODE_REPLACEMENTS = {
    "→": "->",
    "⟶": "->",
    "⇒": "=>",
    "∞": "infinity",
    "α": "alpha",
    "Α": "alpha",
    "β": "beta",
    "Β": "beta",
    "ε": "epsilon",
    "ϵ": "epsilon",
    "Ε": "epsilon",
    "δ": "delta",
    "Δ": "delta",
    "λ": "lambda",
    "Λ": "lambda",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "≈": "~=",
    "×": "*",
    "·": "*",
    "−": "-",
    "–": "-",
    "—": "-",
    "²": "^2",
    "³": "^3",
}

_LATEX_COMMANDS = {
    "\\longrightarrow": "->",
    "\\rightarrow": "->",
    "\\Rightarrow": "=>",
    "\\to": "->",
    "\\infty": "infinity",
    "\\varepsilon": "epsilon",
    "\\epsilon": "epsilon",
    "\\alpha": "alpha",
    "\\beta": "beta",
    "\\delta": "delta",
    "\\lambda": "lambda",
    "\\leq": "<=",
    "\\le": "<=",
    "\\geq": ">=",
    "\\ge": ">=",
    "\\neq": "!=",
    "\\ne": "!=",
    "\\times": "*",
    "\\cdot": "*",
    "\\sim": "~",
    "\\lim": "lim",
    "\\det": "det",
    "\\sin": "sin",
    "\\cos": "cos",
    "\\tan": "tan",
    "\\ln": "ln",
}

_TOKEN_PATTERN = re.compile(
    r"infinity|epsilon|delta|alpha|beta|lambda|"
    r"[a-z]+(?:_[a-z0-9]+)?|\d+(?:\.\d+)?|"
    r"->|=>|<=|>=|!=|~=|[=+\-*/^_~<>()[\]{},;|]"
)


def _replace_simple_fractions(text: str) -> str:
    """把不含嵌套花括号的 ``\frac{a}{b}`` 转成 ``(a)/(b)``。"""
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(r"(\1)/(\2)", text)
    return text


def _replace_latex_matrix(match: re.Match[str]) -> str:
    """把简单 LaTeX 矩阵变成 ``matrix[a,b;c,d]``。"""
    body = match.group(2)
    rows = re.split(r"\\\\", body)
    normalized_rows = []
    for row in rows:
        cells = [cell.strip() for cell in row.split("&")]
        normalized_rows.append(",".join(cells))
    # 前导空格可防止 ``\\det`` 与 ``matrix`` 黏成一个 token。
    return " matrix[" + ";".join(normalized_rows) + "]"


def normalize_formula(text: str) -> str:
    """统一空格、常见符号、希腊字母和最小 LaTeX 写法。

    这是检索规范化，不是数学等价证明。例如 ``x+x`` 不会被改写成
    ``2x``，以免引入不可解释的代数错误。
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")

    # NFKC 会把 ²/³ 直接压成 2/3，因此必须先保留幂运算含义。
    normalized = text.replace("²", "^2").replace("³", "^3")
    normalized = unicodedata.normalize("NFKC", normalized)
    for source, target in _UNICODE_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)

    matrix_pattern = re.compile(
        r"\\begin\{(bmatrix|pmatrix|matrix)\}(.*?)"
        r"\\end\{\1\}",
        flags=re.DOTALL,
    )
    normalized = matrix_pattern.sub(_replace_latex_matrix, normalized)
    normalized = _replace_simple_fractions(normalized)

    for source in sorted(_LATEX_COMMANDS, key=len, reverse=True):
        normalized = normalized.replace(source, _LATEX_COMMANDS[source])
    # 必须在 ``\\rightarrow`` 映射之后删除，否则会把它误改成 ``arrow``。
    normalized = normalized.replace("\\left", "").replace("\\right", "")

    # 去掉公式定界符；普通花括号在处理上下标后也不再保留。
    normalized = normalized.replace("\\(", "").replace("\\)", "")
    normalized = normalized.replace("\\[", "").replace("\\]", "")
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"([_^])\s*\{([^{}]+)\}", r"\1\2", normalized)
    normalized = normalized.replace("{", "(").replace("}", ")")

    # 教材中使用的双层方括号也统一为矩阵结构表示。
    normalized = re.sub(
        r"\[\[([^\[\]]+)\],\[([^\[\]]+)\]\]",
        lambda match: "matrix["
        + match.group(1).replace(",", ",")
        + ";"
        + match.group(2).replace(",", ",")
        + "]",
        normalized,
    )

    normalized = normalized.lower()
    normalized = re.sub(r"\b(infty|inf)\b", "infinity", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # 数学运算符周围的空格没有语义，统一删除；中文说明中的空格保留。
    operator = r"(->|=>|<=|>=|!=|~=|[=+\-*/^_~<>()[\]{},;|])"
    normalized = re.sub(rf"\s*{operator}\s*", r"\1", normalized)
    return normalized


def extract_formula_tokens(text: str) -> list[str]:
    """抽取规范化后的公式 token，供独立稀疏通道使用。"""
    return _TOKEN_PATTERN.findall(normalize_formula(text))


