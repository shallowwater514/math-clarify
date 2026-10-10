"""不调用大模型的 LaTeX 数学符号解析器。

本模块把 ``latex2sympy2_extended`` 返回的 SymPy 对象整理成稳定、可序列化的
结构。当前目标是验证机器有没有保留积分变量、上下限和被积函数等符号含义，
而不是计算积分、证明公式等价或猜测公式所在学科。
"""

from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any

from latex2sympy2_extended import latex2sympy
from latex2sympy2_extended.latex2sympy2 import ConversionConfig
from sympy import Derivative, Eq, Integral, Limit, MatrixBase, Product, Sum, Symbol
from sympy.printing.repr import srepr


_CONVERSION_CONFIG = ConversionConfig(
    # 大小写在数学中可能表示不同对象，例如矩阵 A 与标量 a，不能擅自合并。
    lowercase_symbols=False,
)


@dataclass(frozen=True)
class SymbolicParseResult:
    """一次 LaTeX 解析的机器可读结果。"""

    raw_latex: str
    success: bool
    parser: str
    parser_version: str
    expression_type: str | None
    sympy_expression: str | None
    structural_repr: str | None
    semantics: dict[str, Any]
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换成可以直接写入 JSON 的字典。"""
        return asdict(self)


def _symbol_names(expression: Any) -> tuple[list[str], list[str]]:
    """返回表达式中的全部符号和自由符号。"""
    all_symbols = sorted(str(symbol) for symbol in expression.atoms(Symbol))
    free_symbols = sorted(str(symbol) for symbol in expression.free_symbols)
    return all_symbols, free_symbols


def _limit_records(limits: tuple[Any, ...]) -> list[dict[str, str | None]]:
    """把 SymPy 的积分/求和范围转换成普通字典。"""
    records: list[dict[str, str | None]] = []
    for limit in limits:
        values = tuple(limit)
        records.append(
            {
                "variable": str(values[0]),
                "lower": str(values[1]) if len(values) >= 2 else None,
                "upper": str(values[2]) if len(values) >= 3 else None,
            }
        )
    return records


def _extract_semantics(expression: Any) -> dict[str, Any]:
    """按 SymPy 对象类型提取确定性的结构字段。"""
    all_symbols, free_symbols = _symbol_names(expression)
    common: dict[str, Any] = {
        "all_symbols": all_symbols,
        "free_symbols": free_symbols,
    }

    if isinstance(expression, Integral):
        return {
            **common,
            "operator": "integral",
            "integrand": str(expression.function),
            "limits": _limit_records(expression.limits),
            "bound_symbols": [str(limit[0]) for limit in expression.limits],
        }

    if isinstance(expression, Derivative):
        return {
            **common,
            "operator": "derivative",
            "expression": str(expression.expr),
            "variables": [
                {"variable": str(variable), "order": int(order)}
                for variable, order in expression.variable_count
            ],
        }

    if isinstance(expression, Limit):
        function, variable, target, parser_direction = expression.args
        return {
            **common,
            "operator": "limit",
            "expression": str(function),
            "variable": str(variable),
            "target": str(target),
            # 这是解析器内部值。普通双侧极限也可能被 SymPy 记录成默认 '+'，
            # 因而不能仅凭这个字段断言原 LaTeX 明确写了单侧方向。
            "parser_direction": str(parser_direction),
        }

    if isinstance(expression, Sum):
        return {
            **common,
            "operator": "sum",
            "summand": str(expression.function),
            "limits": _limit_records(expression.limits),
            "bound_symbols": [str(limit[0]) for limit in expression.limits],
        }

    if isinstance(expression, Product):
        return {
            **common,
            "operator": "product",
            "factor": str(expression.function),
            "limits": _limit_records(expression.limits),
            "bound_symbols": [str(limit[0]) for limit in expression.limits],
        }

    if isinstance(expression, Eq):
        return {
            **common,
            "operator": "equality",
            "left": str(expression.lhs),
            "right": str(expression.rhs),
        }

    if isinstance(expression, MatrixBase):
        return {
            **common,
            "operator": "matrix",
            "rows": int(expression.rows),
            "columns": int(expression.cols),
            "elements": [[str(value) for value in row] for row in expression.tolist()],
        }

    return {
        **common,
        "operator": "expression",
        "expression": str(expression),
    }


def parse_latex_formula(latex: str) -> SymbolicParseResult:
    """将一条 LaTeX 公式解析为 SymPy 结构，但不执行 ``doit`` 求值。

    语法错误会返回 ``success=False``，这样离线批量审计能保留失败案例，
    不会因为一条坏公式中断整个数据集。
    """
    if not isinstance(latex, str):
        raise TypeError("latex 必须是字符串")
    if not latex.strip():
        raise ValueError("latex 不能为空")

    parser_version = version("latex2sympy2-extended")
    try:
        expression = latex2sympy(latex, conversion_config=_CONVERSION_CONFIG)
    except Exception as error:  # 第三方解析器会抛出多种 ANTLR/SymPy 异常。
        return SymbolicParseResult(
            raw_latex=latex,
            success=False,
            parser="latex2sympy2_extended",
            parser_version=parser_version,
            expression_type=None,
            sympy_expression=None,
            structural_repr=None,
            semantics={},
            error_type=type(error).__name__,
            error_message=str(error),
        )

    return SymbolicParseResult(
        raw_latex=latex,
        success=True,
        parser="latex2sympy2_extended",
        parser_version=parser_version,
        expression_type=type(expression).__name__,
        sympy_expression=str(expression),
        structural_repr=srepr(expression),
        semantics=_extract_semantics(expression),
    )


