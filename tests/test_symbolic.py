import pytest

pytest.importorskip("latex2sympy2_extended")
from math_clarify.symbolic import parse_latex_formula


def test_integral_keeps_bounds_and_variable():
    result = parse_latex_formula(r"\int_{0}^{1} x^2\,dx")
    assert result.success
    assert result.expression_type == "Integral"
    assert result.semantics["limits"] == [{"variable": "x", "lower": "0", "upper": "1"}]


def test_case_is_preserved():
    assert parse_latex_formula("A+a").semantics["all_symbols"] == ["A", "a"]
