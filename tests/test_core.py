import pytest

from math_clarify import FormulaDocument, analyze_formula, clarify_formula, dual_channel_rank, normalize_formula


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("α(x) → 0", "alpha(x)->0"),
        (r"\lim_{n\to\infty}\frac{1}{n}", "lim_n->infinity(1)/(n)"),
        ("x²+y³", "x^2+y^3"),
        ("[[a,b],[c,d]]", "matrix[a,b;c,d]"),
    ],
)
def test_normalization(source, expected):
    assert normalize_formula(source) == expected


def test_ambiguous_vertical_bars_ask_instead_of_guessing():
    result = clarify_formula("|A|=1")
    assert result.needs_clarification
    assert result.reason_codes == ("vertical_bar_meaning",)
    assert "行列式" in result.message


def test_clear_formula_does_not_force_a_question():
    result = analyze_formula(r"\lim_{x\to0}\frac{\sin x}{x}")
    assert result.needs_clarification is False


def test_formula_channel_can_reorder_dense_results():
    docs = [
        FormulaDocument("generic", "函数极限与连续。"),
        FormulaDocument("alpha", "f(x)=A+alpha(x), alpha(x)->0"),
    ]
    result = dual_channel_rank("f(x)=A+α(x)，α(x)→0", docs, ["generic", "alpha"])
    assert result[0]["document_id"] == "alpha"
