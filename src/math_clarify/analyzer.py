"""One-call public API for deterministic formula analysis."""

from dataclasses import asdict, dataclass
from typing import Any

from math_clarify.clarification import clarify_formula
from math_clarify.retrieval.normalizer import extract_formula_tokens, normalize_formula


@dataclass(frozen=True)
class FormulaAnalysis:
    raw_formula: str
    normalized_formula: str
    tokens: tuple[str, ...]
    needs_clarification: bool
    clarification: dict[str, Any]
    symbolic: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_formula(
    formula: str,
    *,
    segmentation: Any | None = None,
    parse_symbolically: bool = False,
    max_questions: int = 3,
) -> FormulaAnalysis:
    """Normalize, tokenize and detect ambiguity without an LLM."""
    normalized = normalize_formula(formula)
    clarification = clarify_formula(formula, segmentation, max_questions=max_questions)
    symbolic: dict[str, Any] | None = None
    if parse_symbolically:
        try:
            from math_clarify.symbolic import parse_latex_formula
        except ImportError as error:
            raise RuntimeError(
                "Symbolic parsing requires: pip install 'math-clarify[symbolic]'"
            ) from error
        symbolic = parse_latex_formula(formula).to_dict()
    return FormulaAnalysis(
        raw_formula=formula,
        normalized_formula=normalized,
        tokens=tuple(extract_formula_tokens(formula)),
        needs_clarification=clarification.needs_clarification,
        clarification=clarification.to_dict(),
        symbolic=symbolic,
    )
