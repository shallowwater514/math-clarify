"""Offline, formula-aware retrieval and clarification tools."""

from math_clarify.analyzer import FormulaAnalysis, analyze_formula
from math_clarify.clarification import ClarificationQuestion, FormulaClarification, clarify_formula
from math_clarify.retrieval.expander import expand_formula_query
from math_clarify.retrieval.normalizer import extract_formula_tokens, normalize_formula
from math_clarify.retrieval.retriever import FormulaDocument, dual_channel_rank, rank_formula_documents, reciprocal_rank_fusion

__version__ = "0.1.0a1"

__all__ = [
    "ClarificationQuestion", "FormulaAnalysis", "FormulaClarification",
    "FormulaDocument", "analyze_formula", "clarify_formula",
    "dual_channel_rank", "expand_formula_query", "extract_formula_tokens",
    "normalize_formula", "rank_formula_documents", "reciprocal_rank_fusion",
]
