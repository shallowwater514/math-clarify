# Math Clarify

An offline prototype for formula normalization, ambiguity clarification, formula term retrieval, and optional LaTeX-to-SymPy structural parsing.

**Status: v0.1.0 Alpha.** Intended for research, teaching experiments, and offline evaluation. It does not prove mathematical equivalence or infer the meaning of overloaded notation without context. For example, `|A|` can trigger a question about absolute value, determinant, vector norm, or set cardinality.

中文说明：[README.md](README.md)

## Install

Clone the repository, then run:

```bash
python -m pip install -e .
```

Optional features:

```bash
python -m pip install -e ".[symbolic]"  # LaTeX-to-SymPy
python -m pip install -e ".[ml]"        # CRF code; train your own trusted model
python -m pip install -e ".[all,dev]"   # all features and test tooling
```

This is a source release. It is not claimed to be available on PyPI.

## Quick start

```python
from math_clarify import analyze_formula, clarify_formula

result = analyze_formula(r"\lim_{x\to0}\frac{\sin x}{x}")
print(result.normalized_formula)
print(result.tokens)

ambiguous = clarify_formula("|A|=1")
print(ambiguous.needs_clarification)  # True
print(ambiguous.message)              # asks what the bars mean
```

CLI:

```bash
math-clarify "|A|=1"
math-clarify "\int_{0}^{1} x^2\,dx" --symbolic
```

Retrieval:

```python
from math_clarify import FormulaDocument, dual_channel_rank

documents = [
    FormulaDocument("intro", "An introduction to limits"),
    FormulaDocument("alpha", "f(x)=A+alpha(x), alpha(x)->0"),
]
dense_ids = ["intro", "alpha"]  # supplied by the caller
results = dual_channel_rank("f(x)=A+α(x), α(x)→0", documents, dense_ids)
print(results[0]["document_id"])
```

## Features

- **Normalization:** deterministic handling of common Unicode and LaTeX variants, arrows, Greek letters, simple fractions, and matrices for retrieval.
- **Clarification:** rule-based detection of common overloaded symbols; accepts low-confidence signals from an external segmenter.
- **Retrieval:** formula-token BM25 and reciprocal-rank fusion. A natural-language ranking is provided by the caller; this package does not call an embedding API.
- **Optional symbolic parsing:** uses `latex2sympy2-extended` to expose structure such as integral variables and bounds, without automatic evaluation.
- **Optional CRF implementation:** character-level tokenization and role prediction code. No pretrained weights are distributed.

## Limitations

Normalization is not an equivalence checker. It lowercases `A` and `a` for retrieval, which can merge mathematically distinct symbols. Keep the original formula and use the symbolic channel when case matters. The CRF segmenter showed limited generalization beyond synthetic templates, so this release does not claim general-purpose recognition accuracy. Clarification prompts are candidates that should be interpreted with the surrounding question.

The repository contains no textbook PDFs, scans, transcriptions, personal paths, training data, model weights, or private RAG02 indexes. It runs without a cloud API.

## Test

```bash
python -m pip install -e ".[all,dev]"
python -m pytest
python -m build
```

## License

Source code is available under the [MIT License](LICENSE). Third-party dependencies retain their own licenses. Do not redistribute textbook scans or transcriptions with this package.
