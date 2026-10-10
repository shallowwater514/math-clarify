"""Command-line interface."""

import argparse
import json

from math_clarify.analyzer import analyze_formula


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline formula analysis")
    parser.add_argument("formula", help="Formula text or LaTeX")
    parser.add_argument("--symbolic", action="store_true")
    parser.add_argument("--max-questions", type=int, default=3)
    args = parser.parse_args()
    result = analyze_formula(
        args.formula,
        parse_symbolically=args.symbolic,
        max_questions=args.max_questions,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
