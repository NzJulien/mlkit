"""Command-line interface for mlkit.

Usage:
    mlkit train data.csv --target label
    mlkit train data.csv --target label --model rf
    mlkit compare data.csv --target label --cv 5
    mlkit models
    mlkit info data.csv --target label
"""
from __future__ import annotations

import argparse
import sys

from .compare import compare_table, full_report, best_model, print_confusion_matrix
from .data import load_dataset, describe_dataset
from .models import MODELS, get_model, list_models
from .trainer import train, cross_validate_model


class Color:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def c(text: str, color: str) -> str:
    return f"{color}{text}{Color.RESET}"


def cmd_info(args: argparse.Namespace) -> int:
    split = load_dataset(args.csv, args.target, scale=False)
    print(c("\nDataset info", Color.BOLD))
    print("=" * 40)
    print(describe_dataset(split))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    print(c("\nAvailable models:", Color.BOLD))
    for short, display in list_models():
        print(f"  {short:<6}  {display}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    print(f"\nLoading {args.csv}...")
    split = load_dataset(args.csv, args.target, test_size=args.test_size, scale=not args.no_scale)
    print(describe_dataset(split))
    print()

    model_keys = args.model if args.model else list(MODELS.keys())
    results = []

    for key in model_keys:
        try:
            model = get_model(key)
        except ValueError as exc:
            print(c(f"  Warning: {exc}", Color.YELLOW))
            continue

        display_name = MODELS[key][0]
        print(f"  Training {display_name}...", end=" ", flush=True)
        result = train(model, split)

        if args.cv:
            fresh = get_model(key)
            cv_mean, cv_std = cross_validate_model(fresh, split, cv=args.cv)
            result.cv_mean = cv_mean
            result.cv_std = cv_std

        results.append(result)
        print(c(f"accuracy={result.accuracy:.4f}  f1={result.f1:.4f}  ({result.train_time_s:.2f}s)", Color.GREEN))

    if not results:
        print(c("No models were trained.", Color.YELLOW))
        return 1

    print()
    print(c("Results:", Color.BOLD))
    print(compare_table(results))
    print()

    best = best_model(results)
    print(c(f"Best model: {best.model_name}  (F1={best.f1:.4f})", Color.BLUE))

    if args.verbose:
        print(c(f"\nDetailed report — {best.model_name}", Color.BOLD))
        print("-" * 50)
        print(best.report)
        print(c("Confusion matrix:", Color.BOLD))
        print_confusion_matrix(best, split.class_names)

    if args.output:
        report = full_report(results, split.class_names)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(c(f"\nReport saved to {args.output}", Color.GREEN))

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    args.model = None
    args.verbose = True
    args.output = args.output if hasattr(args, "output") else None
    return cmd_train(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlkit",
        description="Train and compare scikit-learn classifiers on any CSV dataset."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # info
    p_info = sub.add_parser("info", help="Describe a dataset")
    p_info.add_argument("csv")
    p_info.add_argument("--target", required=True, help="Target column name")
    p_info.set_defaults(func=cmd_info)

    # models
    p_models = sub.add_parser("models", help="List available model names")
    p_models.set_defaults(func=cmd_models)

    # train
    p_train = sub.add_parser("train", help="Train and evaluate one or more classifiers")
    p_train.add_argument("csv")
    p_train.add_argument("--target", required=True, help="Target column name")
    p_train.add_argument("--model", nargs="+", help="Model(s) to train (default: all)")
    p_train.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default 0.2)")
    p_train.add_argument("--cv", type=int, default=0, help="Number of CV folds (0 = skip)")
    p_train.add_argument("--no-scale", action="store_true", help="Disable StandardScaler")
    p_train.add_argument("-v", "--verbose", action="store_true", help="Print detailed report for best model")
    p_train.add_argument("-o", "--output", help="Save full report to a text file")
    p_train.set_defaults(func=cmd_train)

    # compare (alias for train --all)
    p_compare = sub.add_parser("compare", help="Train ALL models and show a comparison table")
    p_compare.add_argument("csv")
    p_compare.add_argument("--target", required=True, help="Target column name")
    p_compare.add_argument("--test-size", type=float, default=0.2)
    p_compare.add_argument("--cv", type=int, default=5, help="CV folds (default 5)")
    p_compare.add_argument("--no-scale", action="store_true")
    p_compare.add_argument("-o", "--output", help="Save report")
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
