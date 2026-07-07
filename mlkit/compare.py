"""Side-by-side model comparison and ASCII report generation."""
from __future__ import annotations

from .trainer import ModelResult


def compare_table(results: list[ModelResult]) -> str:
    """Render a fixed-width comparison table sorted by F1 score descending."""
    if not results:
        return "(no results)"

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train (s)"]
    rows = [list(r.summary_row().values()) for r in sorted(results, key=lambda r: -r.f1)]

    # Column widths = max of header and each cell
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    sep = "  ".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def best_model(results: list[ModelResult]) -> ModelResult:
    return max(results, key=lambda r: r.f1)


def print_confusion_matrix(result: ModelResult, class_names: list[str]) -> None:
    if result.confusion is None:
        return
    cm = result.confusion
    col_width = max(max(len(n) for n in class_names), 4) + 2
    header = " " * col_width + "".join(n.rjust(col_width) for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(class_names[i].rjust(col_width) + "".join(str(v).rjust(col_width) for v in row))


def full_report(results: list[ModelResult], class_names: list[str]) -> str:
    lines = [
        "=" * 60,
        "  ML KIT — MODEL COMPARISON REPORT",
        "=" * 60,
        "",
        compare_table(results),
        "",
    ]
    best = best_model(results)
    lines.append(f"  Best model by F1: {best.model_name}  (F1={best.f1:.4f})")
    lines.append("")
    lines.append("  Detailed classification report — " + best.model_name)
    lines.append("-" * 60)
    lines.append(best.report)
    return "\n".join(lines)
