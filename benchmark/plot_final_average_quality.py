"""Create the final reviewer plot from the archived benchmark output."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "outputs" / "AIaaS_Composition_Benchmark_Results.csv"
SUMMARY_CSV = ROOT / "outputs" / "AIaaS_Composition_Quality_by_Length.csv"
OUTPUT_PDF = ROOT / "outputs" / "AIaaS_Composition_Quality_Comparison.pdf"
OUTPUT_PNG = ROOT / "outputs" / "AIaaS_Composition_Quality_Comparison.png"

ALGORITHMS = [
    ("Random Search", "random_search_best_score", "#4C78A8", "o"),
    ("Greedy", "greedy_best_score", "#F58518", "s"),
    ("Epsilon-Greedy", "epsilon_greedy_best_score", "#54A24B", "^"),
    ("GA", "ga_best_score", "#E45756", "D"),
    ("DAAGA", "daaga_best_score", "#72B7B2", "v"),
    ("MWOA", "mwoa_best_score", "#B279A2", "P"),
    ("CSSA", "cssa_best_score", "#FF9DA6", "X"),
    ("SDFGA", "sdfga_best_score", "#9D755D", "*"),
    ("BPSC-GA", "bpsc_ga_best_score", "#BAB0AC", "h"),
    ("PK-IDPSO", "pk_idpso_best_score", "#59A14F", "<"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot average service quality by composition length.")
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--output-pdf", type=Path, default=OUTPUT_PDF)
    parser.add_argument("--output-png", type=Path, default=OUTPUT_PNG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_columns = [column for _, column, _, _ in ALGORITHMS]
    values: dict[int, dict[str, list[float]]] = defaultdict(lambda: {column: [] for column in score_columns})
    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                length = int(row["composition_length_K"])
            except (TypeError, ValueError):
                continue
            for column in score_columns:
                try:
                    values[length][column].append(float(row[column]))
                except (TypeError, ValueError):
                    pass

    grouped = {
        length: {
            column: (sum(numbers) / len(numbers) if numbers else float("nan"))
            for column, numbers in columns.items()
        }
        for length, columns in sorted(values.items())
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["composition_length_K", *score_columns])
        for length, columns in grouped.items():
            writer.writerow([length, *[columns[column] for column in score_columns]])

    plt.rcParams.update({
        "font.family": "serif",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "grid.color": "#D0D0D0",
        "grid.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(9.8, 5.6), dpi=300)
    x = list(grouped)
    for label, column, color, marker in ALGORITHMS:
        ax.plot(
            x,
            [grouped[length][column] for length in x],
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6.0,
        )
    ax.set_xlabel("Composition Length", fontsize=18)
    ax.set_ylabel("Average Solution Quality Score", fontsize=18)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylim(0.5, 0.86)
    ax.grid(True, axis="both", linestyle="-", alpha=0.85)
    ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.92)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.14, top=0.90)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf, bbox_inches="tight")
    fig.savefig(args.output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output_pdf}")
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()
