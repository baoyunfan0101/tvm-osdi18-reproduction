# exp1_graph_level/plot.py
from __future__ import annotations
from typing import Dict, List, Tuple

import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_result(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_results(results_dir: Path) -> List[Tuple[str, str, float, float]]:
    records: List[Tuple[str, str, float, float]] = []

    for path in sorted(results_dir.glob("*.json")):
        data = load_result(path)
        if data.get("experiment") != "exp1_graph_level":
            continue

        platform = str(data["platform"])
        workload = str(data["workload"])
        no_opt = float(data["settings"]["no_graph_optimization"]["mean_ms"])
        opt = float(data["settings"]["graph_optimization"]["mean_ms"])
        records.append((platform, workload, no_opt, opt))

    return records


def workload_label(workload: str) -> str:
    mapping = {
        "conv_bias_relu": "Conv+Bias+ReLU",
        "conv_bn_relu": "Conv+BN+ReLU",
    }
    return mapping.get(workload, workload)


def make_chart(records: List[Tuple[str, str, float, float]], output_path: Path) -> None:
    if not records:
        raise ValueError("No exp1_graph_level result JSON files found.")

    records = sorted(records, key=lambda x: (x[0], x[1]))

    labels = [f"{platform}\n{workload_label(workload)}" for platform, workload, _, _ in records]
    no_opt_values = [no_opt for _, _, no_opt, _ in records]
    opt_values = [opt for _, _, _, opt in records]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(max(10, len(labels) * 1.6), 5.5))
    plt.bar(x - width / 2, no_opt_values, width, label="w/o graph optimization")
    plt.bar(x + width / 2, opt_values, width, label="w/ graph optimization")
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Mean Latency (ms)")
    plt.title("Exp1: Graph-Level Optimization Across Platforms and Workloads")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_summary(records: List[Tuple[str, str, float, float]], output_path: Path) -> None:
    summary = []
    for platform, workload, no_opt, opt in records:
        summary.append(
            {
                "platform": platform,
                "workload": workload,
                "no_graph_optimization_mean_ms": no_opt,
                "graph_optimization_mean_ms": opt,
                "speedup": no_opt / opt,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default="exp1_graph_level/results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exp1_graph_level/results/fusion_comparison.png",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="exp1_graph_level/results/summary.json",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    records = collect_results(results_dir)
    make_chart(records, output_path)
    write_summary(records, summary_path)

    print(f"Saved figure to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()