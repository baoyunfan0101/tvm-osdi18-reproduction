# exp1_graph_level/plot.py
from __future__ import annotations

import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def workload_label(workload: str) -> str:
    mapping = {
        "conv_bias_relu": "Conv+Bias+ReLU",
        "conv_bn_relu": "Conv+BN+ReLU",
    }
    return mapping.get(workload, workload)


def add_bar_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default="exp1_graph_level/results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp1_graph_level/results",
    )
    args = parser.parse_args()

    result_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(result_dir.glob("*.json"))

    labels = []
    no_opt = []
    opt = []

    for f in files:
        with f.open(encoding="utf-8") as fp:
            d = json.load(fp)

        if d.get("experiment") != "exp1_graph_level":
            continue

        labels.append(f"{d['platform']}\n{workload_label(d['workload'])}")
        no_opt.append(d["settings"]["no_graph_optimization"]["mean_ms"])
        opt.append(d["settings"]["graph_optimization"]["mean_ms"])

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - w / 2, no_opt, w, label="w/o graph optimization")
    plt.bar(x + w / 2, opt, w, label="w/ graph optimization")

    plt.xticks(x, labels)
    plt.ylabel("Mean Latency (ms)")
    plt.title("Exp1: Graph-Level Optimization Across Platforms and Workloads")
    plt.legend()
    plt.tight_layout()

    latency_out = out_dir / "latency.png"
    plt.savefig(latency_out, dpi=200)
    plt.close()

    speedup = [a / b for a, b in zip(no_opt, opt)]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, speedup, w)
    plt.xticks(x, labels)
    plt.ylabel("Speedup (x)")
    plt.title("Exp1: Graph-Level Optimization Across Platforms and Workloads")
    add_bar_labels(bars)
    plt.tight_layout()

    speedup_out = out_dir / "speedup.png"
    plt.savefig(speedup_out, dpi=200)
    plt.close()

    print(f"Saved: {latency_out}")
    print(f"Saved: {speedup_out}")


if __name__ == "__main__":
    main()