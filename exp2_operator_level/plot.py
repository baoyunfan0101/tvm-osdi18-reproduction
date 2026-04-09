# exp2_operator_level/plot.py
from __future__ import annotations

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def extract_curve(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "curve" in data:
            return data["curve"]
        raise ValueError("Unsupported JSON format: missing 'curve' field.")
    raise ValueError("Unsupported JSON format.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default="exp2_operator_level/results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exp2_operator_level/results/tuning_curve.png",
    )
    args = parser.parse_args()

    result_dir = Path(args.results_dir)
    files = list(result_dir.glob("*.json"))

    plt.figure(figsize=(10, 5))

    for f in files:
        if "log" in f.name:
            continue

        with f.open() as fp:
            data = json.load(fp)

        curve = extract_curve(data)
        trials = [d["trial"] for d in curve]
        latency = [d["best_ms"] for d in curve]

        label = f.stem
        plt.plot(trials, latency, label=label)

    plt.xlabel("Trials")
    plt.ylabel("Best Latency (ms)")
    plt.title("Exp2: Operator-Level Tuning Curve Across Platforms and Workloads")
    plt.legend()
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()