# exp2_operator_level/plot.py
from __future__ import annotations

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


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
    files = sorted(result_dir.glob("*.json"))

    plt.figure()

    for f in files:
        if "log" in f.name:
            continue

        with f.open(encoding="utf-8") as fp:
            data = json.load(fp)

        trials = [d["trial"] for d in data]
        latency = [d["best_ms"] for d in data]

        label = f.stem
        plt.plot(trials, latency, label=label)

    plt.xlabel("Trials")
    plt.ylabel("Best Latency (ms)")
    plt.title("Exp2: Operator-Level Tuning Curve")
    plt.legend()
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()