# exp2_operator_level/plot.py
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    result_dir = Path("exp2_operator_level/results")
    files = list(result_dir.glob("*.json"))

    plt.figure()

    for f in files:
        if "log" in f.name:
            continue

        with f.open() as fp:
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

    out = result_dir / "tuning_curve.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()