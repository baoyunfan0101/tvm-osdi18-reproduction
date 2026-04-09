# exp3_end2end/plot.py
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    result_dir = Path("exp3_end2end/results")
    files = list(result_dir.glob("*.json"))

    labels = []
    no_opt = []
    opt = []

    for f in files:
        with f.open() as fp:
            d = json.load(fp)

        labels.append(f"{d['platform']}\n{d['model']}")
        no_opt.append(d["no_opt_ms"])
        opt.append(d["opt_ms"])

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - w / 2, no_opt, w, label="no opt")
    plt.bar(x + w / 2, opt, w, label="opt")

    plt.xticks(x, labels)
    plt.ylabel("Latency (ms)")
    plt.title("Exp3: End-to-End Performance")
    plt.legend()
    plt.tight_layout()

    out = result_dir / "end2end.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()