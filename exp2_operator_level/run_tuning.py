# exp2_operator_level/run_tuning.py
from __future__ import annotations

import json
import argparse
import numpy as np
from pathlib import Path

import tvm
from tvm import te, auto_scheduler


@auto_scheduler.register_workload
def conv2d_workload():
    N, H, W, CI, CO, KH, KW = 1, 224, 224, 3, 16, 3, 3
    data = te.placeholder((N, CI, H + 2, W + 2), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")

    rc = te.reduce_axis((0, CI), name="rc")
    ry = te.reduce_axis((0, KH), name="ry")
    rx = te.reduce_axis((0, KW), name="rx")

    conv = te.compute(
        (N, CO, H, W),
        lambda n, co, y, x: te.sum(
            data[n, rc, y + ry, x + rx] * kernel[co, rc, ry, rx],
            axis=[rc, ry, rx],
        ),
        name="conv",
    )
    return [data, kernel, conv]


@auto_scheduler.register_workload
def depthwise_conv_workload():
    N, H, W, C, KH, KW = 1, 224, 224, 16, 3, 3
    data = te.placeholder((N, C, H + 2, W + 2), name="data")
    kernel = te.placeholder((C, 1, KH, KW), name="kernel")

    ry = te.reduce_axis((0, KH), name="ry")
    rx = te.reduce_axis((0, KW), name="rx")

    conv = te.compute(
        (N, C, H, W),
        lambda n, c, y, x: te.sum(
            data[n, c, y + ry, x + rx] * kernel[c, 0, ry, rx],
            axis=[ry, rx],
        ),
        name="dwconv",
    )
    return [data, kernel, conv]


WORKLOADS = {
    "conv2d": conv2d_workload,
    "depthwise": depthwise_conv_workload,
}


def extract_curve(log_file: Path):
    from tvm.auto_scheduler.measure_record import RecordReader

    reader = RecordReader(str(log_file))
    best = float("inf")
    curve = []

    for i, (_, res) in enumerate(reader):
        if res.error_no == 0:
            latency = float(np.mean([float(x.value) for x in res.costs]) * 1000.0)
            best = min(best, latency)
            curve.append({"trial": i + 1, "best_ms": best})

    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="llvm")
    parser.add_argument("--platform", type=str, default="local_cpu")
    parser.add_argument("--workload", choices=WORKLOADS.keys(), default="conv2d")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    target = tvm.target.Target(args.target)

    task = auto_scheduler.SearchTask(
        func=WORKLOADS[args.workload],
        args=(),
        target=target,
    )

    result_dir = Path("exp2_operator_level/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    log_file = result_dir / f"{args.platform}_{args.workload}_log.json"

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=args.trials,
        measure_callbacks=[auto_scheduler.RecordToFile(str(log_file))],
        verbose=0,
    )

    task.tune(tune_option)

    curve = extract_curve(log_file)

    output = result_dir / f"{args.platform}_{args.workload}.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2)

    print(f"Saved curve: {output}")


if __name__ == "__main__":
    main()