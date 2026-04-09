# exp2_operator_level/run_tuning.py
from __future__ import annotations

import json
import argparse
import numpy as np
from pathlib import Path

import tvm
from tvm import te, auto_scheduler


@auto_scheduler.register_workload
def conv2d_workload(
    batch_size: int,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
):
    N, H, W, CI, CO, KH, KW = (
        batch_size,
        height,
        width,
        in_channels,
        out_channels,
        kernel_size,
        kernel_size,
    )
    pad = KH // 2
    data = te.placeholder((N, CI, H + 2 * pad, W + 2 * pad), name="data")
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
def depthwise_conv_workload(
    batch_size: int,
    height: int,
    width: int,
    channels: int,
    kernel_size: int,
):
    N, H, W, C, KH, KW = (
        batch_size,
        height,
        width,
        channels,
        kernel_size,
        kernel_size,
    )
    pad = KH // 2
    data = te.placeholder((N, C, H + 2 * pad, W + 2 * pad), name="data")
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
            curve.append({"trial": i + 1, "best_ms": best, "latency_ms": latency})

    return curve


def default_log_path(results_dir: Path, platform: str, workload: str) -> Path:
    return results_dir / f"{platform}_{workload}_log.json"


def default_output_path(results_dir: Path, platform: str, workload: str) -> Path:
    return results_dir / f"{platform}_{workload}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="llvm")
    parser.add_argument("--platform", type=str, default="local_cpu")
    parser.add_argument("--workload", choices=WORKLOADS.keys(), default="conv2d")
    parser.add_argument("--trials", type=int, default=100)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--out-channels", type=int, default=16)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=3)

    parser.add_argument(
        "--results-dir",
        type=str,
        default="exp2_operator_level/results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the tuning curve JSON output.",
    )
    parser.add_argument(
        "--log-output",
        type=str,
        default=None,
        help="Path to the auto-scheduler log JSON output.",
    )
    args = parser.parse_args()

    target = tvm.target.Target(args.target)

    if args.workload == "conv2d":
        task_args = (
            args.batch_size,
            args.height,
            args.width,
            args.in_channels,
            args.out_channels,
            args.kernel_size,
        )
    else:
        task_args = (
            args.batch_size,
            args.height,
            args.width,
            args.channels,
            args.kernel_size,
        )

    task = auto_scheduler.SearchTask(
        func=WORKLOADS[args.workload],
        args=task_args,
        target=target,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = Path(args.log_output) if args.log_output else default_log_path(
        results_dir=results_dir,
        platform=args.platform,
        workload=args.workload,
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=args.trials,
        measure_callbacks=[auto_scheduler.RecordToFile(str(log_file))],
        verbose=0,
    )

    task.tune(tune_option)

    curve = extract_curve(log_file)

    output = Path(args.output) if args.output else default_output_path(
        results_dir=results_dir,
        platform=args.platform,
        workload=args.workload,
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "platform": args.platform,
        "target": args.target,
        "workload": args.workload,
        "shape": {
            "batch_size": args.batch_size,
            "height": args.height,
            "width": args.width,
            "in_channels": args.in_channels,
            "out_channels": args.out_channels,
            "channels": args.channels,
            "kernel_size": args.kernel_size,
        },
        "trials": args.trials,
        "curve": curve,
    }

    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved tuning log: {log_file}")
    print(f"Saved curve: {output}")


if __name__ == "__main__":
    main()