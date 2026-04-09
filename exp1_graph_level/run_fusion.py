# exp1_graph_level/run_fusion.py
from __future__ import annotations
from typing import Dict, Tuple

import argparse
import numpy as np
from pathlib import Path

import tvm
from tvm import relay

from common.benchmark import benchmark_callable
from common.io_utils import save_json
from common.relay_build import build_relay_module, create_graph_module


def build_conv_bias_relu_network(
    batch_size: int = 1,
    in_channels: int = 3,
    height: int = 224,
    width: int = 224,
    out_channels: int = 16,
    kernel_size: int = 3,
) -> Tuple[tvm.IRModule, Dict[str, np.ndarray], str, Tuple[int, int, int, int]]:
    input_name = "input0"
    input_shape = (batch_size, in_channels, height, width)

    data = relay.var(input_name, shape=input_shape, dtype="float32")
    weight = relay.var(
        "weight",
        shape=(out_channels, in_channels, kernel_size, kernel_size),
        dtype="float32",
    )
    bias = relay.var("bias", shape=(out_channels,), dtype="float32")

    conv = relay.nn.conv2d(
        data=data,
        weight=weight,
        channels=out_channels,
        kernel_size=(kernel_size, kernel_size),
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
    )
    biased = relay.nn.bias_add(conv, bias, axis=1)
    out = relay.nn.relu(biased)

    func = relay.Function([data, weight, bias], out)
    mod = tvm.IRModule.from_expr(func)

    rng = np.random.default_rng(0)
    params = {
        "weight": rng.standard_normal(
            (out_channels, in_channels, kernel_size, kernel_size)
        ).astype("float32"),
        "bias": rng.standard_normal((out_channels,)).astype("float32"),
    }
    return mod, params, input_name, input_shape


def build_conv_bn_relu_network(
    batch_size: int = 1,
    in_channels: int = 3,
    height: int = 224,
    width: int = 224,
    out_channels: int = 16,
    kernel_size: int = 3,
) -> Tuple[tvm.IRModule, Dict[str, np.ndarray], str, Tuple[int, int, int, int]]:
    input_name = "input0"
    input_shape = (batch_size, in_channels, height, width)

    data = relay.var(input_name, shape=input_shape, dtype="float32")
    weight = relay.var(
        "weight",
        shape=(out_channels, in_channels, kernel_size, kernel_size),
        dtype="float32",
    )
    gamma = relay.var("gamma", shape=(out_channels,), dtype="float32")
    beta = relay.var("beta", shape=(out_channels,), dtype="float32")
    moving_mean = relay.var("moving_mean", shape=(out_channels,), dtype="float32")
    moving_var = relay.var("moving_var", shape=(out_channels,), dtype="float32")

    conv = relay.nn.conv2d(
        data=data,
        weight=weight,
        channels=out_channels,
        kernel_size=(kernel_size, kernel_size),
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
    )
    bn_tuple = relay.nn.batch_norm(
        data=conv,
        gamma=gamma,
        beta=beta,
        moving_mean=moving_mean,
        moving_var=moving_var,
        axis=1,
        epsilon=1e-5,
        center=True,
        scale=True,
    )
    bn_out = bn_tuple[0]
    out = relay.nn.relu(bn_out)

    func = relay.Function(
        [data, weight, gamma, beta, moving_mean, moving_var],
        out,
    )
    mod = tvm.IRModule.from_expr(func)

    rng = np.random.default_rng(0)
    params = {
        "weight": rng.standard_normal(
            (out_channels, in_channels, kernel_size, kernel_size)
        ).astype("float32"),
        "gamma": rng.standard_normal((out_channels,)).astype("float32"),
        "beta": rng.standard_normal((out_channels,)).astype("float32"),
        "moving_mean": rng.standard_normal((out_channels,)).astype("float32"),
        "moving_var": np.abs(rng.standard_normal((out_channels,)).astype("float32")) + 1.0,
    }
    return mod, params, input_name, input_shape


def infer_device_kind(target: str) -> str:
    lowered = target.lower()
    if "cuda" in lowered:
        return "cuda"
    return "cpu"


def ensure_target_available(target: str) -> None:
    lowered = target.lower()
    if "cuda" in lowered and not tvm.cuda(0).exist:
        raise RuntimeError("CUDA target requested, but tvm.cuda(0).exist is False.")


def benchmark_graph_module(
    module,
    dev,
    input_name: str,
    input_data: np.ndarray,
    repeat: int,
    number: int,
    warmup: int,
):
    input_arr = tvm.nd.array(input_data, device=dev)
    module.set_input(input_name, input_arr)

    def _run() -> None:
        module.run()

    return benchmark_callable(_run, repeat=repeat, number=number, warmup=warmup)


def run_single_setting(
    mod: tvm.IRModule,
    params: Dict[str, np.ndarray],
    target: str,
    input_name: str,
    input_data: np.ndarray,
    opt_level: int,
    repeat: int,
    number: int,
    warmup: int,
) -> Dict[str, object]:
    lib = build_relay_module(mod, params, target, opt_level=opt_level)
    module, dev = create_graph_module(lib, target)
    result = benchmark_graph_module(
        module,
        dev=dev,
        input_name=input_name,
        input_data=input_data,
        repeat=repeat,
        number=number,
        warmup=warmup,
    )
    result["opt_level"] = opt_level
    return result


def get_workload_builder(workload: str):
    builders = {
        "conv_bias_relu": build_conv_bias_relu_network,
        "conv_bn_relu": build_conv_bn_relu_network,
    }
    if workload not in builders:
        raise ValueError(f"Unsupported workload: {workload}")
    return builders[workload]


def default_output_path(platform: str, workload: str) -> str:
    return f"exp1_graph_level/results/{platform}_{workload}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="llvm")
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="A label used in output filenames and plots, e.g. mac_cpu, colab_cpu, colab_gpu",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="conv_bias_relu",
        choices=["conv_bias_relu", "conv_bn_relu"],
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--number", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--out-channels", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ensure_target_available(args.target)

    platform = args.platform
    if platform is None:
        device_kind = infer_device_kind(args.target)
        platform = "local_gpu" if device_kind == "cuda" else "local_cpu"

    builder = get_workload_builder(args.workload)
    mod, params, input_name, input_shape = builder(
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        height=args.height,
        width=args.width,
        out_channels=args.out_channels,
        kernel_size=args.kernel_size,
    )

    output = args.output or default_output_path(platform=platform, workload=args.workload)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    input_data = np.random.default_rng(1).standard_normal(input_shape).astype("float32")

    result_no_opt = run_single_setting(
        mod=mod,
        params=params,
        target=args.target,
        input_name=input_name,
        input_data=input_data,
        opt_level=0,
        repeat=args.repeat,
        number=args.number,
        warmup=args.warmup,
    )

    result_opt = run_single_setting(
        mod=mod,
        params=params,
        target=args.target,
        input_name=input_name,
        input_data=input_data,
        opt_level=3,
        repeat=args.repeat,
        number=args.number,
        warmup=args.warmup,
    )

    speedup = result_no_opt["mean_ms"] / result_opt["mean_ms"]

    payload = {
        "experiment": "exp1_graph_level",
        "paper": "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (OSDI 2018)",
        "platform": platform,
        "target": args.target,
        "workload": args.workload,
        "input_shape": list(input_shape),
        "settings": {
            "no_graph_optimization": result_no_opt,
            "graph_optimization": result_opt,
        },
        "speedup": speedup,
    }

    save_json(payload, output)

    print("=" * 60)
    print("Exp1: Graph-Level Optimization")
    print(f"Platform: {platform}")
    print(f"Target: {args.target}")
    print(f"Workload: {args.workload}")
    print(f"No graph optimization mean latency: {result_no_opt['mean_ms']:.6f} ms")
    print(f"Graph optimization mean latency:    {result_opt['mean_ms']:.6f} ms")
    print(f"Speedup: {speedup:.4f}x")
    print(f"Saved results to: {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()