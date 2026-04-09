# exp3_end2end/run_end2end.py
from __future__ import annotations

import json
import argparse
import numpy as np
from pathlib import Path

import tvm
from tvm import relay

from common.benchmark import benchmark_callable


def get_model(name: str, batch_size: int):
    if name == "resnet":
        from tvm.relay.testing import resnet

        mod, params = resnet.get_workload(num_layers=18, batch_size=batch_size)
    elif name == "mobilenet":
        from tvm.relay.testing import mobilenet

        mod, params = mobilenet.get_workload(batch_size=batch_size)
    else:
        raise ValueError(name)

    return mod, params


def run(mod, params, target, opt_level, number: int, repeat: int, warmup: int):
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(target, 0)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    input_shape = mod["main"].params[0].checked_type.shape
    data = np.random.randn(*[int(x) for x in input_shape]).astype("float32")

    module.set_input("data", tvm.nd.array(data, dev))

    result = benchmark_callable(lambda: module.run(), number=number, repeat=repeat, warmup=warmup)

    return result


def default_output_path(results_dir: Path, platform: str, model: str) -> Path:
    return results_dir / f"{platform}_{model}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="llvm")
    parser.add_argument("--platform", type=str, default="local")
    parser.add_argument("--model", choices=["resnet", "mobilenet"], default="resnet")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--number", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--results-dir",
        type=str,
        default="exp3_end2end/results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    mod, params = get_model(args.model, args.batch_size)

    no_opt = run(mod, params, args.target, opt_level=0, number=args.number, repeat=args.repeat, warmup=args.warmup)
    opt = run(mod, params, args.target, opt_level=3, number=args.number, repeat=args.repeat, warmup=args.warmup)

    result = {
        "platform": args.platform,
        "target": args.target,
        "model": args.model,
        "batch_size": args.batch_size,
        "no_opt_ms": no_opt["mean_ms"],
        "opt_ms": opt["mean_ms"],
        "speedup": no_opt["mean_ms"] / opt["mean_ms"],
        "settings": {
            "no_opt": no_opt,
            "opt": opt,
        },
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_file = Path(args.output) if args.output else default_output_path(
        results_dir=results_dir,
        platform=args.platform,
        model=args.model,
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(result)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()