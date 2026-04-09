# exp3_end2end/exp3_end2end.py
from __future__ import annotations

import json
import argparse
import numpy as np
from pathlib import Path

import tvm
from tvm import relay

from common.benchmark import benchmark_callable


def get_model(name):
    if name == "resnet":
        from tvm.relay.testing import resnet

        mod, params = resnet.get_workload(num_layers=18, batch_size=1)
    elif name == "mobilenet":
        from tvm.relay.testing import mobilenet

        mod, params = mobilenet.get_workload(batch_size=1)
    else:
        raise ValueError(name)

    return mod, params


def run(mod, params, target, opt_level):
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(target, 0)
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    input_shape = mod["main"].params[0].checked_type.shape
    data = np.random.randn(*[int(x) for x in input_shape]).astype("float32")

    module.set_input("data", tvm.nd.array(data, dev))

    result = benchmark_callable(lambda: module.run(), number=100)

    return result["mean_ms"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="llvm")
    parser.add_argument("--platform", type=str, default="local")
    parser.add_argument("--model", choices=["resnet", "mobilenet"], default="resnet")
    args = parser.parse_args()

    mod, params = get_model(args.model)

    no_opt = run(mod, params, args.target, opt_level=0)
    opt = run(mod, params, args.target, opt_level=3)

    result = {
        "platform": args.platform,
        "model": args.model,
        "no_opt_ms": no_opt,
        "opt_ms": opt,
        "speedup": no_opt / opt,
    }

    out_dir = Path("exp3_end2end/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{args.platform}_{args.model}.json"
    with out_file.open("w") as f:
        json.dump(result, f, indent=2)

    print(result)


if __name__ == "__main__":
    main()