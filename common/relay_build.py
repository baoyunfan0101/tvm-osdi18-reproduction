# common/relay_build.py

from __future__ import annotations
from typing import Dict

import tvm
from tvm import relay
from tvm.contrib import graph_executor


def build_relay_module(
    mod: tvm.IRModule,
    params: Dict,
    target: str,
    opt_level: int,
):
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target=target, params=params)
    return lib


def create_graph_module(lib, target: str):
    dev = tvm.device(target, 0)
    module = graph_executor.GraphModule(lib["default"](dev))
    return module, dev