# common/benchmark.py

from __future__ import annotations
from typing import Dict, List

import time
import numpy as np


def benchmark_callable(
    fn,
    repeat: int = 10,
    number: int = 100,
    warmup: int = 5,
) -> Dict[str, object]:
    """
    Parameters
    ----------
    fn:
        A callable with no arguments.
    repeat:
        Number of repeated measurements.
    number:
        Number of runs per measurement.
    warmup:
        Number of warmup runs before timing.

    Returns
    -------
    Dict[str, object]
        Timing statistics.
    """
    for _ in range(warmup):
        fn()

    latencies_ms: List[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(number):
            fn()
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0 / number)

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "latencies_ms": latencies_ms,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "repeat": repeat,
        "number": number,
        "warmup": warmup,
    }