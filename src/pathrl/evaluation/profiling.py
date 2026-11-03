"""Timing and memory measurement.

Time and memory are measured in separate calls. tracemalloc hooks every Python
allocation, which slows code down a lot (often 2-5x), so a timing taken while it
is running would be inflated.

tracemalloc only sees allocations made through Python's allocator. Memory used
inside native libraries (PyTorch tensors, CUDA/MPS buffers) is largely invisible
to it. For RL we will report such memory separately (Stage 8).
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Measurement:
    result: Any
    wall_time_s: float           # median of `repeats` timed calls
    peak_memory_bytes: int | None


def measure(fn: Callable[[], Any], repeats: int = 3, track_memory: bool = True) -> Measurement:
    """Time `fn` `repeats` times (median), then run it once more under tracemalloc.

    The median ignores one-off slow runs (e.g. from background processes or cache warm-up).
    The garbage collector is disabled during timing so a random GC pause is not billed to `fn`.
    """
    times, result = [], None
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fn()
            times.append(time.perf_counter() - t0)
    finally:
        if gc_was_enabled:
            gc.enable()
    times.sort()

    peak = None
    if track_memory:
        gc.collect()
        tracemalloc.start()
        try:
            fn()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    return Measurement(result, times[len(times) // 2], peak)
