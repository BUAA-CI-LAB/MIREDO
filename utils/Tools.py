# this file is prepared for project 026
# Created by iboxl

import os
import struct
import utils.UtilsFunction.ToolFunction as _tool_func
import math
import psutil
from multiprocessing.shared_memory import SharedMemory
from utils.GlobalUT import *


class SharedUB:
    """跨进程共享的metric上界，基于SharedMemory的lock-free double。

    Race policy: 并发写入可能短暂覆盖更优值，但parent进程在每个worker完成后
    重新同步，恢复权威最小值。不使用锁以避免worker崩溃导致的死锁。

    平台说明: x86-64上，自然对齐的8字节mmap写入是架构级原子操作。
    SharedMemory使用mmap返回页对齐缓冲区，offset 0处的double不会被撕裂。
    """
    __slots__ = ('_shm',)

    def __init__(self, shm: SharedMemory):
        self._shm = shm

    @property
    def value(self) -> float:
        return struct.unpack_from('d', self._shm.buf, 0)[0]

    @value.setter
    def value(self, v: float):
        struct.pack_into('d', self._shm.buf, 0, v)

    def update_min(self, v: float):
        """Best-effort set value = min(current, v)."""
        if v < struct.unpack_from('d', self._shm.buf, 0)[0]:
            struct.pack_into('d', self._shm.buf, 0, v)


conv_im2col_info = _tool_func.func_conv_info

def get_PowerOfTwo(x):
    return math.pow(2, math.ceil(math.log2(x)))

def debug_get_im2col_info(FLAG_DEBUG):
    if FLAG_DEBUG:
        for idx, (layer_name, info) in enumerate(conv_im2col_info.items()):
            print(f"{idx}:    Layer: {layer_name}")
            print(f"  ori_M: {info['ori_M']}")
            print(f"  ori_K: {info['ori_K']}")
            print(f"  ori_N: {info['ori_N']}")
            print(f"  num_mul: {info['num_mul']}")
            print(f"  module: {info['module']}")
            print(f"  input shape: {info['input_shape']}, weight shape: {info['weight_shape']}")

def append_scheme_summary(outputdir:str, message:str):
    summary_file = os.path.join(outputdir, "Scheme-Summary.log")
    with open(summary_file, "a", encoding="utf-8") as file:
        file.write(message.rstrip() + "\n")


def detect_parallel_config():
    """Parallel-execution budget = the machine's physical core count.

    No load-average adjustment, no env override, no fallback: deliberately
    targets known production hardware (>=16 physical cores). `usable_cores`
    is simply the physical core count, so the scout/sweep allocation is
    deterministic per machine (reproducible — no loadavg jitter).

    `available_mem_gb` is kept only for the per-worker SOFT memory limit
    (Gurobi spill threshold), NOT for capping worker count.
    """
    logical_cores = psutil.cpu_count()
    physical_cores = psutil.cpu_count(logical=False)
    available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)

    return {
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "usable_cores": physical_cores,
        "available_mem_gb": available_mem_gb,
    }


def auto_parallel_config(usable_cores, available_mem_gb, num_schemes):
    """Parallel config derived from physical core count.

    Returns {"scout": (threads, workers), "sweep": (threads, workers),
             "scout_size": int}.

    FIX 2026-05-20 (root cause of the §5.2 EDP regression, single-variable
    bit-exact proof): SCOUT = 8 threads/scheme is the proven-necessary solve
    depth. The 0516 defect was scout_threads=4 + 4-way contention (NOT the
    window); 8 threads bit-exactly reproduces the pre-0516 curated optima,
    16 gives no further gain (Gurobi B&C plateau). workers = cores // 8.
    SWEEP retained at 1 thread × cores workers — the cheap 16-wide cull for
    the ~98% of schemes that are presolve-infeasible junk (without it every
    layer reverts to the OLD multi-day cost regime). scout_size = 20: a
    202-instance curated-CNN audit found true-winner max util_product rank
    = 15, so the winner stays in the 8-thread scout arm with margin.
    (Open: hard-layer true optima MIREDO currently misses may rank >20 —
    validated empirically before the full rerun, not assumed.)

    MIREDO_DISABLE_SCOUT_SWEEP (default OFF, pre-existing): truthy value
    forces every scheme through the scout config in one uniform phase.
    No effect unless set.
    """
    cores = usable_cores                       # = physical core count
    scout_threads = 8                          # proven solve depth (see above)
    scout_workers = max(1, cores // scout_threads)   # cores//8, no oversub
    sweep_threads = 1                                # cheap wide infeasible
    sweep_workers = max(1, cores // sweep_threads)   # cull: 1t × cores workers

    if os.environ.get("MIREDO_DISABLE_SCOUT_SWEEP", "").strip().lower() \
            not in ("", "0", "false", "no"):
        return {
            "scout": (scout_threads, scout_workers),
            "sweep": (scout_threads, scout_workers),
            "scout_size": num_schemes,
        }

    return {
        "scout": (scout_threads, scout_workers),
        "sweep": (sweep_threads, sweep_workers),
        "scout_size": min(num_schemes, 20),
    }
