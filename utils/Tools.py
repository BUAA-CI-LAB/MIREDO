import os
import struct
import utils.UtilsFunction.ToolFunction as _tool_func
import math
import psutil
from multiprocessing.shared_memory import SharedMemory
from utils.GlobalUT import *


class SharedUB:
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
    cores = usable_cores
    scout_threads = 8
    scout_workers = max(1, cores // scout_threads)
    sweep_threads = 1
    sweep_workers = max(1, cores // sweep_threads)

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
