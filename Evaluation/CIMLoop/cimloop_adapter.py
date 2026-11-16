from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from Architecture.HardwareSpec import HardwareSpec



_THIS_FILE = Path(__file__).resolve()
_CIMLOOP_ADAPTER_DIR = _THIS_FILE.parent
_TIMELOOP_SUBMODULE_ROOT = _CIMLOOP_ADAPTER_DIR / "timeloop-accelergy-infra" / "src" / "timeloop"
_TIMELOOP_SUBMODULE_BIN = _TIMELOOP_SUBMODULE_ROOT / "bin"
_TIMELOOP_SUBMODULE_LIB = _TIMELOOP_SUBMODULE_ROOT / "lib"
_ACCELERGY_SOURCE_ROOT = _CIMLOOP_ADAPTER_DIR / "timeloop-accelergy-infra" / "src" / "accelergy"
_CIMLOOP_REPO = _CIMLOOP_ADAPTER_DIR / "cimloop"
_CIMLOOP_MODELS = _CIMLOOP_REPO / "workspace" / "models"
_CIMLOOP_INCLUDE = _CIMLOOP_MODELS / "include"
_CIMLOOP_COMPONENTS = _CIMLOOP_MODELS / "components"
_CIMLOOP_ACCELERGY_PLUGINS = _CIMLOOP_COMPONENTS / "accelergy_plug_ins"
_MIREDO_CELLS_DIR = _CIMLOOP_ADAPTER_DIR / "MIREDO_cells"
_MIREDO_CELL_YAML = _MIREDO_CELLS_DIR / "MIREDO_sram.cell.yaml"
_MIREDO_COMPONENTS_DIR = _CIMLOOP_ADAPTER_DIR / "MIREDO_components"
_CIMLOOP_SCRIPTS = _CIMLOOP_REPO / "workspace" / "scripts"


_CONDA_PREFIX = os.environ.get("CONDA_PREFIX") or sys.prefix
_CONDA_LIB = Path(_CONDA_PREFIX) / "lib"



@contextmanager
def _timeloop_runtime_env():
    old_path = os.environ.get("PATH", "")
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    old_pythonpath = os.environ.get("PYTHONPATH")
    try:
        os.environ["PATH"] = f"{_TIMELOOP_SUBMODULE_BIN}:{old_path}"
        parts = [str(_TIMELOOP_SUBMODULE_LIB)]
        if _CONDA_LIB.is_dir():
            parts.append(str(_CONDA_LIB))
        if old_ld:
            parts.append(old_ld)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        if (_ACCELERGY_SOURCE_ROOT / "accelergy" / "accelergy_console.py").is_file():
            os.environ["PYTHONPATH"] = str(_ACCELERGY_SOURCE_ROOT) + (":"+old_pythonpath if old_pythonpath else "")
        yield
    finally:
        os.environ["PATH"] = old_path
        if old_ld:
            os.environ["LD_LIBRARY_PATH"] = old_ld
        else:
            os.environ.pop("LD_LIBRARY_PATH", None)
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath


def _shim_pytimeloop_if_needed():
    if "pytimeloop.timeloopfe.v4" in sys.modules:
        return
    import timeloopfe
    import timeloopfe.v4
    pkg = types.ModuleType("pytimeloop")
    pkg_tf = types.ModuleType("pytimeloop.timeloopfe")
    pkg.timeloopfe = pkg_tf
    pkg_tf.v4 = timeloopfe.v4
    sys.modules["pytimeloop"] = pkg
    sys.modules["pytimeloop.timeloopfe"] = pkg_tf
    sys.modules["pytimeloop.timeloopfe.v4"] = timeloopfe.v4


def _load_array_processor():
    _shim_pytimeloop_if_needed()
    import importlib.util
    cached = sys.modules.get("_cimloop_array_processor")
    if cached is not None:
        return cached.ArrayProcessor
    processors_path = _CIMLOOP_SCRIPTS / "processors.py"
    spec = importlib.util.spec_from_file_location("_cimloop_array_processor", processors_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cimloop_array_processor"] = module
    spec.loader.exec_module(module)
    return module.ArrayProcessor



_MIREDO_STORAGE_INNER_TO_OUTER = [
    "Macro", "OReg", "IReg", "Input_buffer", "Output_buffer", "Global_buffer", "Dram"
]

_MIREDO_AXIS_TO_FANOUT: List[Tuple[str, str]] = [
    ("cores", "X"),
    ("row", "Y"),
    ("column", "X"),
]


_OBJECTIVE_TO_METRICS: Dict[str, List[str]] = {
    "Latency": ["delay", "energy"],
    "Energy": ["energy", "delay"],
    "EDP": ["edp"],
}

DEFAULT_MAPPER_SEARCH_SIZE = 100_000
DEFAULT_MAPPER_VICTORY_CONDITION = 1_000
DEFAULT_MAPPER_TIMEOUT = 100_000


@dataclass
class CIMLoopLayerOutput:
    layer_fp: str
    loopdim: Dict[str, int]
    objective: str
    optimization_metric: List[str]
    map_yaml_path: Path
    map_txt_path: Path
    storage_level_names: List[str]
    axis_sources: List[Dict[str, str]]
    axis_to_fanout: List[Tuple[str, str]]
    runtime_s: float
    art_summary_path: Optional[Path] = None



def _storage_level_by_name(spec: HardwareSpec, name: str):
    for lv in spec.memory_hierarchy:
        if lv.name == name:
            return lv
    raise KeyError(f"HardwareSpec has no memory level named {name!r}")


def _bits_to_bytes(bits: int) -> int:
    return max(1, int(bits) // 8)


def cimloop_macro_depth_cells(spec: HardwareSpec) -> int:
    return max(1, int(spec.macro.compartment_depth))


def _derive_template_vars(spec: HardwareSpec, global_cycle_s: float) -> Dict[str, Any]:
    mac = spec.macro
    dram = _storage_level_by_name(spec, "Dram")
    glb = _storage_level_by_name(spec, "Global_buffer")
    ibuf = _storage_level_by_name(spec, "Input_buffer")
    obuf = _storage_level_by_name(spec, "Output_buffer")

    tech_node_nm = int(round(mac.tech_params.tech_node * 1000))

    glb_width = max(int(glb.r_bw_bits_per_cycle), 8)
    obuf_width = max(int(obuf.r_bw_bits_per_cycle), 8)
    ibuf_width = max(int(ibuf.r_bw_bits_per_cycle), 8)
    dram_width = max(int(dram.r_bw_bits_per_cycle), 8)

    glb_depth = max(1, int(glb.size_bits) // glb_width)
    ibuf_depth = max(1, int(ibuf.size_bits) // ibuf_width)
    obuf_depth = max(1, int(obuf.size_bits) // obuf_width)
    dram_block_size = max(1, dram_width // 8)

    n_cores = int(spec.cores)
    dim_x = int(mac.dimX)
    dim_y = int(mac.dimY)

    return {
        "CIM_UNIT_DEPTH_CELLS": cimloop_macro_depth_cells(spec),
        "TECHNOLOGY": tech_node_nm,
        "VOLTAGE": mac.tech_params.vdd,
        "GLOBAL_CYCLE_SECONDS": float(global_cycle_s),
        "INPUT_BITS": int(mac.precision.I),
        "WEIGHT_BITS": int(mac.precision.W),
        "OUTPUT_BITS": int(mac.precision.O_final),
        "ENCODED_OUTPUT_BITS_DEFAULT": int(mac.precision.psum),
        "N_CORES": n_cores,
        "DRAM_WIDTH_BITS": dram_width,
        "DRAM_BLOCK_SIZE": dram_block_size,
        "GLB_WIDTH_BITS": glb_width,
        "GLB_DEPTH": glb_depth,
        "OBUF_WIDTH_BITS": obuf_width,
        "OBUF_DEPTH": obuf_depth,
        "IBUF_WIDTH_BITS": ibuf_width,
        "IBUF_DEPTH": ibuf_depth,
        "N_DIMY_FANOUT": dim_y,
        "ARRAY_BITLINES": dim_x,
        "ARRAY_WORDLINES": dim_x,
        "ARRAY_PARALLEL_OUTPUTS": dim_y,
        "N_ROWS_PER_REG": dim_x,
        "DAC_RESOLUTION": int(mac.input_bit_per_cycle),
        "CELL_CONFIG": str(_MIREDO_CELL_YAML),
        "SLICING_ENCODING_PATH": str(_CIMLOOP_INCLUDE / "slicing_encoding.py"),
        "ACCELERGY_PLUGINS_PATH": str(_CIMLOOP_ACCELERGY_PLUGINS),
        "DEFINES_PATH": str(_CIMLOOP_INCLUDE / "defines.yaml"),
        "VARIABLES_COMMON_PATH": str(_CIMLOOP_INCLUDE / "variables_common.yaml"),
        "COMPONENTS_GLOB": str(_CIMLOOP_COMPONENTS / "*.yaml"),
        "COLONNADE_COMPONENTS_PATH": str(_CIMLOOP_MODELS / "arch" / "1_macro" / "colonnade_jssc_2021" / "components.yaml"),
        "MIREDO_COMPONENTS_GLOB": str(_MIREDO_COMPONENTS_DIR / "*.yaml"),
        "MIREDO_CELLS_DIR": str(_MIREDO_CELLS_DIR),
        "CIMLOOP_INCLUDE_DIR": str(_CIMLOOP_INCLUDE),
        "CIMLOOP_MEMORY_CELLS_DIR": str(_CIMLOOP_MODELS / "memory_cells"),
    }



def _loopdim_to_instance(loopdim: Dict[str, int]) -> Dict[str, Any]:
    stride = int(loopdim.get("Stride", 1))
    dilation = int(loopdim.get("Dilation", 1))
    return {
        "N": 1,
        "C": int(loopdim.get("C", 1)),
        "M": int(loopdim.get("K", 1)),
        "G": int(max(1, loopdim.get("G", 1))),
        "R": int(loopdim.get("R", 1)),
        "S": int(loopdim.get("S", 1)),
        "P": int(loopdim.get("P", 1)),
        "Q": int(loopdim.get("Q", 1)),
        "H": int(loopdim.get("H", 1)),
        "W_img": int(loopdim.get("W", 1)),
        "Hstride": stride,
        "Wstride": stride,
        "Hdilation": dilation,
        "Wdilation": dilation,
    }



_TOP_TEMPLATE = r"""

{{add_to_path(cwd())}}
{{add_to_path('%(CIMLOOP_INCLUDE_DIR)s')}}
{{add_to_path('%(CIMLOOP_MEMORY_CELLS_DIR)s')}}
{{add_to_path('%(MIREDO_CELLS_DIR)s')}}

{{include_text('%(DEFINES_PATH)s')}}

globals:
  version: 0.4
  environment_variables:
    TIMELOOP_OUTPUT_STAT_SCIENTIFIC: 1
    TIMELOOP_OUTPUT_STAT_DEFAULT_FLOAT: 0
    TIMELOOP_HIDE_INCONSEQUENTIAL_STATS: 0
  expression_custom_functions:
  - %(SLICING_ENCODING_PATH)s
  accelergy_plug_ins:
  - %(ACCELERGY_PLUGINS_PATH)s

architecture:
  version: 0.4
  nodes:
  - !Container
    name: system
    <<<: [*container_defaults]
    attributes: {has_power_gating: True}

  - !Component
    name: Dram
    <<<: [*component_defaults]
    class: DRAM
    attributes:
      type: "LPDDR4"
      width: %(DRAM_WIDTH_BITS)s
      datawidth: 8
      block_size: %(DRAM_BLOCK_SIZE)s
    constraints:
      dataspace: {keep_only: [Inputs, Weights, Outputs]}

  - !Component
    name: Global_buffer
    <<<: [*component_defaults]
    subclass: MIREDO_smartbuffer_sram
    attributes:
      width: %(GLB_WIDTH_BITS)s
      depth: %(GLB_DEPTH)s
      datawidth: %(WEIGHT_BITS)s
    constraints:
      dataspace: {keep_only: [Inputs, Weights, Outputs]}

  - !Container
    name: cores
    <<<: [*container_defaults]
    spatial: {meshX: %(N_CORES)s}
    constraints:
      spatial:
        maximize_dims: [[P, Q, M, G]]

  - !Component
    name: Output_buffer
    <<<: [*component_defaults]
    subclass: MIREDO_smartbuffer_sram
    attributes:
      width: %(OBUF_WIDTH_BITS)s
      depth: %(OBUF_DEPTH)s
      datawidth: %(ENCODED_OUTPUT_BITS_DEFAULT)s
    constraints:
      dataspace: {keep_only: [Outputs]}

  - !Component
    name: Input_buffer
    <<<: [*component_defaults]
    subclass: MIREDO_smartbuffer_sram
    attributes:
      width: %(IBUF_WIDTH_BITS)s
      depth: %(IBUF_DEPTH)s
      datawidth: %(INPUT_BITS)s
    constraints:
      dataspace: {keep_only: [Inputs]}

  - !Component
    name: row_drivers
    <<<: [*component_defaults, *keep_inputs, *no_coalesce]
    subclass: input_row_drivers
    attributes: {width: DAC_RESOLUTION, <<: *cim_component_attributes}

  - !Component
    name: IReg
    <<<: [*component_defaults, *keep_inputs, *no_coalesce]
    subclass: colonnade_cim_logic_input_port
    attributes:
      width: DAC_RESOLUTION
      n_instances: column.get_fanout()
      switching_activity: (AVERAGE_INPUT_VALUE * (1 - AVERAGE_INPUT_VALUE)) ** 0.5
      voltage: VOLTAGE

  - !Component
    name: column_drivers
    <<<: [*component_defaults, *keep_outputs, *no_coalesce]
    subclass: column_drivers
    attributes: {<<<: *cim_component_attributes, cols_active_at_once: 1}

  - !Component
    name: weight_drivers
    <<<: [*component_defaults, *keep_weights, *no_coalesce]
    subclass: weight_row_drivers
    attributes: {width: ARRAY_BITLINES, <<: *cim_component_attributes}

  - !ArrayContainer
    name: column
    <<<: [*container_defaults, *spatial_must_reuse_inputs]
    spatial: {meshX: %(N_DIMY_FANOUT)s}
    constraints:
      spatial:
        maximize_dims: [[M]]
    max_utilization:
      spatial: {factors: [('M=' + str(column.get_fanout()))]}

  - !Component
    name: column_bandwidth_limiter
    <<<: [*component_defaults, *keep_weights, *keep_outputs, *no_coalesce]
    attributes:
      width: 1
      read_bandwidth: 1
      per_dataspace_bandwidth_consumption_scale:
        Weights: 1 / WEIGHT_BITS_PER_SLICE
        Outputs: 1 / ENCODED_OUTPUT_BITS

  - !ArrayContainer
    name: row_group
    <<<: [*container_defaults, *spatial_must_reuse_outputs]
    spatial: {meshY: 1}

  - !Component
    name: OReg
    <<<: [*component_defaults, *no_temporal_iteration, *no_temporal_reuse, *keep_outputs]
    subclass: colonnade_register
    attributes: {width: ENCODED_OUTPUT_BITS, depth: 1, <<: *cim_component_attributes}

  - !ArrayContainer
    name: row
    <<<: [*container_defaults, *spatial_must_reuse_outputs]
    spatial: {meshY: %(ARRAY_BITLINES)s}
    constraints:
      spatial:
        maximize_dims: [[R, S, C]]
    max_utilization:
      spatial: {factors: [('C=' + str(row.get_fanout()))]}

  - !Component
    name: Macro
    <<<: [*component_defaults]
    subclass: cell
    attributes:
      <<: *cim_component_attributes
      width: WEIGHT_BITS_PER_SLICE
      depth: CIM_UNIT_DEPTH_CELLS
      width_cells: CIM_UNIT_WIDTH_CELLS
      depth_cells: CIM_UNIT_DEPTH_CELLS
    constraints: !nomerge
      dataspace: {keep_only: [Weights]}
      temporal:
        permutation: [C, M]
        no_iteration_over_dataspaces: []
        maximize_dims_capacity: attributes.depth
        maximize_dims: [C, M]
        factors: [P=1, Q=1, R=1, S=1, N=1, X=1, Y=1, Z=1, G=1]

  - !Component
    name: cim_logic
    <<<: [*component_defaults, *keep_weights, *no_coalesce]
    subclass: colonnade_cim_logic
    attributes:
      width: CIM_UNIT_WIDTH_CELLS * BITS_PER_CELL
      switching_activity: N_ADDERS_CRITICAL_PATH * (AVERAGE_INPUT_VALUE * (1 - AVERAGE_INPUT_VALUE)) ** 0.5

  - !Hierarchical
    nodes: *virtualized_mac

variables:
- {version: 0.4}

- WEIGHT_BITS: %(WEIGHT_BITS)s
  INPUT_BITS: %(INPUT_BITS)s
  OUTPUT_BITS: %(OUTPUT_BITS)s
  BATCH_SIZE: 1
  VOLTAGE: %(VOLTAGE)s
  TECHNOLOGY: %(TECHNOLOGY)s
  CELL_CONFIG: "%(CELL_CONFIG)s"
  VOLTAGE_ENERGY_SCALE: (VOLTAGE / 0.8) ** 2
  VOLTAGE_LATENCY_SCALE: 0.8 / VOLTAGE
  ADC_ENERGY_SCALE: 1 * VOLTAGE_ENERGY_SCALE
  ADC_AREA_SCALE: 1
  ROW_COL_DRIVERS_AREA_SCALE: 1
  SUPPORTED_INPUT_BITS:  %(INPUT_BITS)s
  SUPPORTED_WEIGHT_BITS: %(WEIGHT_BITS)s
  SUPPORTED_OUTPUT_BITS: %(ENCODED_OUTPUT_BITS_DEFAULT)s
  INPUTS_HIST:  [1, 1, 1, 1, 1, 1, 1]
  WEIGHTS_HIST: [1, 1, 1, 1, 1, 1, 1]
  OUTPUTS_HIST: INPUTS_HIST

- CIM_ARCHITECTURE: True
  ARRAY_WORDLINES:        %(ARRAY_WORDLINES)s
  ARRAY_BITLINES:         %(ARRAY_BITLINES)s
  ARRAY_PARALLEL_INPUTS:  1
  ARRAY_PARALLEL_OUTPUTS: %(ARRAY_PARALLEL_OUTPUTS)s
  ARRAY_PARALLEL_WEIGHTS: %(ARRAY_PARALLEL_OUTPUTS)s

  ENCODED_INPUT_BITS:  INPUT_BITS
  ENCODED_WEIGHT_BITS: WEIGHT_BITS + ceil(log2(ARRAY_WORDLINES))
  ENCODED_OUTPUT_BITS: %(ENCODED_OUTPUT_BITS_DEFAULT)s
  INPUT_ENCODING_FUNC:  offset_encode_hist
  WEIGHT_ENCODING_FUNC: offset_encode_hist
  SIGNED_SUM_ACROSS_INPUTS:  False
  SIGNED_SUM_ACROSS_WEIGHTS: False

  CIM_UNIT_WIDTH_CELLS: 1
  CIM_UNIT_DEPTH_CELLS: %(CIM_UNIT_DEPTH_CELLS)s
  BITS_PER_CELL:        %(WEIGHT_BITS)s

  ADC_RESOLUTION:        1
  VOLTAGE_DAC_RESOLUTION: 1
  TEMPORAL_DAC_RESOLUTION: 1
  DAC_RESOLUTION:        %(DAC_RESOLUTION)s

  N_REGS_PER_COL: 1
  N_ROWS_PER_REG: %(N_ROWS_PER_REG)s

  FORCE_100MHZ:   False
  BIT_PIPELINED:  True
  N_ADDERS_CRITICAL_PATH: ENCODED_WEIGHT_BITS + N_ROWS_PER_REG + 2
  CRITICAL_PATH_LATENCY:  N_ADDERS_CRITICAL_PATH * 0.4e-9
  BASE_LATENCY:    10e-9 if FORCE_100MHZ else CRITICAL_PATH_LATENCY
  LATENCY_PIPELINE_SCALE: 1 if BIT_PIPELINED else (N_REGS_PER_COL + 1)
  GLOBAL_CYCLE_SECONDS: BASE_LATENCY * VOLTAGE_LATENCY_SCALE * LATENCY_PIPELINE_SCALE
  READ_PULSE_WIDTH: GLOBAL_CYCLE_SECONDS

- {{include('%(VARIABLES_COMMON_PATH)s', 'variables')}}

components:
  version: 0.4
  classes:
  - {{include_all('%(COMPONENTS_GLOB)s', 'compound_components.classes')}}
  - {{include('%(COLONNADE_COMPONENTS_PATH)s', 'compound_components.classes')}}
  - {{include_all('%(MIREDO_COMPONENTS_GLOB)s', 'compound_components.classes')}}

mapper:
  version: 0.4
  optimization_metrics: [%(OPTIMIZATION_METRICS)s]
  algorithm: random_pruned
  search_size: %(MAPPER_SEARCH_SIZE)s
  victory_condition: %(MAPPER_VICTORY_CONDITION)s
  timeout: %(MAPPER_TIMEOUT)s
  max_permutations_per_if_visit: 16
  max_temporal_loops_in_a_mapping: 14
  num_threads: %(MAPPER_NUM_THREADS)d
  live_status: False
  diagnostics: False
  log_stats: False
  log_suboptimal: False

problem:
  version: 0.4
  instance:
    N: 1
    C: %(pC)s
    M: %(pM)s
    G: %(pG)s
    R: %(pR)s
    S: %(pS)s
    P: %(pP)s
    Q: %(pQ)s
    H: %(pH)s
    W: %(pW)s
    X: 1
    Y: 1
    Z: 1
    Hstride:   %(pHstride)s
    Wstride:   %(pWstride)s
    Hdilation: %(pHdilation)s
    Wdilation: %(pWdilation)s
  shape:
    name: conv2D
    dimensions: [C, M, R, S, N, P, Q, X, Y, Z, G]
    coefficients:
    - {name: Wstride,   default: 1}
    - {name: Hstride,   default: 1}
    - {name: Wdilation, default: 1}
    - {name: Hdilation, default: 1}
    data_spaces:
    - name: Weights
      projection: [[[C]], [[M]], [[R]], [[S]], [[Y]], [[G]]]
    - name: Inputs
      projection:
      - [[N]]
      - [[C]]
      - [[R, Wdilation], [P, Wstride]]
      - [[S, Hdilation], [Q, Hstride]]
      - [[X]]
      - [[G]]
    - name: Outputs
      projection: [[[N]], [[M]], [[Q]], [[P]], [[Z]], [[G]]]
      read_write: True
"""


def _render_top_yaml(
    spec: HardwareSpec,
    loopdim: Dict[str, int],
    objective: str,
    *,
    mapper_search_size: int = DEFAULT_MAPPER_SEARCH_SIZE,
    mapper_victory_condition: int = DEFAULT_MAPPER_VICTORY_CONDITION,
    mapper_timeout: int = DEFAULT_MAPPER_TIMEOUT,
    mapper_num_threads: int = 4,
) -> str:
    tvars = _derive_template_vars(spec, global_cycle_s=1e-9)
    instance = _loopdim_to_instance(loopdim)
    metrics = _OBJECTIVE_TO_METRICS[objective]
    substitutions = {
        **tvars,
        "OPTIMIZATION_METRICS": ", ".join(metrics),
        "MAPPER_SEARCH_SIZE": int(mapper_search_size),
        "MAPPER_VICTORY_CONDITION": int(mapper_victory_condition),
        "MAPPER_TIMEOUT": int(mapper_timeout),
        "MAPPER_NUM_THREADS": int(mapper_num_threads),
        "pC": instance["C"], "pM": instance["M"], "pG": instance["G"],
        "pR": instance["R"], "pS": instance["S"],
        "pP": instance["P"], "pQ": instance["Q"],
        "pH": instance["H"], "pW": instance["W_img"],
        "pHstride": instance["Hstride"], "pWstride": instance["Wstride"],
        "pHdilation": instance["Hdilation"], "pWdilation": instance["Wdilation"],
    }
    return _TOP_TEMPLATE % substitutions



def feasibility_precheck(spec: HardwareSpec, loopdim: Dict[str, int]) -> Optional[str]:
    dram = _storage_level_by_name(spec, "Dram")
    cap_bits = int(dram.size_bits)
    prec = spec.macro.precision
    C = int(loopdim.get("C", 1))
    K = int(loopdim.get("K", 1))
    R = int(loopdim.get("R", 1))
    S = int(loopdim.get("S", 1))
    G = int(max(1, loopdim.get("G", 1)))
    P = int(loopdim.get("P", 1))
    Q = int(loopdim.get("Q", 1))
    H = int(loopdim.get("H", 1))
    W = int(loopdim.get("W", 1))
    weight_bits = C * K * R * S * G * int(prec.W)
    input_bits = C * H * W * G * int(prec.I)
    output_bits = K * P * Q * G * int(prec.O_final)
    for name, val in (("weights", weight_bits), ("inputs", input_bits), ("outputs", output_bits)):
        if val > cap_bits:
            return (f"layer {name} tensor ({val} bits) exceeds DRAM capacity "
                    f"({cap_bits} bits); physically infeasible, not a mapper issue")
    return None



def loopdim_fingerprint(loopdim: Dict[str, int]) -> str:
    canon_keys = (
        "R", "S", "P", "Q", "C", "K", "G",
        "H", "W", "Stride", "Padding", "Dilation", "B",
    )
    def _coerce(v):
        return 1 if v is None else int(v)
    canon = {k: _coerce(loopdim.get(k)) for k in canon_keys}
    raw = json.dumps(canon, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]



def run_cimloop_mapper_for_layer(
    spec: HardwareSpec,
    loopdim: Dict[str, int],
    objective: str,
    work_dir: Path,
    *,
    mapper_options: Optional[Dict[str, int]] = None,
) -> CIMLoopLayerOutput:
    import timeloopfe.v4 as tl

    if objective not in _OBJECTIVE_TO_METRICS:
        raise ValueError(f"unsupported objective {objective!r}; allowed: {list(_OBJECTIVE_TO_METRICS)}")

    reason = feasibility_precheck(spec, loopdim)
    if reason is not None:
        raise RuntimeError(f"CIMLoop baseline feasibility_precheck: {reason}")

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    plugin_root = _ACCELERGY_SOURCE_ROOT.parent
    plugin_dirs = [plugin_root/name for name in (
        "accelergy-cacti-plug-in", "accelergy-aladdin-plug-in", "accelergy-neurosim-plug-in")]
    if not all(path.is_dir() for path in plugin_dirs):
        raise RuntimeError("Required experiment plugin sources are missing")
    accelergy_config = {"version": "0.4", "estimator_plug_ins": [str(x) for x in plugin_dirs],
                        "primitive_components": [], "compound_components": [],
                        "math_functions": [], "python_plug_ins": [], "table_plug_ins": {"roots": []}}
    config_path = work_dir/"accelergy_config.yaml"
    rendered_config = json.dumps(accelergy_config, indent=2)+"\n"
    if config_path.exists() and config_path.read_text() != rendered_config:
        raise ValueError("Existing trial Accelergy config differs; use a fresh directory")
    config_path.write_text(rendered_config)

    layer_fp = loopdim_fingerprint(loopdim)

    options = dict(mapper_options or {})
    allowed = {"mapper_search_size", "mapper_victory_condition", "mapper_timeout", "mapper_num_threads"}
    if set(options)-allowed or any(not isinstance(v, int) or v < 1 for v in options.values()):
        raise ValueError("Mapper options must be positive integer search_size/victory_condition/timeout limits")
    top_path = work_dir / "top.yaml.jinja2"
    top_path.write_text(_render_top_yaml(spec, loopdim, objective, **options))

    ArrayProcessor = _load_array_processor()
    with _timeloop_runtime_env():
        tl_spec = tl.Specification.from_yaml_files(
            str(top_path),
            processors=[ArrayProcessor],
        )

        t0 = time.time()
        log_path = work_dir / "timeloop-mapper.log"
        with open(log_path, "w") as log_file:
            tl.call_mapper(
                specification=tl_spec,
                output_dir=str(work_dir),
                log_to=log_file,
            )
        runtime_s = time.time() - t0

    map_yaml = work_dir / "timeloop-mapper.map.yaml"
    map_txt = work_dir / "timeloop-mapper.map.txt"
    art_summary = work_dir / "timeloop-mapper.ART_summary.yaml"
    if not map_yaml.is_file():
        raise RuntimeError(f"timeloop-mapper did not emit {map_yaml}; see {log_path}")

    axis_sources = [
        dict(ax.source_memory_per_operand) for ax in spec.macro.spatial_axes
    ]

    return CIMLoopLayerOutput(
        layer_fp=layer_fp,
        loopdim=dict(loopdim),
        objective=objective,
        optimization_metric=_OBJECTIVE_TO_METRICS[objective],
        map_yaml_path=map_yaml,
        map_txt_path=map_txt,
        storage_level_names=list(_MIREDO_STORAGE_INNER_TO_OUTER),
        axis_sources=axis_sources,
        axis_to_fanout=list(_MIREDO_AXIS_TO_FANOUT),
        runtime_s=runtime_s,
        art_summary_path=art_summary if art_summary.is_file() else None,
    )

def supports_loopdim(loopdim: Dict[str, int]) -> Optional[str]:
    return None


def run_for_layer(acc, ops, loopdim, model_name, architecture, objective,
                  *, work_dir, mapper_options=None):
    from Simulator.Simulax import tranSimulator
    from utils.Workload import LoopNest
    from Evaluation.CIMLoop.CompatibleCIMLoop import convert_CIMLoop_to_MIREDO
    from Evaluation.Zigzag_imc.CompatibleZigzag import pin_macro_weight_double_flag
    from Evaluation.common.BaselineProvider import BaselineRunResult
    from Evaluation.common.EvalCommon import resolve_spec

    spec = resolve_spec(acc, architecture)
    out = run_cimloop_mapper_for_layer(spec, loopdim, objective, Path(work_dir),
                                      mapper_options=mapper_options)
    loops, legalization = convert_CIMLoop_to_MIREDO(
        loops=LoopNest(acc=acc, ops=ops), out=out, spec=spec, loopdim=loopdim)
    pin_macro_weight_double_flag(loops)
    simulator = tranSimulator(acc=copy.deepcopy(acc), ops=ops, dataflow=loops)
    latency, energy = simulator.run_analytical()
    return BaselineRunResult(
        method="cimloop", objective=objective, latency=latency, energy=energy,
        profile=simulator.PD, dataflow=loops,
        metadata={"model": model_name, "architecture": architecture,
                  "optimization_metric": list(out.optimization_metric),
                  "mapper_wall_sec": out.runtime_s,
                  "mapper_options": dict(mapper_options or {}), **legalization},
    )
