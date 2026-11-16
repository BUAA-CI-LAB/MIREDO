from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.ZigzagUtils import ensure_zigzag_submodule_on_path, zigzag_submodule_root

ensure_zigzag_submodule_on_path()
from zigzag.classes.hardware.architecture.accelerator import Accelerator
from zigzag.classes.hardware.architecture.core import Core
from zigzag.classes.hardware.architecture.ImcArray import ImcArray
from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance

from Architecture.HardwareSpec import HardwareSpec, MemoryLevelSpec


_OPERAND_SPEC_TO_ZIGZAG = {"I": "I1", "W": "I2", "O": "O"}


@contextmanager
def _zigzag_runtime_cwd():
    prev_cwd = os.getcwd()
    os.chdir(zigzag_submodule_root())
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def _build_tech_param(spec: HardwareSpec) -> Dict[str, float]:
    tp = spec.macro.tech_params
    return {
        "tech_node": tp.tech_node,
        "vdd": tp.vdd,
        "nd2_cap": tp.nd2_cap,
        "xor2_cap": tp.xor2_cap,
        "dff_cap": tp.dff_cap,
        "nd2_area": tp.nd2_area,
        "xor2_area": tp.xor2_area,
        "dff_area": tp.dff_area,
        "nd2_dly": tp.nd2_dly,
        "xor2_dly": tp.xor2_dly,
    }


def _build_hd_param(spec: HardwareSpec) -> Dict[str, Any]:
    macro_level = spec.memory_by_name("Macro")
    group_depth = macro_level.size_bits // spec.macro.precision.W
    return {
        "pe_type": "in_sram_computing",
        "imc_type": "digital",
        "input_precision": spec.macro.precision.I,
        "weight_precision": spec.macro.precision.W,
        "input_bit_per_cycle": spec.macro.input_bit_per_cycle,
        "group_depth": group_depth,
        "wordline_dimension": "D1",
        "bitline_dimension": "D2",
        "enable_cacti": True,
    }


def _build_dimensions(spec: HardwareSpec) -> Dict[str, int]:
    return {
        "D1": spec.macro.dimY,
        "D2": spec.macro.dimX,
        "D3": spec.cores,
    }


def _port_alloc_for_operand(op_spec: str) -> Dict[str, Any]:
    if op_spec in ("I", "W"):
        return {"fh": "rw_port_1", "tl": "rw_port_1", "fl": None, "th": None}
    if op_spec == "O":
        return {"fh": "rw_port_1", "tl": "rw_port_1", "fl": "rw_port_1", "th": "rw_port_1"}
    raise ValueError(f"Unknown operand {op_spec!r}")


def _build_memory_instance(level: MemoryLevelSpec) -> MemoryInstance:
    kwargs: Dict[str, Any] = dict(
        name=level.name,
        size=level.size_bits,
        r_bw=level.r_bw_bits_per_cycle,
        w_bw=level.w_bw_bits_per_cycle,
        r_cost=level.r_cost_per_bit_pJ * level.r_bw_bits_per_cycle,
        w_cost=level.w_cost_per_bit_pJ * level.w_bw_bits_per_cycle,
        area=level.area_mm2,
        r_port=level.ports.r,
        w_port=level.ports.w,
        rw_port=level.ports.rw,
        latency=level.r_latency_cycles,
    )
    if level.min_r_granularity_bits is not None:
        kwargs["min_r_granularity"] = level.min_r_granularity_bits
    if level.min_w_granularity_bits is not None:
        kwargs["min_w_granularity"] = level.min_w_granularity_bits
    return MemoryInstance(**kwargs)


def _served_dimensions_for_zigzag(level: MemoryLevelSpec):
    sd = level.served_dimensions_zigzag
    if isinstance(sd, str):
        return sd
    if not sd:
        return set()
    return {tuple(t) for t in sd}


def to_zigzag_accelerator(spec: HardwareSpec, acc_name: str = "CIM_ACC_TEMPLATE") -> Accelerator:
    tech_param = _build_tech_param(spec)
    hd_param = _build_hd_param(spec)
    dimensions = _build_dimensions(spec)

    with _zigzag_runtime_cwd():
        imc_array = ImcArray(tech_param, hd_param, dimensions)

    mem_hierarchy = MemoryHierarchy(operational_array=imc_array)

    for level in reversed(spec.memory_hierarchy):
        instance = _build_memory_instance(level)
        operands_zz = tuple(_OPERAND_SPEC_TO_ZIGZAG[op] for op in level.operands)
        port_alloc = tuple(_port_alloc_for_operand(op) for op in level.operands)
        served = _served_dimensions_for_zigzag(level)
        mem_hierarchy.add_memory(
            memory_instance=instance,
            operands=operands_zz,
            port_alloc=port_alloc,
            served_dimensions=served,
        )

    core = Core(0, imc_array, mem_hierarchy)
    return Accelerator(acc_name, {core})

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


def loopdim_to_zigzag_workload(
    loopdim: Dict[str, int],
    layer_id: int = 0,
    spec: Optional[HardwareSpec] = None,
) -> dict:
    h_val = int(loopdim.get('H', loopdim.get('P', 1)) or 1)
    w_val = int(loopdim.get('W', loopdim.get('Q', 1)) or 1)
    stride = int(loopdim.get('Stride', 1))
    padding = int(loopdim.get('Padding', 0))
    if spec is not None:
        prec = spec.macro.precision
        operand_precision = {
            "O": int(prec.psum),
            "O_final": int(prec.O_final),
            "W": int(prec.W),
            "I": int(prec.I),
        }
    else:
        operand_precision = {"O": 16, "O_final": 8, "W": 8, "I": 8}
    return {layer_id: {
        "operator_type": "Conv",
        "equation": "O[b][g][k][oy][ox]+=W[g][k][c][fy][fx]*I[b][g][c][iy][ix]",
        "loop_dim_size": {
            "B":  int(loopdim.get("B", 1) or 1),
            "K":  int(loopdim["K"]),
            "G":  int(loopdim.get("G", 1) or 1),
            "OX": int(loopdim["Q"]),
            "OY": int(loopdim["P"]),
            "C":  int(loopdim["C"]),
            "FX": int(loopdim["S"]),
            "FY": int(loopdim["R"]),
        },
        "pr_loop_dim_size": {"IX": w_val, "IY": h_val},
        "dimension_relations": [
            f"ix={stride}*ox+1*fx",
            f"iy={stride}*oy+1*fy",
        ],
        "operand_precision":   operand_precision,
        "constant_operands":   ["W"],
        "operand_source":      {"I": []},
        "padding": {
            "IY": (padding, padding),
            "IX": (padding, padding),
        },
    }}


def run_zigzag_mapper_for_layer(
    spec,
    loopdim: Dict[str, int],
    objective: str,
    work_dir: Path,
    *,
    loma_lpf_limit: int = 6,
):
    from Evaluation.common.EvalCommon import objective_to_opt_flag

    work_dir = Path(work_dir).resolve()
    if not isinstance(loma_lpf_limit, int) or loma_lpf_limit < 1:
        raise ValueError("loma_lpf_limit must be a positive integer")
    accelerator = to_zigzag_accelerator(spec, acc_name="CIM_ACC_TEMPLATE")
    workload = loopdim_to_zigzag_workload(loopdim, layer_id=0, spec=spec)
    mapping = {
        "default": {
            "core_allocation": 0,
            "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
            "spatial_mapping_hint": {"D1": ["K"], "D2": ["FX", "FY", "C"], "D3": ["G", "K", "OX", "OY"]},
        }
    }
    opt_flag = objective_to_opt_flag(objective)
    work_dir.mkdir(parents=True, exist_ok=True)
    dump_pattern = str(work_dir / "layer_{layer}.json")
    pickle_filename = str(work_dir / "raw_cmes.pickle")

    ensure_zigzag_submodule_on_path()
    from zigzag.classes.stages import (
        AcceleratorParserStage,
        CompleteSaveStage,
        CostModelStage,
        LomaStage,
        MainStage,
        MinimalEDPStage,
        MinimalEnergyStage,
        MinimalLatencyStage,
        PickleSaveStage,
        SimpleSaveStage,
        SpatialMappingGeneratorStage,
        SumStage,
        WorkloadParserStage,
        WorkloadStage,
    )
    if opt_flag == "energy":
        opt_stage = MinimalEnergyStage
    elif opt_flag == "latency":
        opt_stage = MinimalLatencyStage
    elif opt_flag == "EDP":
        opt_stage = MinimalEDPStage
    else:
        raise NotImplementedError(f"Unsupported opt_flag: {opt_flag!r}")

    prev_level = logging.getLogger().level
    try:
        logging.getLogger().setLevel(logging.WARNING)
        mainstage = MainStage(
            [
                WorkloadParserStage,
                AcceleratorParserStage,
                SimpleSaveStage,
                PickleSaveStage,
                SumStage,
                WorkloadStage,
                CompleteSaveStage,
                opt_stage,
                SpatialMappingGeneratorStage,
                opt_stage,
                LomaStage,
                CostModelStage,
            ],
            accelerator=accelerator,
            workload=workload,
            mapping=mapping,
            dump_filename_pattern=dump_pattern,
            pickle_filename=pickle_filename,
            loma_lpf_limit=loma_lpf_limit,
            enable_mix_spatial_mapping_generation=True,
            maximize_hardware_utilization=True,
            enable_weight_diagonal_mapping=True,
            loma_show_progress_bar=False,
            access_same_data_considered_as_no_access=True,
        )
        cmes = list(mainstage.run())
    finally:
        logging.getLogger().setLevel(prev_level)

    cme = cmes[0][0]
    return cme


def supports_loopdim(loopdim: Dict[str, int]) -> Optional[str]:
    return None


def run_for_layer(acc, ops, loopdim, model_name, architecture, objective,
                  *, work_dir, loma_lpf_limit=6):
    from Simulator.Simulax import tranSimulator
    from utils.Workload import LoopNest
    from utils.ZigzagUtils import convert_Zigzag_to_MIREDO
    from Evaluation.Zigzag_imc.CompatibleZigzag import pin_macro_weight_double_flag
    from Evaluation.common.BaselineProvider import BaselineRunResult
    from Evaluation.common.EvalCommon import resolve_spec
    import time

    spec = resolve_spec(acc, architecture)
    started = time.perf_counter()
    cme = run_zigzag_mapper_for_layer(spec, loopdim, objective, Path(work_dir),
                                     loma_lpf_limit=loma_lpf_limit)
    mapper_seconds = time.perf_counter() - started
    loops = convert_Zigzag_to_MIREDO(LoopNest(acc=acc, ops=ops), cme=cme)
    pin_macro_weight_double_flag(loops)
    simulator = tranSimulator(acc=copy.deepcopy(acc), ops=ops, dataflow=loops)
    latency, energy = simulator.run()
    return BaselineRunResult(
        method="zigzag", objective=objective, latency=latency, energy=energy,
        profile=simulator.PD, dataflow=loops,
        metadata={"model": model_name, "architecture": architecture,
                  "mapper_wall_sec": mapper_seconds, "loma_lpf_limit": loma_lpf_limit},
    )
