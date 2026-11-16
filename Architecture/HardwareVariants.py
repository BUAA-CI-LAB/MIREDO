from __future__ import annotations

from dataclasses import replace
from typing import List

from Architecture.HardwareSpec import HardwareSpec, MemoryLevelSpec
from utils.Cacti_wrapper.EvalCacti import cacti_power, dram_static


def _mem_index(spec: HardwareSpec, name: str) -> int:
    for i, m in enumerate(spec.memory_hierarchy):
        if m.name == name:
            return i
    raise KeyError(f"Memory level {name!r} not in spec")


def _recompute_memory_cost_pJ(spec: HardwareSpec, mem_name: str) -> MemoryLevelSpec:
    level = spec.memory_by_name(mem_name)
    size_bytes = max(1, int(level.size_bits / 8))
    bw = max(8, int(level.r_bw_bits_per_cycle))
    if mem_name == "Dram":
        r_pb = 7.91
        w_pb = 7.91
    else:
        read_pj, write_pj, _, _ = cacti_power(
            tech_node=spec.macro.tech_params.tech_node,
            capacity_bytes=size_bytes,
            bitwidth_bits=bw,
        )
        r_pb = read_pj / bw
        w_pb = write_pj / bw
    return replace(level, r_cost_per_bit_pJ=r_pb, w_cost_per_bit_pJ=w_pb)


def _leakage_per_cycle_nJ(spec: HardwareSpec) -> float:
    tech = spec.macro.tech_params.tech_node

    def _sram_leak_pJ(mem_name: str) -> float:
        level = spec.memory_by_name(mem_name)
        _, _, leak_pJ, _ = cacti_power(
            tech_node=tech,
            capacity_bytes=max(1, int(level.size_bits / 8)),
            bitwidth_bits=max(8, int(level.r_bw_bits_per_cycle)),
        )
        return leak_pJ

    dram = spec.memory_by_name("Dram")
    _, dram_leak_pJ = dram_static(capacity_bytes=dram.size_bits / 8, bus_width_bits=dram.r_bw_bits_per_cycle)
    total_pJ = dram_leak_pJ

    total_pJ += _sram_leak_pJ("Global_buffer")

    total_pJ += _sram_leak_pJ("Output_buffer") * spec.cores
    total_pJ += _sram_leak_pJ("Input_buffer") * spec.cores

    macro = spec.memory_by_name("Macro")
    W = spec.macro.precision.W
    _, _, macro_leak_pJ, _ = cacti_power(
        tech_node=tech,
        capacity_bytes=(macro.size_bits / 8) * spec.macro.dimX * spec.macro.dimY,
        bitwidth_bits=max(8, spec.macro.dimY * W),
    )
    total_pJ += macro_leak_pJ * spec.cores

    return total_pJ * 1e-3


def _replace_memory_level(spec: HardwareSpec, mem_name: str, **updates) -> HardwareSpec:
    new_mems = []
    for m in spec.memory_hierarchy:
        if m.name == mem_name:
            new_mems.append(replace(m, **updates))
        else:
            new_mems.append(m)
    return replace(spec, memory_hierarchy=new_mems)


def _on_chip_buffer_names(spec: HardwareSpec) -> List[str]:
    return ["Global_buffer", "Output_buffer", "Input_buffer"]


def set_core_count(spec: HardwareSpec, count: int) -> HardwareSpec:
    count = max(1, int(count))
    new_axes = []
    for a in spec.macro.spatial_axes:
        if a.name == "cores":
            new_axes.append(replace(a, size=count))
        else:
            new_axes.append(a)
    new_macro = replace(spec.macro, spatial_axes=new_axes)
    new_spec = replace(spec, cores=count, macro=new_macro)
    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def set_gbuf_capacity_kb(spec: HardwareSpec, gbuf_kb: int) -> HardwareSpec:
    gbuf_kb = max(1, int(gbuf_kb))
    target_gbuf_bits = gbuf_kb * 8 * 1024
    new_spec = _replace_memory_level(spec, "Global_buffer", size_bits=target_gbuf_bits)
    _gb = _recompute_memory_cost_pJ(new_spec, "Global_buffer")
    new_spec = _replace_memory_level(
        new_spec, "Global_buffer",
        r_cost_per_bit_pJ=_gb.r_cost_per_bit_pJ,
        w_cost_per_bit_pJ=_gb.w_cost_per_bit_pJ,
    )
    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def set_gbuf_bandwidth(spec: HardwareSpec, bw_bits_per_cycle: int) -> HardwareSpec:
    bw = max(8, int(bw_bits_per_cycle))
    old = spec.memory_by_name("Global_buffer")
    new_spec = _replace_memory_level(
        spec, "Global_buffer",
        r_bw_bits_per_cycle=bw,
        w_bw_bits_per_cycle=bw,
    )
    updated = _recompute_memory_cost_pJ(new_spec, "Global_buffer")
    new_spec = _replace_memory_level(
        new_spec, "Global_buffer",
        r_cost_per_bit_pJ=updated.r_cost_per_bit_pJ,
        w_cost_per_bit_pJ=updated.w_cost_per_bit_pJ,
    )
    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def set_dram_bandwidth(spec: HardwareSpec, bw_bits_per_cycle: int) -> HardwareSpec:
    bw = max(8, int(bw_bits_per_cycle))
    new_spec = _replace_memory_level(
        spec, "Dram",
        r_bw_bits_per_cycle=bw,
        w_bw_bits_per_cycle=bw,
    )
    return new_spec


def set_operand_scratchpad_kb(spec: HardwareSpec, ibuf_kb: int) -> HardwareSpec:
    ibuf_kb = max(1, int(ibuf_kb))
    obuf_kb = ibuf_kb * 2

    new_spec = _replace_memory_level(spec, "Input_buffer", size_bits=ibuf_kb * 8 * 1024)
    new_spec = _replace_memory_level(new_spec, "Output_buffer", size_bits=obuf_kb * 8 * 1024)

    for name in ("Input_buffer", "Output_buffer"):
        updated = _recompute_memory_cost_pJ(new_spec, name)
        new_spec = _replace_memory_level(
            new_spec, name,
            r_cost_per_bit_pJ=updated.r_cost_per_bit_pJ,
            w_cost_per_bit_pJ=updated.w_cost_per_bit_pJ,
        )

    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def _set_macro_axis_size(spec: HardwareSpec, axis_name: str, size: int) -> HardwareSpec:
    size = max(1, int(size))
    new_axes = []
    for a in spec.macro.spatial_axes:
        if a.name == axis_name:
            new_axes.append(replace(a, size=size))
        else:
            new_axes.append(a)
    macro_kw = {"spatial_axes": new_axes}
    if axis_name in ("dimX", "dimY"):
        macro_kw[axis_name] = size
    new_macro = replace(spec.macro, **macro_kw)
    return replace(spec, macro=new_macro)


def set_macro_dimX(spec: HardwareSpec, dimX: int) -> HardwareSpec:
    new_spec = _set_macro_axis_size(spec, "dimX", dimX)
    return replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))


def set_macro_dimY(spec: HardwareSpec, dimY: int) -> HardwareSpec:
    new_spec = _set_macro_axis_size(spec, "dimY", dimY)
    return replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))


def set_macro_spec(spec: HardwareSpec, dimX: int, dimY: int, depth: int) -> HardwareSpec:
    new_spec = _set_macro_axis_size(spec, "dimX", dimX)
    new_spec = _set_macro_axis_size(new_spec, "dimY", dimY)
    return set_compartment_depth(new_spec, depth)


def set_compartment_depth(spec: HardwareSpec, depth: int) -> HardwareSpec:
    depth = int(depth)
    if depth < 1:
        raise ValueError(f"compartment_depth ({depth}) must be >= 1.")
    W = spec.macro.precision.W
    new_macro_bits = depth * W
    old_macro = spec.memory_by_name("Macro")
    old_depth = max(1, spec.macro.compartment_depth)

    new_macro = replace(spec.macro, compartment_depth=depth)
    new_spec = replace(spec, macro=new_macro)
    new_spec = _replace_memory_level(new_spec, "Macro", size_bits=new_macro_bits)

    scale = depth / old_depth
    new_w_cost = old_macro.w_cost_per_bit_pJ * scale
    new_spec = _replace_memory_level(
        new_spec, "Macro",
        w_cost_per_bit_pJ=new_w_cost,
    )

    new_spec = replace(new_spec, leakage_per_cycle_nJ=_leakage_per_cycle_nJ(new_spec))
    return new_spec


def set_macro_weight_double_buffer(spec: HardwareSpec, enabled) -> HardwareSpec:
    flag = str(enabled).strip().lower() in ("1", "true", "yes", "on")
    new_macro = replace(spec.macro, weight_double_buffer=flag)
    return replace(spec, macro=new_macro)


def build_hardware_variant(spec: HardwareSpec, parameter: str, value) -> HardwareSpec:
    if parameter == "macro_weight_double_buffer":
        return set_macro_weight_double_buffer(spec, value)
    if parameter == "core_count":
        return set_core_count(spec, int(value))
    if parameter == "buffer_capacity":
        return set_gbuf_capacity_kb(spec, int(value))
    if parameter == "gbuf_core_bw":
        return set_gbuf_bandwidth(spec, int(value))
    if parameter == "compartment_depth":
        return set_compartment_depth(spec, int(value))
    if parameter == "macro_spec":
        if not (isinstance(value, (tuple, list)) and len(value) == 4):
            raise ValueError(
                f"macro_spec value must be (name, D1, D2, D3) 4-tuple, got: {value}"
            )
        _, dimX, dimY, depth = value
        return set_macro_spec(spec, int(dimX), int(dimY), int(depth))
    if parameter == "dram_bw":
        return set_dram_bandwidth(spec, int(value))
    if parameter == "operand_scratchpad":
        return set_operand_scratchpad_kb(spec, int(value))
    raise ValueError(f"Unsupported sensitivity parameter: {parameter}")
