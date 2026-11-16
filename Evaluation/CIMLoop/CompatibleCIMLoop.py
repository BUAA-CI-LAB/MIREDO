from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from utils.ClassMapping import ClassMapping
from utils.Workload import LoopNest
from Evaluation.Zigzag_imc.CompatibleZigzag import (
    convert_baseMapping_to_MIREDO,
    project_replay_safe_double_flag,
)
from Evaluation.CIMLoop.cimloop_adapter import CIMLoopLayerOutput


_TIMELOOP_TO_MIREDO_DIM = {
    "R": "R", "S": "S", "P": "P", "Q": "Q",
    "C": "C", "M": "K", "G": "G",
}
_IGNORED_DIMS = {"X", "Y", "Z", "N"}

_OPERANDS_MIREDO = ("I", "W", "O")
_TIMELOOP_DS_TO_OP = {"Inputs": "I", "Weights": "W", "Outputs": "O"}


def _parse_factor_string(factors_str: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for token in str(factors_str).split():
        m = re.match(r"^([A-Za-z]+)(\d+)$", token)
        if not m:
            raise ValueError(f"cannot parse factor token {token!r} in {factors_str!r}")
        out[m.group(1)] = int(m.group(2))
    return out


def _factors_in_permutation_order(
    factor_map: Dict[str, int],
    permutation: str,
    dim_filter: List[str] | None = None,
) -> List[Tuple[str, int]]:
    result: List[Tuple[str, int]] = []
    for ch in permutation:
        if dim_filter is not None and ch not in dim_filter:
            continue
        if ch in _IGNORED_DIMS:
            continue
        miredo_dim = _TIMELOOP_TO_MIREDO_DIM.get(ch)
        if miredo_dim is None:
            continue
        factor = int(factor_map.get(ch, 1))
        if factor > 1:
            result.append((miredo_dim, factor))
    return result


def _load_raw_entries(map_yaml_path: Path) -> List[Dict[str, Any]]:
    with open(map_yaml_path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "mapping" not in data:
        raise ValueError(f"{map_yaml_path}: expected top-level 'mapping' key")
    entries = data["mapping"]
    if not isinstance(entries, list):
        raise ValueError(f"{map_yaml_path}: 'mapping' must be a list")
    return entries


def _group_entries_by_target(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for e in entries:
        tgt = e.get("target")
        typ = e.get("type")
        if tgt is None or typ is None:
            raise ValueError(f"bad mapping entry (missing target/type): {e}")
        grouped.setdefault(tgt, {})[typ] = e
    return grouped


def _next_outer_keep_level(per_level: List[Dict[str, Any]], bypass_idx: int, op: str) -> str | None:
    for i in range(bypass_idx + 1, len(per_level)):
        if op in per_level[i]["keep"]:
            return per_level[i]["name"]
    return None


def _next_outer_keep_index(per_level: List[Dict[str, Any]], bypass_idx: int, op: str) -> int | None:
    for i in range(bypass_idx + 1, len(per_level)):
        if op in per_level[i]["keep"]:
            return i
    return None


def baseMapping_from_cimloop_output(out: CIMLoopLayerOutput) -> ClassMapping:
    entries = _load_raw_entries(out.map_yaml_path)
    grouped = _group_entries_by_target(entries)

    inner_to_outer = list(out.storage_level_names)
    name_to_inner_idx = {name: i for i, name in enumerate(inner_to_outer)}

    per_level: List[Dict[str, Any]] = []
    for name in inner_to_outer:
        blk = grouped.get(name, {})
        dt = blk.get("datatype") or {}
        keep_ops = {_TIMELOOP_DS_TO_OP[d] for d in dt.get("keep", []) if d in _TIMELOOP_DS_TO_OP}

        temporal_entry = blk.get("temporal")
        if temporal_entry is not None:
            t_factors = _parse_factor_string(temporal_entry.get("factors", ""))
            t_perm = str(temporal_entry.get("permutation", ""))
            temporal_parsed = _factors_in_permutation_order(t_factors, t_perm)
        else:
            temporal_parsed = []

        per_level.append({
            "name": name,
            "keep": keep_ops,
            "temporal": temporal_parsed,
        })

    fanout_spatial: Dict[str, Dict[str, List[Tuple[str, int]]]] = {}
    for e in entries:
        if e.get("type") != "spatial":
            continue
        tgt = str(e.get("target", ""))
        if not tgt.startswith("inter_") or not tgt.endswith("_spatial"):
            continue
        container_name = tgt[len("inter_"):-len("_spatial")]
        factor_map = _parse_factor_string(e.get("factors", ""))
        perm = str(e.get("permutation", ""))
        split = int(e.get("split", len(perm)))
        space_x_chars = perm[:split]
        space_y_chars = perm[split:]
        fanout_spatial[container_name] = {
            "X": _factors_in_permutation_order(factor_map, space_x_chars),
            "Y": _factors_in_permutation_order(factor_map, space_y_chars),
        }

    axis_factors: List[List[Tuple[str, int]]] = []
    for container_name, side in out.axis_to_fanout:
        axis_factors.append(fanout_spatial.get(container_name, {}).get(side, []))

    temporal_mapping: Dict[str, List[List[Tuple[str, int]]]] = {op: [] for op in _OPERANDS_MIREDO}
    spatial_mapping: Dict[str, List[List[Tuple[str, int]]]] = {op: [] for op in _OPERANDS_MIREDO}
    op_level_index: Dict[str, Dict[str, int]] = {op: {} for op in _OPERANDS_MIREDO}
    op_keep_mask: Dict[str, List[bool]] = {op: [op in lv["keep"] for lv in per_level] for op in _OPERANDS_MIREDO}

    for op in _OPERANDS_MIREDO:
        kept_level_names: List[str] = []
        for lv in per_level:
            if op in lv["keep"]:
                op_level_index[op][lv["name"]] = len(temporal_mapping[op])
                temporal_mapping[op].append([])
                spatial_mapping[op].append([])
                kept_level_names.append(lv["name"])

    for lv_idx, lv in enumerate(per_level):
        factors = lv["temporal"]
        if not factors:
            continue
        for op in _OPERANDS_MIREDO:
            if op in lv["keep"]:
                target_name = lv["name"]
            else:
                target_name = _next_outer_keep_level(per_level, lv_idx, op)
            if target_name is None:
                raise ValueError(
                    f"CIMLoop map.yaml: operand '{op}' has loop(s) {factors} at bypass "
                    f"level '{lv['name']}' with no outer keep level — cannot honour without "
                    f"triggering tranSimulator fallback-to-innermost (tile inflation). "
                    f"Check that HardwareSpec keeps {op} at DRAM."
                )
            temporal_mapping[op][op_level_index[op][target_name]].extend(factors)

    for axis_idx, factors in enumerate(axis_factors):
        if not factors:
            continue
        per_op_source = out.axis_sources[axis_idx]
        for op in _OPERANDS_MIREDO:
            source_name = per_op_source.get(op)
            if source_name is None:
                continue
            idx = op_level_index[op].get(source_name)
            if idx is None:
                continue
            spatial_mapping[op][idx].extend(factors)

    loop_dim = {
        "R": int(out.loopdim.get("R", 1)),
        "S": int(out.loopdim.get("S", 1)),
        "P": int(out.loopdim.get("P", 1)),
        "Q": int(out.loopdim.get("Q", 1)),
        "C": int(out.loopdim.get("C", 1)),
        "K": int(out.loopdim.get("K", 1)),
        "G": int(out.loopdim.get("G", 1)),
    }

    double_buffer_flag = {op: [False] * len(temporal_mapping[op]) for op in _OPERANDS_MIREDO}

    return ClassMapping(
        source="cimloop",
        loop_dim=loop_dim,
        temporal_mapping=temporal_mapping,
        spatial_mapping=spatial_mapping,
        double_buffer_flag=double_buffer_flag,
        top_r_loop_size=None,
        raw=out,
    )


def _apply_native_double_buffer(loops: LoopNest) -> None:
    acc = loops.acc
    nrows = len(loops.usr_defined_double_flag)
    flag = [[0, 0, 0] for _ in range(nrows)]
    for m in range(1, min(nrows, acc.Num_mem)):
        for op in range(3):
            flag[m][op] = int(acc.double_config[m][op])
    loops.usr_defined_double_flag = project_replay_safe_double_flag(loops, flag)


def convert_CIMLoop_to_MIREDO(loops: LoopNest, out: CIMLoopLayerOutput,
                              spec, loopdim):
    from Evaluation.common.CapacityLegalization import legalize_capacity
    base_mapping = baseMapping_from_cimloop_output(out)
    demotions = legalize_capacity(base_mapping.temporal_mapping,
                                  base_mapping.spatial_mapping, spec, loopdim)
    loops = convert_baseMapping_to_MIREDO(loops=loops, baseMapping=base_mapping)
    _apply_native_double_buffer(loops)
    return loops, {"capacity_demoted_count": len(demotions), "capacity_demoted": demotions}
