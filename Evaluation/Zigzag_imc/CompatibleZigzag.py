from utils.Workload import WorkLoad, LoopNest, Mapping
from Architecture.ArchSpec import CIM_Acc
import numpy as np
import copy
import math
from typing import Callable, Dict, List, Tuple
from collections import defaultdict
from utils.ClassMapping import ClassMapping
from utils.GlobalUT import Logger


def align_mapping_levels_to_acc(
    mapping_dict: Dict[str, List[List[Tuple[str, int]]]],
    mapping_array: List[List[int]],
    *,
    source_tag: str,
    array_row_order: Tuple[str, ...] = ("I", "W", "O"),
    warn: Callable[[str], None] | None = None,
) -> Dict[str, List[List[Tuple[str, int]]]]:
    aligned = copy.deepcopy(mapping_dict)
    for row_idx, op in enumerate(array_row_order):
        if op not in aligned:
            raise ValueError(f"{source_tag}: missing operand mapping '{op}'")

        allowed_levels = sum(1 for v in mapping_array[row_idx] if v == 1)
        cur_levels = len(aligned[op])
        if cur_levels > allowed_levels:
            raise ValueError(
                f"{source_tag}: operand '{op}' has {cur_levels} levels, "
                f"but accelerator allows only {allowed_levels}"
            )
        if cur_levels < allowed_levels:
            gap = allowed_levels - cur_levels
            aligned[op] = aligned[op] + [[] for _ in range(gap)]
            if warn is not None:
                warn(
                    f"{source_tag}: padded operand '{op}' with "
                    f"{gap} empty inner level(s) ({cur_levels} -> {allowed_levels})"
                )
    return aligned


def align_double_buffer_flags_to_acc(
    double_buffer_flag: Dict[str, List[bool]],
    mapping_array: List[List[int]],
    *,
    source_tag: str,
    array_row_order: Tuple[str, ...] = ("I", "W", "O"),
    warn: Callable[[str], None] | None = None,
) -> Dict[str, List[bool]]:
    aligned: Dict[str, List[bool]] = {}
    for row_idx, op in enumerate(array_row_order):
        raw = list(double_buffer_flag.get(op, []))
        allowed_levels = sum(1 for v in mapping_array[row_idx] if v == 1)

        if len(raw) == 0:
            prefix = False
            level_flags: List[bool] = []
        elif len(raw) == allowed_levels:
            prefix = False
            level_flags = [bool(x) for x in raw]
        else:
            prefix = bool(raw[0])
            level_flags = [bool(x) for x in raw[1:]]

        if len(level_flags) > allowed_levels:
            raise ValueError(
                f"{source_tag}: operand '{op}' has {len(level_flags)} double-buffer flags, "
                f"but accelerator allows only {allowed_levels}"
            )
        if len(level_flags) < allowed_levels:
            gap = allowed_levels - len(level_flags)
            level_flags.extend([False] * gap)
            if warn is not None:
                warn(
                    f"{source_tag}: padded operand '{op}' double-buffer flags with "
                    f"{gap} false level(s)"
                )

        aligned[op] = [prefix] + level_flags

    return aligned


def _pad_missing_inner_levels(levels, expected_len, fill_factory):
    if len(levels) > expected_len:
        raise ValueError(f"Too many levels: expected <= {expected_len}, got {len(levels)}")
    missing = expected_len - len(levels)
    if missing == 0:
        return copy.deepcopy(levels)
    return [fill_factory() for _ in range(missing)] + copy.deepcopy(levels)



def convert_ZZ_dflag_to_doubleflag(acc:CIM_Acc, ops:WorkLoad, cme_dflag):
    fixed_order = ['I', 'W', 'O']
    num_operations = len(fixed_order)

    dflag = {}
    for op in fixed_order:
        op_index = fixed_order.index(op)
        allowed_levels = sum(1 for level in range(acc.Num_mem) if acc.mappingArray[op_index][level] == 1)
        dflag[op] = _pad_missing_inner_levels(cme_dflag[op][1:], allowed_levels, lambda: False)

    for op_type in fixed_order:
        if op_type in dflag:
            op_index = fixed_order.index(op_type)
            allowed_levels = sum(1 for level in range(acc.Num_mem) if acc.mappingArray[op_index][level] == 1)
            if allowed_levels != len(dflag[op_type]):
                raise ValueError(f"Mismatch in levels count for operation '{op_type}': expected {allowed_levels}, got {len(dflag[op_type])}")

    double_tag = np.zeros((acc.Num_mem, num_operations), dtype=int)
    for op_type in fixed_order:
        if op_type in dflag:
            op_index = fixed_order.index(op_type)
            levels = dflag[op_type][::-1]
            level_counter = 0
            for level_index in range(acc.Num_mem):
                if acc.mappingArray[op_index][level_index] == 1:
                    double_tag[level_index][op_index] = int(levels[level_counter])
                    level_counter += 1
    macro_dflag = [ 0 for _ in range(num_operations)]

    double_tag = np.vstack([double_tag, macro_dflag])
    return double_tag

def process_top_r(ori_tm_dict, cme_top_r_loop):
    top_r_loop = {
        op: [int(round(x)) for x in cme_top_r_loop[op][1:]]
        for op in ['I', 'W', 'O']
    }

    new_tm_dict = copy.deepcopy(ori_tm_dict)
    for op in ['I', 'W', 'O']:
        layers = new_tm_dict[op]
        thresholds = top_r_loop[op]
        for level_idx in range(len(layers) - 1):
            threshold = thresholds[level_idx]
            if threshold <= 1 or len(layers[level_idx]) == 0:
                continue

            product = 1
            move_from = None
            for tuple_idx in range(len(layers[level_idx]) - 1, -1, -1):
                product *= int(round(layers[level_idx][tuple_idx][1]))
                if product == threshold:
                    move_from = tuple_idx
                    break
                if product > threshold:
                    break

            if move_from is None:
                continue

            promoted_tail = layers[level_idx][move_from:]
            layers[level_idx] = layers[level_idx][:move_from]
            layers[level_idx + 1] = promoted_tail + layers[level_idx + 1]

    return new_tm_dict


def _extract_top_r_loop_size(cme):
    if hasattr(cme.temporal_mapping, 'top_r_loop_size'):
        return cme.temporal_mapping.top_r_loop_size
    return cme.mapping.temporal_mapping.top_r_loop_size


def _calc_explicit_mem_usage(loops: LoopNest):
    acc = loops.acc
    ops = loops.ops
    loops.preprogress()

    dim_tp = [[[1 for _ in range(ops.Num_dim)] for _ in range(3)] for _ in range(acc.Num_mem)]
    dim_sp = [[[1 for _ in range(ops.Num_dim)] for _ in range(3)] for _ in range(acc.Num_mem)]

    for mem in range(1, acc.Num_mem):
        for op in range(3):
            for mapping in loops.tm:
                if mem <= mapping.mem[op] and acc.mappingArray[op][mem] == 1:
                    dim_tp[mem][op][mapping.dim] *= mapping.dimSize
            for mapping in loops.sm:
                if mem <= mapping.mem[op] and acc.mappingArray[op][mem] == 1:
                    dim_sp[mem][op][mapping.dim] *= mapping.dimSize

    data_size = {}
    tmp_dim = [0 for _ in range(ops.Num_dim)]
    for op in range(3):
        data_size[acc.Num_mem, op] = 1

    for mem in range(1, acc.Num_mem):
        op = 0
        for dim in range(1, ops.Num_dim):
            tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
        if acc.mappingArray[op][mem]:
            data_size[mem, op] = ops.get_operand_size(tmp_dim, op)
        else:
            data_size[mem, op] = 0

        op = 1
        for dim in range(1, ops.Num_dim):
            tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
        if acc.mappingArray[op][mem]:
            data_size[mem, op] = ops.get_operand_size(tmp_dim, op)
        else:
            data_size[mem, op] = 0

        op = 2
        for dim in range(1, ops.Num_dim):
            tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
        if acc.mappingArray[op][mem]:
            data_size[mem, op] = ops.get_operand_size(tmp_dim, op)
        else:
            data_size[mem, op] = 0

    for op in range(3):
        for mem in range(1, acc.Num_mem):
            if loops.bypassMem[mem][op]:
                data_size[mem, op] = 0

    raw_bits = np.zeros((acc.Num_mem + 1, 3), dtype=int)
    used_bits = np.zeros((acc.Num_mem + 1, 3), dtype=int)
    for mem in range(1, acc.Num_mem):
        for op in range(3):
            if acc.mappingArray[op][mem] == 0:
                continue
            if hasattr(loops, "get_precision"):
                precision = loops.get_precision(mem, op)
            else:
                precision = acc.precision[mem, op]
            raw_bits[mem][op] = int(data_size[mem, op] * precision)
            used_bits[mem][op] = raw_bits[mem][op] * (1 + int(loops.usr_defined_double_flag[mem][op]))
    return raw_bits, used_bits


def _assert_mapping_unrolling_complete(loops: LoopNest, source_tag: str):
    unrolling = [1 for _ in loops.ops.dim2Dict]
    for mapping in loops.tm + loops.sm:
        if mapping.dim < 0 or mapping.dim >= len(unrolling):
            raise ValueError(
                f"{source_tag}: illegal dimension id {mapping.dim} in mapping before replay"
            )
        unrolling[mapping.dim] *= int(round(mapping.dimSize))

    deficits = []
    for dim, dim_name in enumerate(loops.ops.dim2Dict):
        bound = int(loops.ops.dim2bound[dim])
        got = int(unrolling[dim])
        if got < bound:
            deficits.append(f"{dim_name}: got={got}, need={bound}")

    if deficits:
        raise ValueError(
            f"{source_tag}: incomplete unrolling before replay -> " + "; ".join(deficits)
        )


def project_replay_safe_double_flag(loops: LoopNest, double_tag):
    safe_double_tag = np.array(double_tag, dtype=int, copy=True)
    loops.usr_defined_double_flag = safe_double_tag

    while True:
        raw_bits, used_bits = _calc_explicit_mem_usage(loops)
        changed = False

        for mem in range(1, loops.acc.Num_mem):
            total_bits = int(np.sum(used_bits[mem]))
            if total_bits <= loops.acc.memSize[mem]:
                continue

            removable = []
            for op in range(3):
                if safe_double_tag[mem][op] == 1 and raw_bits[mem][op] > 0:
                    removable.append((int(raw_bits[mem][op]), op))
            removable.sort(reverse=True)

            for released_bits, op in removable:
                if total_bits <= loops.acc.memSize[mem]:
                    break
                safe_double_tag[mem][op] = 0
                total_bits -= released_bits
                changed = True

        loops.usr_defined_double_flag = safe_double_tag
        if changed is False:
            break

    return safe_double_tag

def pin_macro_weight_double_flag(loops: LoopNest):
    acc = loops.acc
    flags = [list(row) for row in np.array(loops.usr_defined_double_flag, dtype=int)]
    flags[acc.Macro2mem][1] = int(acc.double_config[acc.Macro2mem][1])
    loops.usr_defined_double_flag = project_replay_safe_double_flag(loops, flags)
    return loops.usr_defined_double_flag


from collections import deque
def convert_ZZMP_to_loopMP(
    mapping_dict: Dict[str, List[List[Tuple[str, int]]]],
    mappingArray: List[List[int]],
    mappingList: List[Mapping],
    *,
    array_row_order: Tuple[str, ...] = ("I", "W", "O"),
    dim_code = None
) -> List[Mapping]:
    if dim_code is None:
        dim_code = {
            "FX": 2,
            "FY": 1,
            "OX": 4,
            "OY": 3,
            "R": 1,
            "S": 2,
            "P": 3,
            "Q": 4,
            "C": 5,
            "K": 6,
            "G": 7,
            "N": 0,
            "B": 0,
        }

    if len(mappingArray) != len(array_row_order):
        raise ValueError("mappingArray 行数必须等于 array_row_order 长度")

    level_map: Dict[str, Dict[Tuple[str, int], deque[int]]] = {
        op: defaultdict(deque) for op in array_row_order
    }
    aligned_mapping_dict: Dict[str, List[List[Tuple[str, int]]]] = {}
    fallback_mem_level: Dict[str, int] = {}

    for row_idx, op in enumerate(array_row_order):
        layers = mapping_dict[op]
        pos = [col for col, v in enumerate(mappingArray[row_idx]) if v == 1]

        layers = _pad_missing_inner_levels(layers, len(pos), list)
        aligned_mapping_dict[op] = layers

        if len(pos) != len(layers):
            raise ValueError(
                f"{op}: 1 的数量 {len(pos)} ≠ 层数 {len(layers)}"
                "请检查 mappingArray / mapping_dict"
            )

        fallback_mem_level[op] = pos[-1]
        rev_pos = list(reversed(pos))
        for layer_idx, layer in enumerate(layers):
            mem_level = rev_pos[layer_idx]
            for tup in layer:
                level_map[op][tuple(tup)].appendleft(mem_level)

    required_count: Dict[Tuple[str, int], int] = {}
    for op in array_row_order:
        for tup, dq in level_map[op].items():
            required_count[tup] = max(required_count.get(tup, 0), len(dq))

    max_levels = max(len(aligned_mapping_dict[op]) for op in array_row_order)
    produced_count: Dict[Tuple[str, int], int] = defaultdict(int)
    order: List[Tuple[str, int]] = []
    for layer_idx in reversed(range(max_levels)):
        for op in array_row_order:
            layers = aligned_mapping_dict[op]
            if layer_idx >= len(layers):
                continue
            for tup in reversed(layers[layer_idx]):
                key = tuple(tup)
                if produced_count[key] < required_count.get(key, 0):
                    order.append(key)
                    produced_count[key] += 1

    for tup, need in required_count.items():
        while produced_count[tup] < need:
            order.append(tup)
            produced_count[tup] += 1

    for tup in order:
        levels = []
        for op in array_row_order:
            dq = level_map[op].get(tup)
            levels.append(dq.popleft() if dq else fallback_mem_level[op])

        mappingList.append(
            Mapping(
                dim=dim_code[tup[0]],
                dimSize=round(tup[1]),
                mem=levels
            )
        )
    return mappingList


def linearize_temporal_mapping(
    mapping_dict: Dict[str, List[List[Tuple[str, int]]]],
    mappingArray: List[List[int]],
    *,
    source_tag: str,
    array_row_order: Tuple[str, ...] = ("I", "W", "O"),
) -> List[Mapping]:
    dim_code = {"FX": 2, "FY": 1, "OX": 4, "OY": 3, "R": 1, "S": 2, "P": 3, "Q": 4,
                "C": 5, "K": 6, "G": 7, "N": 0, "B": 0}
    flat: Dict[str, List[Tuple[Tuple[str, int], int]]] = {}
    for row_idx, op in enumerate(array_row_order):
        pos = [col for col, v in enumerate(mappingArray[row_idx]) if v == 1]
        layers = _pad_missing_inner_levels(mapping_dict[op], len(pos), list)
        rev_pos = list(reversed(pos))
        flat[op] = [((dim, round(size)), rev_pos[layer_idx])
                    for layer_idx, layer in enumerate(layers) for dim, size in layer]
    seqs = {op: [tup for tup, _ in flat[op]] for op in array_row_order}
    ref = seqs[array_row_order[0]]
    for op in array_row_order[1:]:
        if seqs[op] != ref:
            raise ValueError(f"{source_tag}: operands disagree on the temporal loop order: "
                             f"{array_row_order[0]}={ref} {op}={seqs[op]}")
    tm = []
    for k in range(len(ref) - 1, -1, -1):
        dim, size = ref[k]
        tm.append(Mapping(dim=dim_code[dim], dimSize=size, mem=[flat[op][k][1] for op in array_row_order]))
    return tm


def _find_preprogress_orphans(tm: List[Mapping]) -> List[Tuple[int, int, int]]:
    orphans: List[Tuple[int, int, int]] = []
    if not tm:
        return orphans
    n = len(tm)
    for op in range(3):
        tail = tm[-1].mem[op]
        for i in range(n - 1):
            M = tm[i].mem[op]
            if M == tail:
                continue
            if not any(tm[j].mem[op] > M for j in range(i + 1, n)):
                orphans.append((i, op, M))
    return orphans


def _enforce_tm_preprogress_invariant(tm: List[Mapping], *, source_tag: str) -> None:
    tm.sort(key=lambda m: max(m.mem))
    orphans = _find_preprogress_orphans(tm)
    if orphans:
        head = ", ".join(f"(i={i}, op={op}, level={M})" for i, op, M in orphans[:4])
        Logger.warning(
            f"{source_tag}: {len(orphans)} tm orphan level(s) remain after "
            f"monotonicity pass [{head}]; preprogress safety-net will engage."
        )


def fix_all_memHierarchy(acc:CIM_Acc, tm:list[Mapping]):
    tm_first = tm[0]
    tm_last = tm[-1]
    if (tm_first.mem[0]!=1) or (tm_first.mem[1]!=1) or (tm_first.mem[2]!=1):
        tm.insert(0, Mapping(tm_first.dim, 1, [1, 1, 1]))
    return tm

def normalize_spatial_mapping(mapping):
    mp = copy.deepcopy(mapping)

    canonical: Dict[str, List[float]] = {}
    for layers in mp.values():
        counts: Dict[str, List[float]] = {}
        for fl in layers:
            for v, val in fl:
                counts.setdefault(v, []).append(float(val))
        for v, seq in counts.items():
            if len(seq) > len(canonical.get(v, [])):
                canonical[v] = seq[:]

    for layers in mp.values():
        occ: Dict[str, List[Tuple[int, int, float]]] = {}
        for depth, fl in enumerate(layers):
            for idx, (v, val) in enumerate(fl):
                occ.setdefault(v, []).append((depth, idx, float(val)))

        for var, finest in canonical.items():
            if len(finest) <= 1:
                continue
            fin_prod = math.prod(finest)

            positions = occ.get(var, [])
            if len(positions) == 1:
                depth, idx, cur_val = positions[0]
                if math.isclose(cur_val, fin_prod):
                    layer = layers[depth]

                    layer.pop(idx)

                    for offset, f in enumerate(finest):
                        layer.insert(idx + offset, (var, f))

    if mp and all(l and l[-1] == [] for l in mp.values()):
        for k in mp:
            mp[k] = mp[k][:-1]

    return mp


def baseMapping_from_zigzag_cme(cme) -> ClassMapping:
    cme_dim = cme.layer.loop_dim_size
    return ClassMapping(
        source="zigzag",
        loop_dim={
            "R": cme_dim["FY"],
            "S": cme_dim["FX"],
            "P": cme_dim["OY"],
            "Q": cme_dim["OX"],
            "C": cme_dim["C"],
            "K": cme_dim["K"],
            "G": cme_dim["G"],
        },
        temporal_mapping=cme.temporal_mapping.mapping_dic_origin,
        spatial_mapping=cme.spatial_mapping_int.mapping_dict_origin,
        double_buffer_flag=cme.double_buffer_true,
        top_r_loop_size=_extract_top_r_loop_size(cme),
        raw=cme,
    )


def convert_baseMapping_to_MIREDO(loops: LoopNest, baseMapping: ClassMapping):
    temporal_mapping_dic = baseMapping.temporal_mapping
    spatial_mapping_dict = baseMapping.spatial_mapping
    double_buffer_flag = baseMapping.double_buffer_flag

    if baseMapping.top_r_loop_size is None:
        temporal_processed = temporal_mapping_dic
    else:
        temporal_processed = process_top_r(
            ori_tm_dict=temporal_mapping_dic,
            cme_top_r_loop=baseMapping.top_r_loop_size,
        )

    temporal_aligned = align_mapping_levels_to_acc(
        mapping_dict=temporal_processed,
        mapping_array=loops.acc.mappingArray,
        source_tag=f"{baseMapping.source}:temporal",
        warn=Logger.warning,
    )
    spatial_aligned = align_mapping_levels_to_acc(
        mapping_dict=normalize_spatial_mapping(spatial_mapping_dict),
        mapping_array=loops.acc.mappingArray,
        source_tag=f"{baseMapping.source}:spatial",
        warn=Logger.warning,
    )
    dflag_aligned = align_double_buffer_flags_to_acc(
        double_buffer_flag=double_buffer_flag,
        mapping_array=loops.acc.mappingArray,
        source_tag=f"{baseMapping.source}:double_buffer",
        warn=Logger.warning,
    )

    loops.tm = loops.tm + linearize_temporal_mapping(
        mapping_dict=temporal_aligned,
        mappingArray=loops.acc.mappingArray,
        source_tag=f"{baseMapping.source}:temporal",
    )
    _enforce_tm_preprogress_invariant(loops.tm, source_tag=baseMapping.source)

    loops.sm = convert_ZZMP_to_loopMP(
        mapping_dict=spatial_aligned,
        mappingArray=loops.acc.mappingArray,
        mappingList=loops.sm,
    )

    if not loops.tm:
        loops.tm.append(
            Mapping(
                dim=0,
                dimSize=1,
                mem=[loops.acc.Dram2mem for _ in range(3)],
            )
        )

    _assert_mapping_unrolling_complete(loops, source_tag=f"{baseMapping.source}:unrolling_check")

    try:
        loops.usr_defined_double_flag = convert_ZZ_dflag_to_doubleflag(
            loops.acc, loops.ops, dflag_aligned
        )
    except Exception as e:
        Logger.warning(
            f"Falling back to all-disabled double buffer flags for source={baseMapping.source}: {e}"
        )
        loops.usr_defined_double_flag = [
            [0 for _ in range(3)] for __ in range(loops.acc.Num_mem + 1)
        ]
    pin_macro_weight_double_flag(loops)
    return loops


def convert_Zigzag_to_MIREDO(loops: LoopNest, cme=None):
    baseMapping = baseMapping_from_zigzag_cme(cme)
    return convert_baseMapping_to_MIREDO(loops=loops, baseMapping=baseMapping)


def compare_ops_cme(loopDim, cme):
    cme_dim = cme.layer.loop_dim_size
    if len(cme_dim) < 6:
        return False
    return (
        loopDim['R'] == cme_dim['FY'] and
        loopDim['S'] == cme_dim['FX'] and
        loopDim['P'] == cme_dim['OY'] and
        loopDim['Q'] == cme_dim['OX'] and
        loopDim['C'] == cme_dim['C']  and
        loopDim['K'] == cme_dim['K']  and
        loopDim['G'] == cme_dim['G']
    )

def get_dim_from_cme(cme):
    dims = cme.layer.loop_dim_size
    loop_dim = f"{dims['FX']}_{dims['FY']}_{dims['OX']}_{dims['OY']}_{dims['C']}_{dims['K']}"
    if len(loop_dim) < 6:
        return False
