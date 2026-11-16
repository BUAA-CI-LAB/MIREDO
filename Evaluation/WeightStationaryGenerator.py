import math
import logging
from dataclasses import dataclass

from Architecture.ArchSpec import CIM_Acc
from Evaluation.Zigzag_imc.CompatibleZigzag import project_replay_safe_double_flag
from Simulator.Simulax import tranSimulator
from utils.GlobalUT import CONST, Logger
from utils.UtilsFunction.ToolFunction import getDivisors
from utils.Workload import LoopNest, Mapping, WorkLoad
from utils.factorization import flexible_factorization


DEFAULT_WS_TEMPORAL_ORDER = ("G", "K", "C", "R", "S", "P", "Q")
DEFAULT_WS_AXIS0_PRIORITY = ("K", "G", "P", "Q")
DEFAULT_WS_AXIS1_PRIORITY = ("C", "R", "S")
DEFAULT_WS_AXIS2_PRIORITY = ("K",)
DEFAULT_OP_PRIORITY = (1, 0, 2)


@dataclass
class WSBaselineResult:
    dataflow: LoopNest
    latency: float
    energy: float
    profile: object
    scheme: list
    policy: str = "canonical_cim_ws"
    mapper_wall_sec: float = 0.0

    @property
    def edp(self):
        return self.latency * self.energy


def _largest_divisor_at_most(value, limit):
    return max(divisor for divisor in getDivisors(value) if divisor <= limit)


def _assign_axis_dims(ops: WorkLoad, remaining_bounds, axis_capacity, dim_order):
    axis_scheme = [1] * ops.Num_dim
    remaining_capacity = axis_capacity

    for dim_char in dim_order:
        if remaining_capacity == 1:
            break
        dim = ops.dict2Dim(dim_char)
        factor = _largest_divisor_at_most(remaining_bounds[dim], remaining_capacity)
        axis_scheme[dim] = factor
        remaining_bounds[dim] //= factor
        remaining_capacity //= factor

    return axis_scheme


def derive_weight_stationary_spatial_scheme(acc: CIM_Acc, ops: WorkLoad):
    remaining_bounds = ops.dim2bound[:]
    scheme = [[1] * ops.Num_dim for _ in range(acc.Num_SpUr)]

    scheme[2] = _assign_axis_dims(ops, remaining_bounds, acc.SpUnrolling[2], DEFAULT_WS_AXIS2_PRIORITY)
    scheme[1] = _assign_axis_dims(ops, remaining_bounds, acc.SpUnrolling[1], DEFAULT_WS_AXIS1_PRIORITY)
    scheme[0] = _assign_axis_dims(ops, remaining_bounds, acc.SpUnrolling[0], DEFAULT_WS_AXIS0_PRIORITY)

    for axis in range(acc.Num_SpUr):
        axis_prod = math.prod(scheme[axis])
        if axis_prod > acc.SpUnrolling[axis]:
            raise ValueError(f"Illegal WS spatial policy on axis {axis}: {axis_prod} > {acc.SpUnrolling[axis]}")

    return scheme


def _spatial_dims_per_memory(acc: CIM_Acc, ops: WorkLoad, scheme):
    spatial_dims = {}
    for mem in range(1, acc.Num_mem):
        spatial_dims[mem] = {}
        for op in range(3):
            dims = [1] * ops.Num_dim
            for axis in range(acc.Num_SpUr):
                if mem <= acc.SpUr2Mem[axis, op]:
                    for dim in range(1, ops.Num_dim):
                        dims[dim] *= scheme[axis][dim]
            spatial_dims[mem][op] = dims
    return spatial_dims


def _temporal_factors(ops: WorkLoad, temporal_unrolling, loop_order, M):
    factors = []
    for dim_char in loop_order:
        dim = ops.dict2Dim(dim_char)
        for factor in sorted(flexible_factorization(temporal_unrolling[dim], M), reverse=True):
            if factor > 1:
                factors.append((dim, factor))
    return factors


def _output_uses_psum_precision(ops: WorkLoad, factors, assignments, mem, candidate_idx=None, candidate_mem=None):
    for idx, (dim, _) in enumerate(factors):
        mapped_mem = assignments[2][idx]
        if idx == candidate_idx:
            mapped_mem = candidate_mem
        if ops.relevance[2][dim] == 0 and mapped_mem is None:
            return True
        if ops.relevance[2][dim] == 0 and mapped_mem <= mem:
            return True
    return False


def _resident_bits(acc: CIM_Acc, ops: WorkLoad, spatial_dims, factors, assignments, mem, op,
                   candidate_idx=None, candidate_mem=None):
    if acc.mappingArray[op][mem] == 0:
        return 0

    dims = spatial_dims[mem][op][:]
    for idx, mapped_mem in enumerate(assignments[op]):
        current_mem = mapped_mem
        if idx == candidate_idx:
            current_mem = candidate_mem
        if current_mem is not None and mem <= current_mem:
            dims[factors[idx][0]] *= factors[idx][1]

    if op == 2:
        precision = acc.precision_psum if _output_uses_psum_precision(
            ops=ops,
            factors=factors,
            assignments=assignments,
            mem=mem,
            candidate_idx=candidate_idx,
            candidate_mem=candidate_mem,
        ) else acc.precision_final
    else:
        precision = acc.precision[mem, op]

    return ops.get_operand_size(dims, op) * precision


def _fits_shared_capacity(acc: CIM_Acc, ops: WorkLoad, spatial_dims, factors, assignments,
                          op, factor_idx, candidate_mem, reserve_double_buffer=False):
    for mem in range(1, candidate_mem + 1):
        if acc.mappingArray[op][mem] == 0:
            continue

        total_bits = 0
        for operand in range(3):
            bits = _resident_bits(
                acc=acc,
                ops=ops,
                spatial_dims=spatial_dims,
                factors=factors,
                assignments=assignments,
                mem=mem,
                op=operand,
                candidate_idx=(factor_idx if operand == op else None),
                candidate_mem=(candidate_mem if operand == op else None),
            )
            if reserve_double_buffer and bits > 0 and acc.double_config[mem][operand]:
                bits *= 2
            total_bits += bits
        if total_bits > acc.memSize[mem]:
            return False
    return True


def _candidate_double_flags(acc: CIM_Acc, loops: LoopNest):
    double_flag = [[0] * 3 for _ in range(acc.Num_mem + 1)]
    used_memories = {(mapping.mem[op], op) for mapping in loops.tm for op in range(3)}

    for mem in range(1, acc.Num_mem):
        for op in range(3):
            if acc.double_config[mem][op] and (mem, op) in used_memories:
                double_flag[mem][op] = 1
    return double_flag


def build_weight_stationary_dataflow(acc: CIM_Acc, ops: WorkLoad, scheme=None,
                                     loop_order=DEFAULT_WS_TEMPORAL_ORDER,
                                     op_priority=DEFAULT_OP_PRIORITY,
                                     enable_double_buffer=False):
    if scheme is None:
        scheme = derive_weight_stationary_spatial_scheme(acc=acc, ops=ops)

    spatial_unrolling = [math.prod(col) for col in zip(*scheme)]
    temporal_unrolling = [math.ceil(bound / unroll) for bound, unroll in zip(ops.dim2bound, spatial_unrolling)]
    factors = _temporal_factors(ops, temporal_unrolling, loop_order, acc.placement_depth)
    spatial_dims = _spatial_dims_per_memory(acc, ops, scheme)

    assignments = {op: [None] * len(factors) for op in range(3)}
    for op in op_priority:
        allowed_levels = [mem for mem in range(1, acc.Num_mem) if acc.mappingArray[op][mem] == 1]
        inner_to_outer = list(reversed(allowed_levels))
        level_idx = 0

        for factor_idx in range(len(factors) - 1, -1, -1):
            while True:
                candidate_mem = inner_to_outer[level_idx]
                if _fits_shared_capacity(
                    acc=acc,
                    ops=ops,
                    spatial_dims=spatial_dims,
                    factors=factors,
                    assignments=assignments,
                    op=op,
                    factor_idx=factor_idx,
                    candidate_mem=candidate_mem,
                    reserve_double_buffer=enable_double_buffer,
                ):
                    assignments[op][factor_idx] = candidate_mem
                    break

                level_idx += 1
                if level_idx >= len(inner_to_outer):
                    dim_char, factor = factors[factor_idx]
                    raise ValueError(
                        f"WS baseline cannot place factor {ops.dim2Dict[dim_char]}={factor} for operand {op}"
                    )

    loops = LoopNest(acc=acc, ops=ops)
    for axis in range(acc.Num_SpUr):
        for dim in range(1, ops.Num_dim):
            if scheme[axis][dim] > 1:
                loops.sm.append(
                    Mapping(
                        dim=dim,
                        dimSize=scheme[axis][dim],
                        mem=[acc.SpUr2Mem[axis, op] for op in range(3)],
                    )
                )

    for factor_idx, (dim, factor) in enumerate(factors):
        loops.tm.append(
            Mapping(
                dim=dim,
                dimSize=factor,
                mem=[assignments[op][factor_idx] for op in range(3)],
            )
        )
    if not loops.tm:
        loops.tm.append(
            Mapping(
                dim=0,
                dimSize=1,
                mem=[acc.Dram2mem for _ in range(3)],
            )
        )

    if enable_double_buffer:
        loops.usr_defined_double_flag = _candidate_double_flags(acc, loops)
        loops.usr_defined_double_flag = project_replay_safe_double_flag(loops, loops.usr_defined_double_flag)
    else:
        loops.usr_defined_double_flag = [[0] * 3 for _ in range(acc.Num_mem + 1)]

    loops.preprogress()
    return loops


def _simulate_ws(acc, ops, scheme, enable_double_buffer):
    import copy as _copy
    loops = build_weight_stationary_dataflow(
        acc=_copy.deepcopy(acc), ops=ops, scheme=scheme,
        enable_double_buffer=enable_double_buffer,
    )
    simulator = tranSimulator(acc=_copy.deepcopy(acc), ops=ops, dataflow=loops)
    latency, energy = simulator.run()
    return loops, latency, energy, simulator.PD


def generate_weight_stationary_baseline(acc: CIM_Acc, ops: WorkLoad, quiet=True, enable_double_buffer=True):
    import copy as _copy
    import time as _time
    previous_level = Logger.level
    previous_disable = logging.root.manager.disable
    if quiet:
        Logger.setLevel(logging.ERROR)
        logging.disable(logging.CRITICAL)

    _mapper_t0 = _time.time()
    scheme = derive_weight_stationary_spatial_scheme(acc=acc, ops=ops)
    _ = build_weight_stationary_dataflow(
        acc=_copy.deepcopy(acc), ops=ops, scheme=scheme,
        enable_double_buffer=enable_double_buffer,
    )
    mapper_wall_sec = _time.time() - _mapper_t0

    try:
        if enable_double_buffer:
            loops_db, lat_db, eng_db, pd_db = _simulate_ws(acc, ops, scheme, True)
            loops_no, lat_no, eng_no, pd_no = _simulate_ws(acc, ops, scheme, False)
            if lat_db <= lat_no:
                loops, latency, energy, pd = loops_db, lat_db, eng_db, pd_db
                policy = "canonical_cim_ws_db_reserved"
            else:
                loops, latency, energy, pd = loops_no, lat_no, eng_no, pd_no
                policy = "canonical_cim_ws_no_db"
        else:
            loops, latency, energy, pd = _simulate_ws(acc, ops, scheme, False)
            policy = "canonical_cim_ws_no_db"
    finally:
        if quiet:
            Logger.setLevel(previous_level)
            logging.disable(previous_disable)

    return WSBaselineResult(
        dataflow=loops,
        latency=latency,
        energy=energy,
        profile=pd,
        scheme=scheme,
        policy=policy,
        mapper_wall_sec=mapper_wall_sec,
    )

from typing import Optional as _Optional, Dict as _Dict
import copy as _copy


def supports_loopdim(loopdim: _Dict[str, int]) -> _Optional[str]:
    return None


def run_for_layer(acc, ops, loopdim, model_name, architecture, objective):
    from Evaluation.common.BaselineProvider import BaselineRunResult
    result = generate_weight_stationary_baseline(acc=_copy.deepcopy(acc), ops=ops)
    return BaselineRunResult(
        method="ws",
        objective=objective,
        latency=result.latency,
        energy=result.energy,
        profile=result.profile,
        dataflow=result.dataflow,
        metadata={
            "policy": result.policy,
            "mapper_wall_sec": result.mapper_wall_sec,
        },
    )
