# 空间方案评估与筛选工具
# score_scheme: 空间利用率评分
# scheme_objective_lb: 解析下界计算 (Propositions 1-3)
# dominance_filter: Pareto前沿支配筛选

import math
from utils.GlobalUT import *
from Architecture.ArchSpec import CIM_Acc
from utils.Workload import WorkLoad


def score_scheme(acc:CIM_Acc, ops:WorkLoad, scheme):
    """计算空间方案的利用率评分，用于排序（高分优先求解）。"""
    axis_products = [math.prod(axis_scheme) for axis_scheme in scheme]
    axis_utils = [x / y if y > 0 else 1.0 for x, y in zip(axis_products, acc.SpUnrolling)]

    spatial_unrolling = [math.prod(col) for col in zip(*scheme)]
    temporal_unrolling = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial_unrolling)]

    util_product = math.prod(axis_utils) if axis_utils else 1.0
    util_min = min(axis_utils) if axis_utils else 1.0
    util_avg = sum(axis_utils) / len(axis_utils) if axis_utils else 1.0
    temporal_logsum = sum(math.log(max(1, tu)) for tu in temporal_unrolling)

    return {
        "spatial_unrolling": spatial_unrolling,
        "temporal_unrolling": temporal_unrolling,
        "util_product": util_product,
        "util_min": util_min,
        "util_avg": util_avg,
        "sort_key": (util_product, util_min, util_avg, -temporal_logsum),
    }


def scheme_objective_lb(acc:CIM_Acc, ops:WorkLoad, scheme, temporal_unrolling):
    """可证明的解析下界，对任意可行temporal mapping成立。

    每个下界对应MIP中的约束，保证 LB ≤ MIP_optimal ≤ solver_result。

    Latency LB (Proposition 1):
      LB_lat = max(LB_compute, LB_dram)
      LB_compute = ∏temporal × t_MAC          ← MIP: latency_Process[0,op] ≥ MIN_INNER_PROD[0] × t_MAC
      LB_dram = max_op{coeff_rw × |D_op| × prec / BW}  ← MIP: Cut_DRAM_BW
        I/W: prec = precision[Dram2mem, op]
        O:   prec = precision_final (holdPsum-dependent cut的最小值)

    Energy LB (Proposition 2):
      LB_eng = E_compute + E_dram
      E_compute = cost_ActMacro × ∏temporal × cores
      E_dram = Σ_op coeff_rw × |D_op| × min(cost_r, cost_w)

    EDP LB (Proposition 3):
      LB_edp = LB_lat × LB_eng        (raw physical EDP; no solve-time conditioner)
    """
    temporal_iters = math.prod(temporal_unrolling)

    # ── Latency LB ──
    compute_lat = temporal_iters * acc.t_MAC
    dram_bw_lat = 0
    for op in range(3):
        coeff_rw = 2 if op == 2 else 1
        prec = acc.precision_final if op == 2 else acc.precision[acc.Dram2mem, op]
        dram_bw_lat = max(dram_bw_lat, coeff_rw * ops.size[op] * prec / acc.bw[acc.Dram2mem])
    latency_lb = max(compute_lat, dram_bw_lat)

    # ── Energy LB ──
    count_core = math.prod(scheme[0])
    mac_energy = acc.cost_ActMacro * temporal_iters * count_core
    dram_energy = 0
    dram_m = acc.Dram2mem
    min_cost_rw = min(acc.cost_r[dram_m], acc.cost_w[dram_m])
    for op in range(3):
        coeff_rw = 2 if op == 2 else 1
        dram_energy += coeff_rw * ops.size[op] * min_cost_rw
    energy_lb = mac_energy + dram_energy

    # ── EDP LB ──
    # Raw physical EDP bound (latency*energy). Matches the raw EDP that the
    # solver reports and that pruning compares against; the solve-time
    # conditioner is applied only inside the MIP, never on this bound.
    edp_lb = latency_lb * energy_lb

    return latency_lb, energy_lb, edp_lb


def dominance_filter(scheme_records):
    """Pareto前沿支配筛选：移除被其他方案在所有temporal维度上支配的方案。

    Scheme A dominates B iff temporal_A[d] ≤ temporal_B[d] for ALL d.
    复杂度: O(|front| × N × D)，front通常较小。

    注意：此机制为实验性验证，非可证明安全（因子分解差异）。
    """
    pareto_front = []   # (temporal_unrolling, index into survived)
    survived = []

    for rec in scheme_records:
        tu = rec["meta"]["temporal_unrolling"]
        dominated = False
        for front_tu, _ in pareto_front:
            if all(ft <= bt for ft, bt in zip(front_tu, tu)):
                dominated = True
                break
        if not dominated:
            new_front = []
            for ft, idx in pareto_front:
                if all(t <= f for t, f in zip(tu, ft)):
                    survived[idx] = None
                else:
                    new_front.append((ft, idx))
            new_front.append((tu, len(survived)))
            pareto_front = new_front
            survived.append(rec)

    return [r for r in survived if r is not None]
