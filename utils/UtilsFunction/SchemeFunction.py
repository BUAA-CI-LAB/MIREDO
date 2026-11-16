import math
from utils.GlobalUT import *
from Architecture.ArchSpec import CIM_Acc
from utils.Workload import WorkLoad


def score_scheme(acc:CIM_Acc, ops:WorkLoad, scheme):
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
    temporal_iters = math.prod(temporal_unrolling)

    compute_lat = temporal_iters * acc.t_MAC
    dram_bw_lat = 0
    for op in range(3):
        coeff_rw = 2 if op == 2 else 1
        prec = acc.precision_final if op == 2 else acc.precision[acc.Dram2mem, op]
        dram_bw_lat = max(dram_bw_lat, coeff_rw * ops.size[op] * prec / acc.bw[acc.Dram2mem])
    latency_lb = max(compute_lat, dram_bw_lat)

    count_core = math.prod(scheme[0])
    mac_energy = acc.cost_ActMacro * temporal_iters * count_core
    dram_energy = 0
    dram_m = acc.Dram2mem
    min_cost_rw = min(acc.cost_r[dram_m], acc.cost_w[dram_m])
    for op in range(3):
        coeff_rw = 2 if op == 2 else 1
        dram_energy += coeff_rw * ops.size[op] * min_cost_rw
    energy_lb = mac_energy + dram_energy

    edp_lb = latency_lb * energy_lb

    return latency_lb, energy_lb, edp_lb


def dominance_filter(scheme_records):
    pareto_front = []
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
