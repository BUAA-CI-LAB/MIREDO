# this file is prepared for project 112
# Created by iboxl

import os
import psutil
from Architecture.ArchSpec import CIM_Acc
from utils.Workload import WorkLoad, LoopNest, Mapping
import math
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from dataclasses import dataclass
from utils.GlobalUT import *
from utils.UtilsFunction.SolverFunction import *
from utils.factorization import flexible_factorization
from utils.UtilsFunction.ToolFunction import getDivisors, getUniqueFactors
from utils.UtilsFunction.SchemeFunction import scheme_objective_lb
import copy

@dataclass
class SolverProfile:
    objective: str = ""
    metric_upper_bound: float = 0.0
    threads: int | None = None
    soft_mem_limit_gb: float | None = None
    model_build_time_sec: float = 0.0
    feasibility_time_sec: float = 0.0
    optimize_time_sec: float = 0.0
    total_time_sec: float = 0.0
    num_vars: int = 0
    num_bin_vars: int = 0
    num_constrs: int = 0
    num_sos: int = 0
    num_genconstrs: int = 0
    status: int = 0
    sol_count: int = 0
    mip_gap: float | None = None
    best_bound: float | None = None
    node_count: float = 0.0
    callback_mipsol_updates: int = 0
    callback_dynamic_terminations: int = 0
    metric_upper_bound_applied: bool = False

class Solver():
    def __init__(self, acc:CIM_Acc, ops:WorkLoad, tu, su, metric_ub, outputdir=None,
                 threads=None, soft_mem_limit_gb=None, fixed_factor_ordering=None,
                 shared_ub=None, env=None, fixed_value_sequence=None,
                 fixed_partial_ordering=None):
        self.acc = copy.deepcopy(acc)
        self.ops = copy.deepcopy(ops)
        self.tu = tu
        self.su = su
        # EDP solve-time numerical conditioner, computed per spatial candidate
        # (per Solver). Raw EDP (latency*energy) spans ~8 orders across layers,
        # so a fixed factor leaves small layers' objective near the solver's 1e-6
        # tolerance and falsely "optimal". Normalize from the candidate's raw
        # roofline lower bound so res_EDP lands far above tolerance at any size.
        # The reported EDP metric stays raw (this factor never leaves the solve);
        # Latency/Energy modes need no conditioner.
        self.edp_scaling_factor = 1.0  # inert default; only consumed (and overwritten) in EDP mode
        if CONST.FLAG_OPT == "EDP":
            try:
                _lat_lb, _eng_lb, _ = scheme_objective_lb(acc, ops, su, tu)
                _raw_edp_est = float(_lat_lb) * float(_eng_lb)
            except Exception as _e:
                _raw_edp_est = None
                Logger.warning(f"EDP scaling: roofline LB computation failed ({_e})")
            if _raw_edp_est is not None and math.isfinite(_raw_edp_est) and _raw_edp_est > 0:
                self.edp_scaling_factor = CONST.SCALE_LATENCY / (10 ** math.floor(math.log10(_raw_edp_est)))
            else:
                # Neutral fallback: objective = raw_EDP / SCALE_LATENCY, always far
                # above the solver tolerance — never the old 1e-6 mis-conditioning.
                self.edp_scaling_factor = 1.0
                Logger.warning("EDP scaling: unusable roofline LB; neutral fallback edp_scaling_factor=1.0")
        self.metric_ub = metric_ub
        self.shared_ub = shared_ub
        self.outputdir = outputdir
        self.threads = threads
        self.soft_mem_limit_gb = soft_mem_limit_gb
        self.fixed_factor_ordering = fixed_factor_ordering  # dict{(d,f)->i} for enumeration mode
        # Value-sequence mode (coarser than full ordering): list of length Num_Loops
        # giving the factor VALUE pinned at each loop position. Pins indic_loop2Factor
        # only (makes Z_crit exact, killing the dominant McCormick gap) while leaving
        # the dimension->position assignment indic_factor2Loop FREE for the MIP. This
        # collapses the same-value cross-dimension permutations that blow the full
        # ordering count to ~1e12 down by 1e2-1e5x, trading explicit enumeration for
        # MIP search at a granularity where each sub-MIP still closes.
        self.fixed_value_sequence = fixed_value_sequence
        # Partial-ordering mode (adaptive branch-and-bound node): dict {(d,f)->i}
        # pinning a SUBSET of factors to positions. Pinned positions get exact
        # loop2Factor (Z_crit exact there); unpinned factors/positions stay free.
        # Generalizes both full ordering (all pinned) and free root (none pinned),
        # letting an outer B&B branch only where a node's MIP fails to close.
        self.fixed_partial_ordering = fixed_partial_ordering
        self.gurobi_output = FLAG.GUROBI_OUTPUT
        self._owns_env = True

        if env is not None:
            self.env = env
            self._owns_env = False
        else:
            self.env = gp.Env(empty=True)
            self.env.setParam('OutputFlag', self.gurobi_output)
            if self.threads is not None:
                self.env.setParam('ThreadLimit', max(1, int(self.threads)))
            self.env.start()

        self.model = gp.Model(name="MIREDO", env=self.env)
        self.model.setParam('OutputFlag', self.gurobi_output)
        self.model.setParam('Seed', 112)                               # 随机种子
        if self.gurobi_output:
            self.model.setParam('LogFile', os.path.join(self.outputdir, "Solver.log"))
        
        if self.threads is None:
            self.model.setParam('Threads', psutil.cpu_count(logical=False))
        else:
            self.model.setParam('Threads', max(1, int(self.threads)))
        if self.soft_mem_limit_gb is not None:
            self.model.setParam('SoftMemLimit', float(self.soft_mem_limit_gb))

        # self.model.setParam('NonConvex', 2)
        # self.model.setParam('PreQLinearize', 2)

        self.model.setParam('Cuts', -1)
        self.model.setParam('Presolve', 2)
        self.model.setParam('Heuristics', 0.25)
        self.model.setParam("ScaleFlag", 2)                          # 用于调节数值比例问题coefficient range
        self.model.setParam("NumericFocus", 1)
        self.model.setParam("Symmetry", 2)                           # 自动对称检测+破坏
        # self.model.setParam('MIPGap', 0.03)

        self.model.setParam('FeasibilityTol', 1e-6)
        self.model.setParam('IntFeasTol', 1e-6)
        self.model.setParam('OptimalityTol', 1e-6)

        # Spend first 10s of the 60s budget in the no-relaxation heuristic to
        # land a feasible incumbent fast; the remaining 50s goes to B&C tightening.
        self.model.setParam('NoRelHeurTime', 10)
        
        self.result = {}
        self.dataflow = {}
        self.profile = SolverProfile(
            objective=CONST.FLAG_OPT,
            metric_upper_bound=self.metric_ub,
            threads=self.threads,
            soft_mem_limit_gb=self.soft_mem_limit_gb,
        )

        # FlexFact coarsening depth comes from the hardware (acc.placement_depth,
        # the deepest per-operand memory-placement chain), never a constant.
        self.FACTORS = [flexible_factorization(_, acc.placement_depth) for _ in self.tu]
        self.spatial_unrolling = [math.prod(col) for col in zip(*su)]
        self.MAX_TRANS = [ max([math.ceil(min(ops.size[op] * (acc.precision_psum if op == 2 else acc.precision[m,op]), acc.memSize[m]) / acc.bw[m]) / CONST.SCALE_LATENCY
                                for m in range(1, acc.Num_mem) if acc.mappingArray[op][m]] 
                            + [0]) for op in range(3) ]                             # maxTrans[op] = max transfer time for op in {I,W,O}
        
        Logger.critical(f"Best {CONST.FLAG_OPT} metric upper bound is {self.metric_ub}")

    def _refresh_metric_ub(self):
        current_ub = CONST.MAX_POS if self.metric_ub is None else self.metric_ub
        if self.shared_ub is not None:
            current_ub = min(current_ub, self.shared_ub.value)
        self.metric_ub = current_ub
        self.profile.metric_upper_bound = current_ub
        return current_ub

    def _has_finite_metric_ub(self):
        return self.metric_ub is not None and self.metric_ub < CONST.MAX_POS * 0.5

    def _solve_trivial(self, acc: CIM_Acc, ops: WorkLoad, factors):
        """Analytical solution for the degenerate case: Num_Loops == 0.

        When spatial unrolling covers the entire computation, all temporal
        factors are 1 and the mapping has zero degrees of freedom. The result
        is computed directly from the MIP model's L5/L6 latency formulas and
        per-memory energy formulas, evaluated at the unique feasible point.
        """
        start_time = time.time()
        Logger.info("Num_Loops == 0: analytical solution (no MIP needed)")

        # ── Spatial volume at each memory level ───────────────────────────────
        spur = {}
        for m in range(1, acc.Num_mem):
            for d in range(1, ops.Num_dim):
                for op in range(3):
                    spur[m, op, d] = 1
                    for u in range(acc.Num_SpUr):
                        if m <= acc.SpUr2Mem[u, op]:
                            spur[m, op, d] *= self.su[u][d]

        # ── Latency (MIP model L5/L6, simplified for Num_Loops=0) ────────────
        # No temporal loops → all data bootstrapped from DRAM before compute.
        # bootstrap[op] = full operand size / DRAM bandwidth
        # Output uses precision_final (no partial sums when Num_Loops=0).
        def dram_prec(op):
            return acc.precision_final if op == 2 else acc.precision[acc.Dram2mem, op]

        MAX_SIZE = ops.size
        bootstrap = [MAX_SIZE[op] * dram_prec(op) / acc.bw[acc.Dram2mem] for op in range(3)]
        max_startup = max(bootstrap)
        t_MAC = acc.t_MAC
        # L6: res_latency >= MaxStartup + Process[0,op] + BootstrapWrite[O]
        # Process[0,op] = t_MAC (base case); BootstrapWrite only for Output.
        solved_latency = max_startup + t_MAC + bootstrap[2]

        # ── Energy (per-memory formulas, simplified for Num_Loops=0) ──────────
        # All temporal factor contributions vanish (log(1)=0).
        # Only unconditional memories contribute (DRAM, IReg, OReg);
        # all other memories are bypassed (indic_usedMem=0).
        # No partial sums → Output precision = precision_final everywhere.

        # MAC energy: count_mac = 1 (single temporal pass), count_core from spatial
        count_core = 1
        for d in range(1, ops.Num_dim):
            count_core *= self.su[0][d]
        energy_comp = acc.cost_ActMacro * 1 * count_core

        # Per-memory read/write energy for unconditional memories
        energy_rw = 0.0
        unconditional_mems = {acc.Dram2mem, acc.IReg2mem, acc.OReg2mem}
        for m in unconditional_mems:
            for op in range(3):
                if not acc.mappingArray[op][m]:
                    continue

                # Tile volume at this memory level (spatial-only)
                dim_at_m = [1] * ops.Num_dim
                for d in range(1, ops.Num_dim):
                    dim_at_m[d] = spur.get((m, op, d), 1)
                vol = ops.get_operand_size(dim_at_m, op)

                # Access count: product of spatial unrollings BELOW this level
                count = 1
                for d in range(1, ops.Num_dim):
                    for u in range(acc.Num_SpUr):
                        if m > acc.SpUr2Mem[u, op]:
                            count *= self.su[u][d]

                prec = acc.precision_final if op == 2 else acc.precision[m, op]

                can_read = (m not in (acc.IReg2mem, acc.OReg2mem, acc.Macro2mem)
                            and acc.cost_r[m] > 0)
                can_write = m > 1

                e = 0.0
                if can_read:
                    e += acc.cost_r[m] * prec * count * vol       # r2L: read toward compute
                if can_write:
                    e += acc.cost_w[m] * prec * count * vol       # w2L: write from DRAM side
                if op == 2 and can_read and m > acc.Dram2mem:
                    e += acc.cost_r[m] * prec * count * vol       # r2H: output read-back
                if op == 2:
                    e += acc.cost_w[m] * prec * count * vol       # w2H: output write-back
                energy_rw += e

        energy_leakage = acc.leakage_per_cycle * solved_latency
        solved_energy = energy_rw + energy_comp + energy_leakage

        # Reported EDP is raw physical latency*energy; the solve-time conditioner
        # never leaks into the reported metric.
        solved_edp = solved_latency * solved_energy

        self.result = [solved_latency, solved_energy, solved_edp]

        # ── Build trivial dataflow for simulator validation ───────────────────
        # Single dummy temporal loop (dimSize=1) at DRAM satisfies simulator's
        # non-empty tm requirement while being semantically equivalent to no loop.
        loops = LoopNest(acc=acc, ops=ops)
        loops.tm.append(Mapping(dim=1, dimSize=1,
                                mem=[acc.Dram2mem, acc.Dram2mem, acc.Dram2mem]))
        for u in range(acc.Num_SpUr):
            for d in range(1, ops.Num_dim):
                if self.su[u][d] > 1:
                    loops.sm.append(Mapping(dim=d, dimSize=self.su[u][d],
                                           mem=[acc.SpUr2Mem[u, op] for op in range(3)]))
        # The opMAC row (index Num_mem) stays 0: macro weight double buffering is
        # carried by double_config[Macro2mem][W].
        double_tag = [[0] * 3 for _ in range(acc.Num_mem + 1)]
        loops.usr_defined_double_flag = double_tag
        loops.psum_flag = {m: False for m in range(1, acc.Num_mem)}
        self.dataflow = loops

        # ── Profile ───────────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        self.profile.total_time_sec = elapsed
        self.profile.model_build_time_sec = elapsed
        self.profile.status = 2  # OPTIMAL equivalent
        self.profile.sol_count = 1
        self.profile.mip_gap = 0.0
        self.profile.num_vars = 0
        self.profile.num_constrs = 0

        Logger.critical(f"Trivial solution: Latency={solved_latency}, Energy={round(solved_energy, 3)}, "
                        f"EDP={solved_edp:.3e} (analytical, {elapsed:.4f}s)")

    def run(self):
        run_start_time = time.time()
        Logger.info('* '*20 + "Start Running MIP Solver" + ' *'*20)
        model = self.model
        acc:CIM_Acc = self.acc
        ops:WorkLoad = self.ops
        factors = self.FACTORS

        TEMP_DIVISORS = [getDivisors(d) for d in self.tu]

        Num_Loops = sum(len(f) for f in factors[1:ops.Num_dim] if f != [1])

        # ── Degenerate case: Num_Loops == 0 (spatial covers everything) ────────
        # All temporal factors are 1 → zero degrees of freedom → unique solution.
        # Compute analytically from MIP model formulas (L5/L6, per-memory energy).
        if Num_Loops == 0:
            self._solve_trivial(acc, ops, factors)
            return

        MAX_SIZE = self.ops.size

        UNIQUE_FACTOR = getUniqueFactors(factors)

        def best_prec(mem, op):
            if op != 2: # op_name != 'O'
                return acc.precision[mem, op]
            return acc.precision_final

        def worst_prec(mem, op):
            if op != 2: # op_name != 'O'
                return acc.precision[mem, op]
            return acc.precision_psum

        logF = {(d, f): math.log(factors[d][f]) 
                        for d in range(1, ops.Num_dim) 
                        for f in range(len(factors[d]))}
        INNERMOST_MEM = {
            0: {acc.IReg2mem},
            1: {acc.Macro2mem},
            2: {acc.OReg2mem},
        }
        
        factors_val = [f for fs in factors[1:ops.Num_dim] for f in fs if fs != [1]]
        f_asc = sorted(factors_val)
        f_desc = sorted(factors_val, reverse=True)

        MIN_INNER_PROD = {}
        for i in range(Num_Loops):
            MIN_INNER_PROD[i] = int(np.prod(f_asc[:Num_Loops - i]))
        MIN_INNER_PROD[Num_Loops] = 1

        MIN_OUTER_PROD = [1] * (Num_Loops + 1)
        for i in range(1, Num_Loops + 1):
            MIN_OUTER_PROD[i] = MIN_OUTER_PROD[i - 1] * f_asc[i - 1]

        LAT_UNIT = 1 / CONST.SCALE_LATENCY
        t_MAC = acc.t_MAC / CONST.SCALE_LATENCY
        MAX_TRANSFER_LEVELS = {0: min(4, Num_Loops), 1: min(2, Num_Loops), 2: min(4, Num_Loops)}

        TOTAL_TEMPORAL_ITERS = max(MIN_INNER_PROD[0], 1)
        MAX_STAGE_TRANSFER = max((2 if op == 2 else 1) * self.MAX_TRANS[op] for op in range(3))
        UB_offchipBootstrap = max(
            (2 if op == 2 else 1) * MAX_SIZE[op] * worst_prec(acc.Dram2mem, op) / acc.bw[acc.Dram2mem] / CONST.SCALE_LATENCY
            for op in range(3)
        )
        UB_latencySimple = max(
            LAT_UNIT,
            TOTAL_TEMPORAL_ITERS * max(t_MAC, MAX_STAGE_TRANSFER) + UB_offchipBootstrap,
        )
        if CONST.FLAG_OPT == "Latency" and self.metric_ub is not None and self.metric_ub < CONST.MAX_POS * 0.5:
            UB_latencySimple = min(UB_latencySimple, self.metric_ub / CONST.SCALE_LATENCY)
        UB_latencySimple = max(UB_latencySimple, LAT_UNIT)
        _f_max = max(UNIQUE_FACTOR) if UNIQUE_FACTOR else 2
        _UB_P = [0] * (Num_Loops + 1)
        _UB_P[Num_Loops] = t_MAC
        for _i in range(Num_Loops - 1, -1, -1):
            _UB_P[_i] = _f_max * (MAX_STAGE_TRANSFER + 2 * LAT_UNIT + _UB_P[_i + 1])
        UB_latencyLevel = {
            i: max(min(UB_latencySimple, _UB_P[i]), max(t_MAC, LAT_UNIT))
            for i in range(Num_Loops)
        }

        LB_Process = [0] * (Num_Loops + 1)
        LB_Process[Num_Loops] = t_MAC
        for _i in range(Num_Loops - 1, -1, -1):
            LB_Process[_i] = MIN_INNER_PROD[_i] * t_MAC

        # Critical[i] ≤ c_max*(Transfer+bypass) + Body; bypass adds at most 2*LAT_UNIT (c_max=2, 1 cycle each)
        UB_Critical = {}
        for _i in range(Num_Loops):
            _ub_child = UB_latencyLevel[_i + 1] if _i + 1 < Num_Loops else t_MAC
            UB_Critical[_i] = MAX_STAGE_TRANSFER + 2 * LAT_UNIT + _ub_child

        spur = {}
        UB_dataVolume, UB_lg_dataVolume = {}, {}
        UB_lg_transVolume = {}
        UB_transLatency, UB_lg_transLatency = {}, {}
        LB_lg_transLatency = {}
        LB_lg_transEnergy = math.log(1 / CONST.MAX_POS)
        for m in range(1, acc.Num_mem):
            for d in range(1, ops.Num_dim):
                for op in range(3):
                    spur[m, op, d] = 1
                    for u in range(acc.Num_SpUr):
                        if m <= acc.SpUr2Mem[u, op]:
                            spur[m, op, d] *= self.su[u][d]

            for op in range(3):
                UB_dataVolumeCap = max(1, min(acc.memSize[m] // best_prec(m, op), MAX_SIZE[op]))
                UB_dataVolume[m, op] = UB_dataVolumeCap
                UB_lg_dataVolume[m, op] = math.log(UB_dataVolumeCap)
                UB_lg_transVolume[m, op] = UB_lg_dataVolume[m, op]

                UB_transLatencyCap = max(
                    LAT_UNIT,
                    UB_dataVolumeCap * worst_prec(m, op) / acc.bw[m] / CONST.SCALE_LATENCY,
                )
                UB_transLatency[m, op] = UB_transLatencyCap
                UB_lg_transLatency[m, op] = math.log(UB_transLatencyCap)
                LB_lg_transLatency[m, op] = math.log(
                    max(best_prec(m, op) / acc.bw[m] / CONST.SCALE_LATENCY, 1 / CONST.MAX_POS)
                )

        # ── Spatial-unrolling lower bounds on tile volume and transLatency ──
        LB_transLatency_const = {}
        LB_lg_dataVolume, LB_lg_transVolume = {}, {}
        for m in range(1, acc.Num_mem):
            for op in range(3):
                if not acc.mappingArray[op][m]: continue
                _min_dims = [1] * ops.Num_dim
                for d in range(1, ops.Num_dim):
                    _min_dims[d] = spur[m, op, d]
                _min_vol = max(1, min(ops.get_operand_size(_min_dims, op), UB_dataVolume[m, op]))
                _prec = best_prec(m, op)  # must use best (smallest) precision for valid LB
                LB_transLatency_const[m, op] = max(LAT_UNIT, _min_vol * _prec / acc.bw[m] / CONST.SCALE_LATENCY)
                if m >= 2:
                    LB_lg_dataVolume[m, op] = math.log(max(1, _min_vol))
                    LB_lg_transVolume[m, op] = math.log(max(1, _min_vol))

        ###########################################################  Variable & Constant & Constraints  ##################################################################
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        res_latency = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="res_latency")
        res_energy = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="res_energy")
        res_EDP = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="res_EDP")

        # Finite, provably-valid upper bound on res_latency. The EDP McCormick
        # (res_EDP >= res_latency*res_energy under NonConvex=2) has an UNBOUNDED-below
        # underestimator whenever either factor is unbounded above, so the root
        # relaxation yields ObjBound = -inf and B&B cannot prune on the objective.
        # From the L5/L6 constraints res_latency = MaxStartup - SumTransfer[op]
        # + Process[0,op] + BootstrapWrite, with MaxStartup.ub = Process[0].ub =
        # UB_latencyLevel[0] and BootstrapWrite <= UB_offchipBootstrap and
        # SumTransfer >= 0, so res_latency <= 2*UB_latencyLevel[0] + UB_offchipBootstrap
        # for EVERY feasible mapping. Setting it as the var ub never removes a
        # mapping (the bound only tightens the bilinear relaxation) -> sound.
        res_latency.ub = 2.0 * UB_latencyLevel[0] + UB_offchipBootstrap

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        
        indic_factor2Loop = gp.tupledict()          # indic_factor2Loop[d,f,i] = {0,1}
                                                    
        for i in range(Num_Loops):
            for d in range(1, ops.Num_dim):
                if factors[d] == [1]:     # DimSize == 1
                    indic_factor2Loop[d,0,i] = 0
                    continue
                for f in range(len(factors[d])):
                    indic_factor2Loop[d,f,i] = model.addVar(vtype=GRB.BINARY, name=f"Indic_factor2Loop_({ops.dim2Dict[d]},{f},{i})")
            
        # # # # # Key Mapping Constraints # # # # # 
        for d in range(1, ops.Num_dim):
            if factors[d] == [1]:     # DimSize == 1
                continue
            for f in range(len(factors[d])):
                model.addConstr( quicksum(indic_factor2Loop[d,f,i] for i in range(Num_Loops)) == 1,
                                 name=f"C_Uniqueness_FactorMapping_({ops.dim2Dict[d]},{f})")
        for i in range(Num_Loops):
            model.addConstr( quicksum(indic_factor2Loop[d,f,i]
                                        for d in range(1, ops.Num_dim)
                                        for f in range(len(factors[d]))) == 1) 
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        # ['PLACEHOLD'0, 'Dram'1, 'Global_buffer'2, 'Output_buffer'3, 'Input_buffer'4, 'OReg'5, 'IReg'6, 'Macro'7] acc.Num_mem = 8

        indic_factor2Mem = gp.tupledict()           # indic_factor2Mem[d,f,op,m] = {0,1}
        for d in range(1, ops.Num_dim):
            if factors[d] == [1]:     # DimSize == 1
                for op, op_name in enumerate(['I','W','O']):
                    for m in range(1,acc.Num_mem):
                        indic_factor2Mem[d,0,op,m] = 0
                continue
            for f in range(len(factors[d])):
                for op, op_name in enumerate(['I','W','O']):
                    for m in range(1,acc.Num_mem):
                        if d == ops.dict2Dim('G') and m in INNERMOST_MEM[op]:
                            indic_factor2Mem[d,f,op,m] = 0
                        elif acc.mappingArray[op][m] == 1:
                            indic_factor2Mem[d,f,op,m] = model.addVar(vtype=GRB.BINARY, name=f"indic_factor2Mem_({ops.dim2Dict[d]},{f},{op_name},{acc.mem2dict(m)})")
                        else:
                            indic_factor2Mem[d,f,op,m] = 0
                    model.addConstr(quicksum(indic_factor2Mem[d,f,op,m] 
                                             for m in range(1,acc.Num_mem) ) == 1,
                                     name=f"C_Uniqueness_indic_factor2Mem_({ops.dim2Dict[d]},{f},{op_name})")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        # symmetry-breaking 对称约束
        for d in range(1, ops.Num_dim):
            if factors[d] == [1]:     # DimSize == 1
                continue
            for f in range(len(factors[d])):
                for f1 in range(f+1, len(factors[d])):
                    if factors[d][f] == factors[d][f1]:   # 相同因子之间才需要加对称约束
                        model.addConstr(quicksum(i * indic_factor2Loop[d,f,i] for i in range(Num_Loops)) <= 
                                        quicksum(i * indic_factor2Loop[d,f1,i] for i in range(Num_Loops)),
                                                name=f"C_SymmetryBreaking_Factor2Loop_({ops.dim2Dict[d]},{f},{f1})")

        # Fix factor-to-loop assignment for enumeration mode
        if self.fixed_factor_ordering is not None:
            FIXED_FACTOR_VAL_AT = {}  # FIXED_FACTOR_VAL_AT[i] = factor value at loop position i
            for d in range(1, ops.Num_dim):
                if factors[d] == [1]: continue
                for f in range(len(factors[d])):
                    target_i = self.fixed_factor_ordering[(d, f)]
                    FIXED_FACTOR_VAL_AT[target_i] = factors[d][f]
                    for i in range(Num_Loops):
                        var = indic_factor2Loop[d, f, i]
                        if isinstance(var, gp.Var):
                            var.lb = 1.0 if i == target_i else 0.0
                            var.ub = 1.0 if i == target_i else 0.0
        elif self.fixed_value_sequence is not None:
            # Value-sequence mode: pin only the factor VALUE at each position (via
            # FIXED_FACTOR_VAL_AT, consumed by the indic_loop2Factor fixing below) and
            # leave indic_factor2Loop (dimension->position) free for the MIP to choose.
            FIXED_FACTOR_VAL_AT = {i: v for i, v in enumerate(self.fixed_value_sequence)}
        elif self.fixed_partial_ordering is not None:
            # Partial-ordering mode (B&B node): pin the subset of factors named in the
            # dict to their positions (both factor2Loop here and loop2Factor below, so
            # Z_crit is exact at those positions); everything else stays free.
            FIXED_FACTOR_VAL_AT = {}
            for (d, f), target_i in self.fixed_partial_ordering.items():
                FIXED_FACTOR_VAL_AT[target_i] = factors[d][f]
                for i in range(Num_Loops):
                    var = indic_factor2Loop[d, f, i]
                    if isinstance(var, gp.Var):
                        var.lb = 1.0 if i == target_i else 0.0
                        var.ub = 1.0 if i == target_i else 0.0

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

        indic_usedMem = gp.tupledict()            # indic_usedMem[m,op] = {0,1}           
        for m in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 1:
                    indic_usedMem[m,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_usedMem_({acc.mem2dict(m)},{op_name})")
                    
                    sum_factors_in_mem = quicksum(indic_factor2Mem[d,f,op,m] 
                                                for d in range(1, ops.Num_dim) if factors[d]!=[1]
                                                for f in range(len(factors[d])))

                    model.addConstr(sum_factors_in_mem <= Num_Loops * indic_usedMem[m,op], name=f"C_UsedMem_Trigger_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(indic_usedMem[m,op] <= sum_factors_in_mem, name=f"C_UsedMem_Limit_({acc.mem2dict(m)},{op_name})")
                
                else:
                    indic_usedMem[m,op] = 0
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
            
        indic_loop2Mem = gp.tupledict()           # indic_loop2Mem[i,op,m] = {0,1}
        for op, op_name in enumerate(['I','W','O']):
            for i in range(Num_Loops):
                for m in range(1,acc.Num_mem):
                    if acc.mappingArray[op][m] == 1:
                        indic_loop2Mem[i,op,m] = model.addVar(vtype=GRB.BINARY, name=f"Indic_loop2Mem_({i},{op_name},{acc.mem2dict(m)})")
                        for d in range(1, ops.Num_dim):
                            if factors[d] == [1]:
                                continue
                            for f in range(len(factors[d])):
                                model.addConstr(indic_loop2Mem[i,op,m] >= indic_factor2Mem[d,f,op,m] + indic_factor2Loop[d,f,i]-1,
                                                name=f"C_Indic_loop2Mem_({i},{op_name},{acc.mem2dict(m)},{ops.dim2Dict[d]},{f})")
                                model.addConstr(indic_loop2Mem[i,op,m] <= 1 - indic_factor2Loop[d,f,i] + indic_factor2Mem[d,f,op,m],
                                                name=f"C_Indic_loop2Mem_Disagg_({i},{op_name},{acc.mem2dict(m)},{ops.dim2Dict[d]},{f})")
                    else:
                        indic_loop2Mem[i,op,m] = 0

                    model.addConstr(indic_loop2Mem[i,op,m] <= indic_usedMem[m,op], name=f"C_Loop2Mem_Used_({i},{op_name},{acc.mem2dict(m)})")

                model.addConstr(quicksum(indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem)) == 1, 
                                name=f"C_Indic_loop2Mem_({i},{op_name})")
                        
        for i in range(Num_Loops-1):
            for op, op_name in enumerate(['I','W','O']):
                model.addConstr(quicksum(m*indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem)) <=
                                quicksum(m*indic_loop2Mem[i+1,op,m] for m in range(1,acc.Num_mem)),
                                  name=f"C_Sequence_loop2Mem_({i},{op_name})")

        for m in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 1:
                    model.addConstr(indic_usedMem[m,op] <= quicksum(indic_loop2Mem[i,op,m] for i in range(Num_Loops)),
                                    name=f"C_UsedMem_FromLoop2Mem_({acc.mem2dict(m)},{op_name})")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 
        
        indic_loop2Factor = gp.tupledict()              # indic_loop2Factor[i,p] = {0,1}
        indic_relevantLoop = gp.tupledict()             # indic_relevantLoop[i,op] = {0,1} : whether loop i is relevant to op
        for i in range(Num_Loops):
            for p in range(len(UNIQUE_FACTOR)):
                indic_loop2Factor[i,p] = model.addVar(vtype=GRB.BINARY, name=f"Indic_loop2Factor_({i},{p})")
            model.addConstr(quicksum(indic_loop2Factor[i,p] for p in range(len(UNIQUE_FACTOR))) == 1, name=f"C_Uniqueness_Indic_loop2Factor_({i})")
            model.addConstr(quicksum(UNIQUE_FACTOR[p] * indic_loop2Factor[i,p] for p in range(len(UNIQUE_FACTOR)))
                            ==
                            quicksum(indic_factor2Loop[d,f,i] * factors[d][f] for d in range(1, ops.Num_dim) for f in range(len(factors[d]))),
                             name=f"C_loop2Factor_definition_({i})")
            for op, op_name in enumerate(['I','W','O']):
                indic_relevantLoop[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_relevantLoop_({i},{op_name})")
                model.addConstr(indic_relevantLoop[i,op] == quicksum(ops.relevance[op][d] * indic_factor2Loop[d,f,i] 
                                                                     for d in range(1, ops.Num_dim) for f in range(len(factors[d]))),
                                 name=f"C_relevantLoop_({i},{op_name})")

        # Fix loop2Factor in enumeration mode (eliminates Z_crit McCormick gap — Theorem 3).
        # Triggered by full-ordering mode OR value-sequence mode; both build
        # FIXED_FACTOR_VAL_AT above. Value-sequence mode leaves indic_factor2Loop free.
        if (self.fixed_factor_ordering is not None or self.fixed_value_sequence is not None
                or self.fixed_partial_ordering is not None):
            for i in range(Num_Loops):
                target_val = FIXED_FACTOR_VAL_AT.get(i)
                if target_val is not None:
                    for p in range(len(UNIQUE_FACTOR)):
                        var = indic_loop2Factor[i, p]
                        if isinstance(var, gp.Var):
                            var.lb = 1.0 if UNIQUE_FACTOR[p] == target_val else 0.0
                            var.ub = 1.0 if UNIQUE_FACTOR[p] == target_val else 0.0

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#

        KEEP_temp_Output_irDim = any(factors[d] != [1] and ops.relevance[2][d] == 0 for d in range(1, ops.Num_dim))

        indic_hasIrrelevantDim = gp.tupledict()                # indic_hasIrrelevantDim[m] = {0,1}
        for m in range(1, acc.Num_mem):
            op, op_name = 2, 'O'
            if acc.mappingArray[op][m] == 0 or (not KEEP_temp_Output_irDim):
                indic_hasIrrelevantDim[m] = 0
                continue
            else:
                indic_hasIrrelevantDim[m] = model.addVar(vtype=GRB.BINARY, name=f"indic_hasIrrelevantDim_({acc.mem2dict(m)})")

                sum_ir_terms = 0
                for d in range(1, ops.Num_dim):
                    if factors[d] == [1] or ops.relevance[op][d] == 1:
                        continue
                    for f in range(len(factors[d])):
                        sum_ir_terms += indic_factor2Mem[d, f, op, m]
                        model.addConstr(indic_hasIrrelevantDim[m] >= indic_factor2Mem[d, f, op, m],
                                        name=f"C_has_ir_lb_({acc.mem2dict(m)},{ops.dim2Dict[d]},{f})")
                model.addConstr(indic_hasIrrelevantDim[m] <= sum_ir_terms, name=f"C_has_ir_ub_({acc.mem2dict(m)})")

        indic_holdPartialSum = gp.tupledict()                     # indic_holdPartialSum[m] = {0,1}
        for m in range(1, acc.Num_mem):
            op, op_name = 2, 'O'
            if acc.mappingArray[op][m] == 0 or (not KEEP_temp_Output_irDim):
                indic_holdPartialSum[m] = 0
                continue
            else:
                indic_holdPartialSum[m] = model.addVar(vtype=GRB.BINARY, name=f"indic_holdPartialSum_({acc.mem2dict(m)})")

                model.addConstr( m*indic_holdPartialSum[m] >= quicksum(indic_hasIrrelevantDim[m1] for m1 in range(1,m+1)),
                                name=f"C_indic_holdPartialSum_({acc.mem2dict(m)})_1")
                model.addConstr( indic_holdPartialSum[m] <= quicksum(indic_hasIrrelevantDim[m1] for m1 in range(1,m+1)),
                                name=f"C_indic_holdPartialSum_({acc.mem2dict(m)})_2")

        def lg_prec_O(mem):
            state = indic_holdPartialSum[mem]
            if isinstance(state, gp.Var):
                return math.log(acc.precision_final) + (math.log(acc.precision_psum) - math.log(acc.precision_final)) * state
            return math.log(acc.precision_psum) if state else math.log(acc.precision_final)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        for op, op_name in enumerate(['I','W','O']):
            acc.mappingArray[op].append(1)
            indic_usedMem[acc.Num_mem,op] = 1
        indic_nextMem = gp.tupledict()
        for m in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 0:
                    continue

                sum_nxt = 0
                for m1 in range(m+1, acc.Num_mem+1):
                    if acc.mappingArray[op][m1] == 0:
                        indic_nextMem[m,m1,op] = 0
                        continue
                        
                    indic_nextMem[m,m1,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_nextMem_({acc.mem2dict(m)},{acc.mem2dict(m1)},{op_name})")

                    model.addConstr(indic_nextMem[m,m1,op] <= indic_usedMem[m1,op], name=f"C_nxtMem_Valid_({acc.mem2dict(m)},{acc.mem2dict(m1)},{op_name})")
                    for m2 in range(m+1, m1):
                        if acc.mappingArray[op][m2] == 1:
                            model.addConstr(indic_nextMem[m, m1, op] <= 1 - indic_usedMem[m2, op],
                                            name=f"C_NxtMem_Block_({acc.mem2dict(m)},{acc.mem2dict(m1)},{op_name})_by_({acc.mem2dict(m2)})")
                    
                    sum_nxt += indic_nextMem[m,m1,op]

                model.addConstr(sum_nxt == indic_usedMem[m,op])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -# 

        indic_doubleMem = gp.tupledict()        # indic_doubleMem[m,op] = {0,1}
        indic_doubleLoop = gp.tupledict()       # indic_doubleloop[i,op] = {0,1} 
        indic_feedsDoubleBuf = gp.tupledict()         # indic_feedsDoubleBuf[m,op] = {0,1}  表示 Memory m 的下一级是否双缓冲（无论是通过 Bypass 还是直接连接）

        for m in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.double_config[m][op] == 0:
                    indic_doubleMem[m,op] = 0
                else:
                    indic_doubleMem[m,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_doubleMem_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(indic_doubleMem[m,op] <= indic_usedMem[m,op], name=f"C_DoubleMem_Valid_{m}_{op_name}")

        for m in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 0:
                    indic_feedsDoubleBuf[m, op] = 0
                    continue
                indic_feedsDoubleBuf[m, op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_feedsDoubleBuf_({acc.mem2dict(m)},{op_name})")
                
                quad_expr = 0
                for m1 in range(m+1, acc.Num_mem): # 注意边界，假设 acc.Num_mem 是最大索引(如Macro)
                    if acc.mappingArray[op][m1]:
                        if acc.double_config[m1][op] == 1:
                            quad_expr += var_AandB(model, indic_nextMem[m,m1,op], indic_doubleMem[m1,op],
                                                    name=f"Quad_Indic_feedsDB_({acc.mem2dict(m)},{op_name})_to_{acc.mem2dict(m1)}")
                
                model.addConstr(indic_feedsDoubleBuf[m,op] == quad_expr, name=f"C_indic_feedsDoubleBuf_{m}_{op_name}")

        for i in range(Num_Loops):
            for op, op_name in enumerate(['I','W','O']):
                indic_doubleLoop[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_doubleLoop_({i},{op_name})")
                
                quad_expr_loop = 0
                for m in range(1, acc.Num_mem):
                    if acc.mappingArray[op][m]:
                        quad_expr_loop += var_AandB(model, indic_loop2Mem[i,op,m], indic_feedsDoubleBuf[m,op],
                                                     name=f"Quad_Indic_doubleLoop_({i},{op_name})_{acc.mem2dict(m)}")
                
                model.addConstr(indic_doubleLoop[i,op] == quad_expr_loop, name=f"C_Indic_doubleLoop_({i},{op_name})")
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#    
        
        indic_hasTransfer = gp.tupledict()                         # indic_hasTransfer[i,op] = {0,1}
        indic_hasTransferCarry = gp.tupledict()                    # indic_hasTransferCarry[i,op] = {0,1}
        for op, op_name in enumerate(['I', 'W', 'O']):
            indic_sameMem, indic_memBoundary = {}, {}

            for i in range(Num_Loops):
                if i == Num_Loops-1:
                    indic_memBoundary[i] = 1 - indic_loop2Mem[i,op,acc.Macro2mem] if op_name == 'W' else 1 
                else:
                    indic_memBoundary[i] = model.addVar(vtype=GRB.BINARY, name=f"Indic_memBoundary_({i},{op_name})")
                    tmp_diff = quicksum(m*indic_loop2Mem[i+1,op,m] for m in range(1,acc.Num_mem)) - quicksum(m*indic_loop2Mem[i,op,m] for m in range(1,acc.Num_mem))
                    model.addConstr(tmp_diff >= indic_memBoundary[i], name=f"C_tmp_diff_({i},{op_name})")
                    model.addConstr(tmp_diff <= acc.Num_mem * indic_memBoundary[i], name=f"C_tmp_diff_ub_({i},{op_name})")
                indic_sameMem[i] = 1 - indic_memBoundary[i]

            for i in range(Num_Loops-1, -1, -1):
                '''
                indic_hasTransferCarry[i, op] = 
                        indic_memBoundary[i, op] OR (indic_sameMem[i, op] AND indic_hasTransferCarry[i+1,op] AND (NOT indic_relevantLoop[i+1, op]))
                '''
                if i == Num_Loops-1:
                    indic_hasTransferCarry[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_transferCarry_({i},{op_name})")
                    model.addConstr(indic_hasTransferCarry[i,op] == indic_memBoundary[i],
                                    name=f"C_transferCarry_last_({op_name})")
                else:
                    indic_hasTransferCarry[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_transferCarry_({i},{op_name})")
                    model.addConstr(indic_hasTransferCarry[i,op] >= indic_memBoundary[i], name=f"C_transferCarry_boundary_({i},{op_name})")
                    model.addConstr(indic_hasTransferCarry[i,op] <= indic_memBoundary[i] + indic_hasTransferCarry[i+1,op],
                                    name=f"C_transferCarry_next_ub_({i},{op_name})")
                    model.addConstr(indic_hasTransferCarry[i,op] <= indic_memBoundary[i] + 1 - indic_relevantLoop[i+1, op],
                                    name=f"C_transferCarry_irrel_ub_({i},{op_name})")
                    model.addConstr(indic_hasTransferCarry[i,op] >= indic_sameMem[i] + indic_hasTransferCarry[i+1,op] - indic_relevantLoop[i+1, op] - 1,
                                    name=f"C_transferCarry_same_lb_({i},{op_name})")

            for i in range(Num_Loops): 
                '''
                indic_hasTransfer[i, op] = indic_hasTransferCarry[i, op] AND (indic_relevantLoop[i, op] OR (NOT indic_sameMem[i-1, op]))
                '''       
                if i == 0:
                    indic_hasTransfer[i,op] = indic_hasTransferCarry[i,op]
                else:
                    indic_hasTransfer[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_hasTransfer_({i},{op_name})")
                    model.addConstr(indic_hasTransfer[i, op] <= indic_hasTransferCarry[i,op], name=f"C_hasTransfer_carry_({i},{op_name})")
                    model.addConstr(indic_hasTransfer[i, op] <= indic_relevantLoop[i, op] + 1 - indic_sameMem[i-1],
                                    name=f"C_hasTransfer_assignable_ub_({i},{op_name})")
                    model.addConstr(indic_hasTransfer[i, op] >= indic_hasTransferCarry[i,op] - indic_sameMem[i-1],
                                    name=f"C_hasTransfer_blockstart_lb_({i},{op_name})")
                    model.addConstr(indic_hasTransfer[i, op] >= indic_hasTransferCarry[i,op] + indic_relevantLoop[i, op] + indic_sameMem[i-1] - 2,
                                    name=f"C_hasTransfer_relevant_lb_({i},{op_name})")
                if op_name == 'W':
                    model.addConstr(indic_hasTransfer[i,op] <= 1 - indic_loop2Mem[i, op, acc.Macro2mem], name=f"C_hasTransfer_macro_({i},{op_name})")
                    # CIM inherent mapping constraint

            model.addConstr(quicksum(indic_hasTransfer[i,op] for i in range(Num_Loops)) <= MAX_TRANSFER_LEVELS[op], name=f"max_hasTransfer_{op_name}")
            # Valid inequality: #xMem boundaries >= #used_memory_levels - 1
            _sum_used = quicksum(indic_usedMem[m,op] for m in range(1, acc.Num_mem) if acc.mappingArray[op][m])
            if Num_Loops > 1:
                model.addConstr(quicksum(indic_hasTransfer[i,op] for i in range(Num_Loops)) >= _sum_used - 1,
                                name=f"min_hasTransfer_from_usedMem_{op_name}")

        # SOS1 declarations: guide Gurobi to use efficient SOS branching on "exactly-one" indicator groups.
        # Not a constraint (already implied by Σ=1 + binary), but changes branching from linear to log depth.
        for i in range(Num_Loops):
            _s = [indic_loop2Factor[i,p] for p in range(len(UNIQUE_FACTOR)) if isinstance(indic_loop2Factor.get((i,p), 0), gp.Var)]
            if len(_s) > 1: model.addSOS(GRB.SOS_TYPE1, _s)
        for d in range(1, ops.Num_dim):
            if factors[d] == [1]: continue
            for f in range(len(factors[d])):
                _s = [indic_factor2Loop[d,f,i] for i in range(Num_Loops) if isinstance(indic_factor2Loop.get((d,f,i), 0), gp.Var)]
                if len(_s) > 1: model.addSOS(GRB.SOS_TYPE1, _s)
            for f in range(len(factors[d])):
                for op in range(3):
                    _s = [indic_factor2Mem[d,f,op,m] for m in range(1, acc.Num_mem) if isinstance(indic_factor2Mem.get((d,f,op,m), 0), gp.Var)]
                    if len(_s) > 1: model.addSOS(GRB.SOS_TYPE1, _s)
        for i in range(Num_Loops):
            for op in range(3):
                _s = [indic_loop2Mem[i,op,m] for m in range(1, acc.Num_mem) if isinstance(indic_loop2Mem.get((i,op,m), 0), gp.Var)]
                if len(_s) > 1: model.addSOS(GRB.SOS_TYPE1, _s)

        # Branch priority: resolve factor assignment (combinatorial core) first,
        # then memory assignment. Once both are fixed, remaining continuous/buffer decisions are trivial.
        for d in range(1, ops.Num_dim):
            if factors[d] == [1]: continue
            for f in range(len(factors[d])):
                for i in range(Num_Loops):
                    v = indic_factor2Loop.get((d,f,i), None)
                    if isinstance(v, gp.Var): v.BranchPriority = 100
        for i in range(Num_Loops):
            for op in range(3):
                for m in range(1, acc.Num_mem):
                    v = indic_loop2Mem.get((i,op,m), None)
                    if isinstance(v, gp.Var): v.BranchPriority = 50

        ####################################################################  Capacity Constraints   #######################################################################

        lg_dimExistMem = gp.tupledict()         # lg_dimExistMem[m,op,d] = dimSize
        for op, op_name in enumerate(['I','W','O']):
            for d in range(1, ops.Num_dim):
                for m in range(2,acc.Num_mem):
                    if acc.mappingArray[op][m] == 0 or ops.relevance[op][d] == 0:
                        continue
                    if factors[d] == [1]:     # DimSize == 1
                        lg_dimExistMem[m,op,d] = 0
                        for u in range(acc.Num_SpUr):
                            if m <= acc.SpUr2Mem[u,op]:
                                lg_dimExistMem[m,op,d] += math.log(self.su[u][d])
                        continue
                    else:    
                        lg_dimExistMem[m,op,d] = model.addVar(lb=0, ub=math.log(ops.dim2bound[d]), vtype=GRB.CONTINUOUS,
                                                            name=f"lg_dimExistMem_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
                        dimExistMem = 0
                        for m1 in range(m,acc.Num_mem):
                            if acc.mappingArray[op][m1] == 1:
                                for f in range(len(factors[d])):
                                    dimExistMem += logF[d,f]*indic_factor2Mem[d,f,op,m1]
                            
                        for u in range(acc.Num_SpUr):
                            if m <= acc.SpUr2Mem[u,op]:
                                dimExistMem += math.log(self.su[u][d])
                                
                        model.addConstr(lg_dimExistMem[m,op,d] == dimExistMem, 
                                                    name=f"C_sum_lgdimExistMem_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
                lg_dimExistMem[1,op,d] = math.log(ops.dim2bound[d])
                        
        lg_dimOfTile = gp.tupledict()         # lg_dimOfTile[m,op,d] = dimSize
        for m in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                for d in range(1, ops.Num_dim):
                    if acc.mappingArray[op][m] == 0 or ops.relevance[op][d] == 0:
                        continue
                    if factors[d] == [1]:     # DimSize == 1
                        lg_dimOfTile[m,op,d] = 0
                        for u in range(acc.Num_SpUr):
                            if m <= acc.SpUr2Mem[u,op]:
                                lg_dimOfTile[m,op,d] += math.log(self.su[u][d])
                        continue
                    else:    
                        lg_dimOfTile[m,op,d] = model.addVar(lb=0, ub=math.log(ops.dim2bound[d]), vtype=GRB.CONTINUOUS,
                                                            name=f"lg_dimOfTile_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
                        dimOfTile = 0
                        for m1 in range(m+1,acc.Num_mem):               #  m -> m+1
                            if acc.mappingArray[op][m1] == 1:
                                for f in range(len(factors[d])):
                                    dimOfTile += logF[d,f]*indic_factor2Mem[d,f,op,m1]
                            
                        for u in range(acc.Num_SpUr):
                            if m <= acc.SpUr2Mem[u,op]:
                                dimOfTile += math.log(self.su[u][d])
                                
                        model.addConstr(lg_dimOfTile[m,op,d] == dimOfTile, 
                                                    name=f"C_sum_lgdimOfTile_({acc.mem2dict(m)},{op_name},{ops.dim2Dict[d]})")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          

        exp_dataVolume = gp.tupledict()     # exp_dataVolume[m,op]
        lg_dataVolume = gp.tupledict()     # lg_dataVolume[m,op]
        for op, op_name in enumerate(['I','W','O']):
            lg_dataVolume[1, op] = math.log(MAX_SIZE[op])
            exp_dataVolume[1, op] = MAX_SIZE[op]
        for m in range(2,acc.Num_mem):      # Sufficient off-chip capacity [1-Dram]
            op, op_name = 0,'I'     # Input
            if acc.mappingArray[op][m] == True:
                sum_r, sum_s, sum_c, sum_p, sum_q, = 0,0,0,0,0
                indic_sum, sum_dim_h = 0, 0
                for i_r, rd in enumerate(TEMP_DIVISORS[ops.dict2Dim('R')]):
                    for i_p, pd in enumerate(TEMP_DIVISORS[ops.dict2Dim('P')]):
                        r = rd * spur[m,op,ops.dict2Dim('R')]
                        p = pd * spur[m,op,ops.dict2Dim('P')]
                        h = min(r + (p - 1) * min(ops.Stride, r), ops.H)
                        indic_dim = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_dim_Height_({acc.mem2dict(m)},{op_name},{i_r},{i_p})")
                        sum_dim_h += indic_dim * math.log(h)
                        sum_r += indic_dim * math.log(r)
                        sum_p += indic_dim * math.log(p)
                        indic_sum += indic_dim
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicSum_Height_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_r == lg_dimExistMem[m,op,ops.dict2Dim('R')], name=f"C_Uniqueness_IndicSumR_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_p == lg_dimExistMem[m,op,ops.dict2Dim('P')], name=f"C_Uniqueness_IndicSumP_({acc.mem2dict(m)},{op_name})")

                indic_sum, sum_dim_w = 0, 0
                for i_s, sd in enumerate(TEMP_DIVISORS[ops.dict2Dim('S')]):
                    for i_q, qd in enumerate(TEMP_DIVISORS[ops.dict2Dim('Q')]):
                        s = sd * spur[m,op,ops.dict2Dim('S')]
                        q = qd * spur[m,op,ops.dict2Dim('Q')]
                        w = min(s + (q - 1) * min(ops.Stride, s), ops.W)
                        indic_dim = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_dim_Width_({acc.mem2dict(m)},{op_name},{i_s},{i_q})")
                        sum_dim_w += indic_dim * math.log(w)
                        sum_s += indic_dim * math.log(s)
                        sum_q += indic_dim * math.log(q)
                        indic_sum += indic_dim
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicSum_Width_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_s == lg_dimExistMem[m,op,ops.dict2Dim('S')], name=f"C_Uniqueness_IndicSumS_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_q == lg_dimExistMem[m,op,ops.dict2Dim('Q')], name=f"C_Uniqueness_IndicSumQ_({acc.mem2dict(m)},{op_name})")

                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_dataVolume.get((m,op), 0), ub=UB_lg_dataVolume[m,op],
                                                    name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == sum_dim_h + sum_dim_w + lg_dimExistMem[m,op,ops.dict2Dim('C')] + lg_dimExistMem[m,op,ops.dict2Dim('G')],
                                    name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=0, ub=UB_dataVolume[m,op], vtype=GRB.CONTINUOUS,
                                                         name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_dataVolume[m,op], yvar=exp_dataVolume[m,op], options=CONST.ExpOption, name=f"C_exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                        
            op, op_name = 1,'W'     # Weight
            if acc.mappingArray[op][m] == True:
                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_dataVolume.get((m,op), 0), ub=UB_lg_dataVolume[m,op],
                                                name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == quicksum(lg_dimExistMem[m,op,ops.dict2Dim(dChar)] for dChar in ['R','S','C','K','G']),
                                name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=0, ub=UB_dataVolume[m,op], vtype=GRB.CONTINUOUS,
                                                        name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_dataVolume[m,op], yvar=exp_dataVolume[m,op], options=CONST.ExpOption, name=f"C_exp_dataVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 2,'O'     # Output
            if acc.mappingArray[op][m] == True:
                lg_dataVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_dataVolume.get((m,op), 0), ub=UB_lg_dataVolume[m,op],
                                                    name=f"lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_dataVolume[m,op] == quicksum(lg_dimExistMem[m,op,ops.dict2Dim(dChar)] for dChar in ['P','Q','K','G']),
                                    name=f"C_lg_dataVolume_({acc.mem2dict(m)},{op_name})")
                if acc.shareMemory[m] == True:
                    exp_dataVolume[m,op] = model.addVar(lb=0, ub=UB_dataVolume[m,op], vtype=GRB.CONTINUOUS,
                                                        name=f"exp_dataVolume_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_dataVolume[m,op], yvar=exp_dataVolume[m,op], options=CONST.ExpOption, name=f"C_exp_dataVolume_({acc.mem2dict(m)},{op_name})")

        lg_transVolume = gp.tupledict()     # lg_transVolume[m,op]
        for m in range(1,acc.Num_mem):      

            op, op_name = 0,'I'     # Input
            if acc.mappingArray[op][m]:
                sum_r, sum_s, sum_c, sum_p, sum_q, = 0,0,0,0,0

                indic_sum, sum_dim_h = 0, 0
                for i_r, rd in enumerate(TEMP_DIVISORS[ops.dict2Dim('R')]):
                    for i_p, pd in enumerate(TEMP_DIVISORS[ops.dict2Dim('P')]):
                        r = rd * spur[m,op,ops.dict2Dim('R')]
                        p = pd * spur[m,op,ops.dict2Dim('P')]
                        h = min(r + (p - 1) * min(ops.Stride, r), ops.H)
                        indic_dim = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_dim_TileHeight_({acc.mem2dict(m)},{op_name},{i_r},{i_p})")
                        sum_dim_h += indic_dim * math.log(h)
                        sum_r += indic_dim * math.log(r)
                        sum_p += indic_dim * math.log(p)
                        indic_sum += indic_dim
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicTile_Height_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_r == lg_dimOfTile[m,op,ops.dict2Dim('R')], name=f"C_Uniqueness_IndicTileR_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_p == lg_dimOfTile[m,op,ops.dict2Dim('P')], name=f"C_Uniqueness_IndicTileP_({acc.mem2dict(m)},{op_name})")

                indic_sum, sum_dim_w = 0, 0
                for i_s, sd in enumerate(TEMP_DIVISORS[ops.dict2Dim('S')]):
                    for i_q, qd in enumerate(TEMP_DIVISORS[ops.dict2Dim('Q')]):
                        s = sd * spur[m,op,ops.dict2Dim('S')]
                        q = qd * spur[m,op,ops.dict2Dim('Q')]
                        w = min(s + (q - 1) * min(ops.Stride, s), ops.W)
                        indic_dim = model.addVar(vtype=gp.GRB.BINARY, name=f"Indic_dim_TileWidth_({acc.mem2dict(m)},{op_name},{i_s},{i_q})")
                        sum_dim_w += indic_dim * math.log(w)
                        sum_s += indic_dim * math.log(s)
                        sum_q += indic_dim * math.log(q)
                        indic_sum += indic_dim
                model.addConstr(indic_sum == 1, name=f"C_Uniqueness_IndicTile_Width_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_s == lg_dimOfTile[m,op,ops.dict2Dim('S')], name=f"C_Uniqueness_IndicTileS_({acc.mem2dict(m)},{op_name})")
                model.addConstr(sum_q == lg_dimOfTile[m,op,ops.dict2Dim('Q')], name=f"C_Uniqueness_IndicTileQ_({acc.mem2dict(m)},{op_name})")

                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_transVolume.get((m,op), 0), ub=UB_lg_transVolume[m,op],
                                                     name=f"lg_transVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_transVolume[m,op] == sum_dim_h + sum_dim_w + lg_dimOfTile[m,op,ops.dict2Dim('C')] + lg_dimOfTile[m,op,ops.dict2Dim('G')],
                                    name=f"C_lg_transVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 1,'W'     # Weight
            if acc.mappingArray[op][m]:
                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_transVolume.get((m,op), 0), ub=UB_lg_transVolume[m,op],
                                                     name=f"lg_transVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_transVolume[m,op] == quicksum(lg_dimOfTile[m,op,ops.dict2Dim(dChar)] for dChar in ['R','S','C','K','G']),
                                     name=f"C_lg_transVolume_({acc.mem2dict(m)},{op_name})")

            op, op_name = 2,'O'     # Output
            if acc.mappingArray[op][m]:
                lg_transVolume[m,op] = model.addVar(vtype=GRB.CONTINUOUS, lb=LB_lg_transVolume.get((m,op), 0), ub=UB_lg_transVolume[m,op],
                                                     name=f"lg_transVolume_({acc.mem2dict(m)},{op_name})")
                model.addConstr(lg_transVolume[m,op] == quicksum(lg_dimOfTile[m,op,ops.dict2Dim(dChar)] for dChar in ['P','Q','K','G']),
                                     name=f"C_lg_transVolume_({acc.mem2dict(m)},{op_name})")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#          

        for m in range(2,acc.Num_mem):
            if acc.shareMemory[m] == True:
                tmp_datavolume = 0
                for op, op_name in enumerate(['I','W','O']):
                    if acc.mappingArray[op][m] == 0:
                        continue

                    tmp_volume = var_mul01(model, indic_usedMem[m,op], exp_dataVolume[m,op], f"dataVolume_used_({acc.mem2dict(m)},{op_name})")
                    tmp_volume += var_mul01(model, indic_doubleMem[m,op], exp_dataVolume[m,op], f"dataVolume_double_{m}_{op}")

                    if op_name != 'O':
                        tmp_datavolume += tmp_volume * acc.precision[m,op]
                    else:
                        vol_total_o = model.addVar(lb=0, ub=2 * UB_dataVolume[m, op], vtype=GRB.CONTINUOUS,
                                                name=f"vol_total_({acc.mem2dict(m)},{op_name})")
                        model.addConstr(vol_total_o == tmp_volume,
                                        name=f"C_vol_total_({acc.mem2dict(m)},{op_name})")

                        vol_psum_o = var_mul01(model, indic_holdPartialSum[m], vol_total_o,
                                            name=f"vol_psum_({acc.mem2dict(m)},{op_name})",
                                            A_ub=2 * UB_dataVolume[m, op],
                                            var_ub=2 * UB_dataVolume[m, op])
                        tmp_datavolume += vol_total_o * acc.precision_final
                        tmp_datavolume += vol_psum_o * (acc.precision_psum - acc.precision_final)
                model.addConstr( tmp_datavolume <= acc.memSize[m], name=f"C_dataVolume_({acc.mem2dict(m)})" )
            else:
                for op, op_name in enumerate(['I','W','O']):
                    if acc.mappingArray[op][m] == 0:
                        continue
                    if op_name == 'O':
                        model.addConstr(lg_dataVolume[m,op] + math.log(2)*indic_doubleMem[m,op] <= math.log(acc.memSize[m]) - lg_prec_O(m),
                                        name=f"C_dataVolume_({acc.mem2dict(m)},{op_name})")
                    else:
                        model.addConstr(lg_dataVolume[m,op] + math.log(2)*indic_doubleMem[m,op] <= math.log(acc.memSize[m])-math.log(acc.precision[m,op]),
                                        name=f"C_dataVolume_({acc.mem2dict(m)},{op_name})")

        ####################################################################  Execution Performance   #######################################################################

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - Energy - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#                  
        
        count_mac = 1
        count_core = 1
        for d in range(1, ops.Num_dim):
            for f in range(len(factors[d])):
                count_mac *= factors[d][f]
            count_core *= self.su[0][d]
        energy_expr_comp = acc.cost_ActMacro * count_mac * count_core

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#   

        # A loop can be irrelevant to the operand yet still trigger a real transfer.
        # Only the stationary irrelevant suffix inside a memory block should be filtered.
        indic_stationaryIrrelevant = gp.tupledict()        # irrelevant carry loop that is not an actual xMem transfer
        for i in range(Num_Loops):
            for op, op_name in enumerate(['I','W','O']):
                indic_stationaryIrrelevant[i,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_stationaryIrrelevant_({i},{op_name})")
                model.addConstr(indic_stationaryIrrelevant[i,op] <= indic_hasTransferCarry[i,op], name=f"C_stationaryIrrelevant_carry_({i},{op_name})")
                model.addConstr(indic_stationaryIrrelevant[i,op] <= 1 - indic_relevantLoop[i,op], name=f"C_stationaryIrrelevant_irrel_({i},{op_name})")
                model.addConstr(indic_stationaryIrrelevant[i,op] <= 1 - indic_hasTransfer[i,op], name=f"C_stationaryIrrelevant_noX_({i},{op_name})")
                model.addConstr(indic_stationaryIrrelevant[i,op] >= indic_hasTransferCarry[i,op] - indic_relevantLoop[i,op] - indic_hasTransfer[i,op],
                                name=f"C_stationaryIrrelevant_lb_({i},{op_name})")
                '''
                indic_stationaryIrrelevant[i, op] = xMemCarry AND NOT(relevantLoop OR xMem)
                '''

        indic_factor2Mem_effective = gp.tupledict()
        indic_effectiveFactor = gp.tupledict()
        for op, op_name in enumerate(['I','W','O']):
            for d in range(1, ops.Num_dim):
                if factors[d] == [1]:
                    continue
                for f in range(len(factors[d])):
                    if ops.relevance[op][d] == 1:
                        indic_effectiveFactor[d,f,op] = 1
                        for m in range(1, acc.Num_mem):
                            if acc.mappingArray[op][m] == 0:
                                continue
                            indic_factor2Mem_effective[d,f,op,m] = indic_factor2Mem[d,f,op,m]
                    else:
                        indic_effectiveFactor[d,f,op] = model.addVar(vtype=GRB.BINARY, name=f"Indic_NotBottleIR_({ops.dim2Dict[d]},{f},{op_name})")
                        model.addConstr(
                            indic_effectiveFactor[d,f,op] == 1 - quicksum(var_AandB(model, indic_factor2Loop[d,f,i], indic_stationaryIrrelevant[i,op],
                                                                                 name=f"Indic_effectiveFactor_src_({ops.dim2Dict[d]},{f},{i},{op_name})")
                                                                       for i in range(Num_Loops)), name=f"C_Indic_effectiveFactor_({ops.dim2Dict[d]},{f},{op_name})")

                        for m in range(1, acc.Num_mem):
                            if acc.mappingArray[op][m] == 0:
                                continue
                            indic_factor2Mem_effective[d,f,op,m] = var_AandB(model, indic_factor2Mem[d,f,op,m], indic_effectiveFactor[d,f,op],
                                                                                name=f"Indic_factor2Mem_effective_({ops.dim2Dict[d]},{f},{op_name},{acc.mem2dict(m)})")
        
        indic_factor2Mem_WriteIn = gp.tupledict()
        count_ReadOut, count_WriteIn = gp.tupledict(), gp.tupledict()
        for op, op_name in enumerate(['I','W','O']):
            for m in range(1, acc.Num_mem):
                if acc.mappingArray[op][m] == 0:
                    continue
                count_expr_readOut = 0
                count_expr_writeIn = 0

                for d in range(1, ops.Num_dim):
                    if factors[d] != [1]:
                        for f in range(len(factors[d])):
                            for m1 in range(1,m+1):
                                if m1 == m:
                                    count_expr_readOut += logF[d,f] * indic_factor2Mem_effective[d,f,op,m1]
                                else:
                                    count_expr_readOut += logF[d,f] * indic_factor2Mem[d,f,op,m1]
                            for m1 in range(1,m):
                                if acc.mappingArray[op][m1] == 0 or (m1, m, op) not in indic_nextMem:
                                    count_expr_writeIn += logF[d,f] * indic_factor2Mem[d,f,op,m1]
                                else:
                                    cname = f"Indic_factor2Mem_WriteIn_({ops.dim2Dict[d]},{f},{op_name},{acc.mem2dict(m1)},{acc.mem2dict(m)})"
                                    indic_factor2Mem_WriteIn[d,f,op,m1,m] = model.addVar(vtype=GRB.BINARY, name=cname)
                                    model.addConstr(indic_factor2Mem_WriteIn[d,f,op,m1,m] <= indic_factor2Mem[d,f,op,m1],
                                                    name=f"C_{cname}_base_ub")
                                    model.addConstr(indic_factor2Mem_WriteIn[d,f,op,m1,m] >= indic_factor2Mem_effective[d,f,op,m1],
                                                    name=f"C_{cname}_nb_lb")
                                    model.addConstr(indic_factor2Mem_WriteIn[d,f,op,m1,m] <= indic_factor2Mem_effective[d,f,op,m1] + 1 - indic_nextMem[m1,m,op],
                                                    name=f"C_{cname}_nb_ub")
                                    model.addConstr(indic_factor2Mem_WriteIn[d,f,op,m1,m] >= indic_factor2Mem[d,f,op,m1] - indic_nextMem[m1,m,op],
                                                    name=f"C_{cname}_base_lb")
                                    '''
                                    W = A AND ((NOT G) OR N)
                                    indic_factor2Mem_WriteIn = indic_factor2Mem AND ((NOT indic_nextMem) OR indic_effectiveFactor)
                                    '''
                                    count_expr_writeIn += logF[d,f] * indic_factor2Mem_WriteIn[d,f,op,m1,m]

                    for u in range(acc.Num_SpUr):
                        if m > acc.SpUr2Mem[u,op]:
                            count_expr_readOut += math.log(self.su[u][d])
                            count_expr_writeIn += math.log(self.su[u][d])
                count_ReadOut[m,op] = count_expr_readOut
                count_WriteIn[m,op] = count_expr_writeIn
       
        # ─── Precompute energy upper bounds for tighter PWL approximation ───
        _max_lg_factor_count = sum(logF[d, f]
                                    for d in range(1, ops.Num_dim)
                                    for f in range(len(factors[d]))
                                    if factors[d] != [1])
        UB_lg_energy, UB_exp_energy, UB_energy_perMem = {}, {}, {}
        for _m in range(1, acc.Num_mem):
            for _op, _opn in enumerate(['I', 'W', 'O']):
                if acc.mappingArray[_op][_m] == 0:
                    continue
                _spatial_lg = sum(math.log(self.su[u][d])
                                  for d in range(1, ops.Num_dim)
                                  for u in range(acc.Num_SpUr)
                                  if _m > acc.SpUr2Mem[u, _op])
                _ub_count = _max_lg_factor_count + _spatial_lg
                _lg_prec_ub = math.log(worst_prec(_m, _op))
                _can_read = _m not in [acc.IReg2mem, acc.OReg2mem, acc.Macro2mem] and acc.cost_r[_m] > 0
                _can_write = _m > 1
                _ub_sum = 0
                if _can_read:
                    _v = math.log(acc.cost_r[_m]) + _lg_prec_ub + _ub_count + UB_lg_transVolume[_m, _op]
                    UB_lg_energy[_m, _op, 'r2L'] = _v; UB_exp_energy[_m, _op, 'r2L'] = math.exp(_v); _ub_sum += math.exp(_v)
                if _can_write:
                    _v = math.log(max(acc.cost_w[_m], 1e-30)) + _lg_prec_ub + _ub_count + UB_lg_dataVolume[_m, _op]
                    UB_lg_energy[_m, _op, 'w2L'] = _v; UB_exp_energy[_m, _op, 'w2L'] = math.exp(_v); _ub_sum += math.exp(_v)
                if _opn == 'O' and _can_read and _m > acc.Dram2mem:
                    _v = math.log(acc.cost_r[_m]) + math.log(acc.precision_psum) + _ub_count + UB_lg_dataVolume[_m, _op]
                    UB_lg_energy[_m, _op, 'r2H'] = _v; UB_exp_energy[_m, _op, 'r2H'] = math.exp(_v); _ub_sum += math.exp(_v)
                if _opn == 'O':
                    _v = math.log(max(acc.cost_w[_m], 1e-30)) + math.log(acc.precision_psum) + _ub_count + UB_lg_transVolume[_m, _op]
                    UB_lg_energy[_m, _op, 'w2H'] = _v; UB_exp_energy[_m, _op, 'w2H'] = math.exp(_v); _ub_sum += math.exp(_v)
                UB_energy_perMem[_m, _op] = max(_ub_sum, 1.0)

        energy_perMem = gp.tupledict()
        lg_transEnergy_r2L, lg_transEnergy_w2L = gp.tupledict(), gp.tupledict()
        lg_transEnergy_r2H, lg_transEnergy_w2H = gp.tupledict(), gp.tupledict()
        for op, op_name in enumerate(['I','W','O']):
            for m in range(1, acc.Num_mem):
                if acc.mappingArray[op][m] == 0:
                    continue
                tmp_energy_expr = 0
                count_expr_readOut = count_ReadOut[m,op]
                count_expr_writeIn = count_WriteIn[m,op]
                can_read = m not in [acc.IReg2mem, acc.OReg2mem, acc.Macro2mem] and acc.cost_r[m] > 0
                can_write = m > 1

                # 1) Base read-out energy of this memory level.
                if can_read:
                    lg_transEnergy_r2L[m,op] = model.addVar(lb=LB_lg_transEnergy, ub=UB_lg_energy[m,op,'r2L'], vtype=GRB.CONTINUOUS, name=f"lg_transEnergy_r2L_({acc.mem2dict(m)},{op_name})")
                    prec_term_r2l = lg_prec_O(m) if op_name == 'O' else math.log(acc.precision[m,op])
                    model.addConstr(lg_transEnergy_r2L[m,op] == math.log(acc.cost_r[m]) + prec_term_r2l + count_expr_readOut + lg_transVolume[m,op],
                                    name=f"C_lg_transEnergy_r2L_({acc.mem2dict(m)},{op_name})")

                    tmp_energy_expr += var_exp(model=model, lg_term=lg_transEnergy_r2L[m,op], lb=0, ub=UB_exp_energy[m,op,'r2L'], name=f"transEnergy_r2L_({acc.mem2dict(m)},{op_name})")

                # 2) Base write-in energy of this memory level.
                if can_write:
                    lg_transEnergy_w2L[m,op] = model.addVar(lb=LB_lg_transEnergy, ub=UB_lg_energy[m,op,'w2L'], vtype=GRB.CONTINUOUS, name=f"lg_transEnergy_w2L_({acc.mem2dict(m)},{op_name})")
                    prec_term_w2l = lg_prec_O(m) if op_name == 'O' else math.log(acc.precision[m,op])
                    model.addConstr(lg_transEnergy_w2L[m,op] == math.log(acc.cost_w[m]) + prec_term_w2l + count_expr_writeIn + lg_dataVolume[m,op],
                                     name=f"C_lg_transEnergy_w2L_({acc.mem2dict(m)},{op_name})")

                    tmp_energy_expr += var_exp(model=model, lg_term=lg_transEnergy_w2L[m,op], lb=0, ub=UB_exp_energy[m,op,'w2L'], name=f"transEnergy_w2L_({acc.mem2dict(m)},{op_name})")

                # 3) Output readout for sending upstream memory hierarchy.
                # output往里面写多少次就要读多少次写回 ⬆
                if op_name == 'O' and can_read and m > acc.Dram2mem:
                    lg_transEnergy_r2H[m,op] = model.addVar(lb=LB_lg_transEnergy, ub=UB_lg_energy[m,op,'r2H'], vtype=GRB.CONTINUOUS, name=f"lg_transEnergy_r2H_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(lg_transEnergy_r2H[m,op] == math.log(acc.cost_r[m]) + lg_prec_O(m) + count_expr_writeIn + lg_dataVolume[m,op],
                                     name=f"C_lg_transEnergy_r2H_({acc.mem2dict(m)},{op_name})")

                    tmp_energy_expr += var_exp(model=model, lg_term=lg_transEnergy_r2H[m,op], lb=0, ub=UB_exp_energy[m,op,'r2H'], name=f"transEnergy_r2H_({acc.mem2dict(m)},{op_name})")

                # 4) Output write-back energy when this level sends output upward.
                if op_name == 'O':
                    lg_transEnergy_w2H[m,op] = model.addVar(lb=LB_lg_transEnergy, ub=UB_lg_energy[m,op,'w2H'], vtype=GRB.CONTINUOUS, name=f"lg_transEnergy_w2H_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(lg_transEnergy_w2H[m,op] == math.log(acc.cost_w[m]) + lg_prec_O(m) + count_expr_readOut + lg_transVolume[m,op],
                                     name=f"C_lg_transEnergy_w2H_({acc.mem2dict(m)},{op_name})")

                    tmp_energy_expr += var_exp(model=model, lg_term=lg_transEnergy_w2H[m,op], lb=0, ub=UB_exp_energy[m,op,'w2H'], name=f"transEnergy_w2H_({acc.mem2dict(m)},{op_name})")

                energy_perMem[m,op] = model.addVar(lb=0, ub=UB_energy_perMem[m,op], vtype=GRB.CONTINUOUS, name=f"energy_perMem_({acc.mem2dict(m)},{op_name})")
                model.addConstr(energy_perMem[m,op] == tmp_energy_expr, name=f"C_energy_perMem_({acc.mem2dict(m)},{op_name})")

        energy_expr_rw, energy_usedMem = 0, gp.tupledict()
        for m in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][m] == 0:
                    continue
                if m in [acc.Dram2mem, acc.IReg2mem, acc.OReg2mem]:
                    energy_expr_rw += energy_perMem[m,op]
                else:
                    energy_usedMem[m,op] = var_mul01(model, indic_usedMem[m,op], energy_perMem[m,op],
                                                      A_ub=UB_energy_perMem[m,op],
                                                      name=f"energy_usedMem_({acc.mem2dict(m)},{op_name})")
                    energy_expr_rw += energy_usedMem[m,op]

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#                 

        energy_expr_leakage = acc.leakage_per_cycle * CONST.SCALE_LATENCY * res_latency

        # ═══════════════════════════════════════════ Latency Recursive Model ════════════════════════════════════════════#
        #   L0  Trans(m,λ)     = max(tileVol×prec/bw, 1 cycle);  EffTrans = Trans + bypass×1cycle                  #
        #   L1  Transfer(i,λ)  = xMem(i,λ) × EffTrans(mem(i,λ), λ)                                                #
        #   L2  Body(i)   = max_λ { Process(i+1,λ) − dbl(i+1,λ)×Transfer(i+1,λ) }   [I/W dbl overlap]       #
        #   L3  Critical(i)    = max { Body, c(λ)×T, c(λ)×T+Body|single-buf }   c(O)=2, c(I/W)=1       #
        #   L4  Process(i,λ)   = c(λ)×Transfer + Process(i+1,λ) + (F(i)−1)×Critical(i);   Process(N)=t_MAC       #
        #   L5  MaxStartup     = max_λ { BootstrapRead(λ) + Σ_i Transfer(i,λ) }                                    #
        #   L6  Latency        ≥ MaxStartup − Σ_i Transfer(i,λ) + Process(0,λ) + BootstrapWrite(O)                 #
        # ══════════════════════════════════════════════════════════════════════════════════════════════════════════════#

        # ─── L0: Per-Memory 传输代价  Trans(m,λ) = max(rawTrans, 1 cycle) ───────────────────────────────────────────
        transLatency = gp.tupledict()                       # transLatency[m,op] — per-memory 传输延迟
        for op, op_name in enumerate(['I','W','O']):
            for m in range(1, acc.Num_mem):
                if acc.mappingArray[op][m]:
                    lg_transLatency = model.addVar(lb=LB_lg_transLatency[m,op], ub=UB_lg_transLatency[m,op], vtype=GRB.CONTINUOUS,
                                                    name=f"tmp_lg_transLatency_({acc.mem2dict(m)},{op_name})")
                    prec_term_lat = lg_prec_O(m) if op_name == 'O' else math.log(acc.precision[m,op])
                    model.addConstr(lg_transLatency == lg_transVolume[m,op] + prec_term_lat - math.log(acc.bw[m]) - math.log(CONST.SCALE_LATENCY),
                                     name=f"C_lg_transLatency_({acc.mem2dict(m)},{op_name})")

                    # ceil 语义：硬件不存在亚周期传输，非零搬移至少 1 cycle
                    rawTransLatency = model.addVar(lb=0, ub=UB_transLatency[m,op], vtype=GRB.CONTINUOUS,
                                                   name=f"tmp_rawTransLatency_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrExp(xvar=lg_transLatency, yvar=rawTransLatency, options=CONST.ExpOption,
                                          name=f"C_rawTransLatency_({acc.mem2dict(m)},{op_name})")
                    _lb_trans = LB_transLatency_const.get((m, op), LAT_UNIT)
                    transLatency[m,op] = model.addVar(lb=_lb_trans, ub=max(UB_transLatency[m,op], _lb_trans), vtype=GRB.CONTINUOUS,
                                                       name=f"transLatency_({acc.mem2dict(m)},{op_name})")
                    model.addGenConstrMax(transLatency[m,op], [rawTransLatency], constant=LAT_UNIT,
                                          name=f"C_transLatency_ceil_({acc.mem2dict(m)},{op_name})")

            # Reg bypass: CIM 位串行要求数据经 Reg 串转并。Reg 被 bypass 时每次传输 +1 cycle。
            # 嵌入 EffTrans 使所有映射到该 memory 的 level 自动继承。
            if op == 0: effectiveTransLatency = gp.tupledict()
            for m in range(1, acc.Num_mem):
                if acc.mappingArray[op][m]:
                    bp = model.addVar(vtype=GRB.BINARY, name=f"indic_regBypass_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(bp <= indic_loop2Mem[Num_Loops-1,op,m],     name=f"C_bp_a_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(bp <= 1 - indic_usedMem[acc.lastMem[op],op],name=f"C_bp_b_({acc.mem2dict(m)},{op_name})")
                    model.addConstr(bp >= indic_loop2Mem[Num_Loops-1,op,m] - indic_usedMem[acc.lastMem[op],op],
                                                                                name=f"C_bp_c_({acc.mem2dict(m)},{op_name})")
                    effectiveTransLatency[m,op] = transLatency[m,op] + bp * LAT_UNIT
                else:
                    effectiveTransLatency[m,op] = 0

        # ─── L1: Per-Memory Transfer — Transfer(i,λ) = Σ_m (loop2Mem AND xMem) × EffTrans(m) ───────────────────────
        # Per-memory big-M replaces global MAX_TRANS, eliminating 200-50000× looseness for inner memories.
        latency_Critical = gp.tupledict()
        latency_Process = gp.tupledict()
        latency_Transfer = gp.tupledict()
        latency_Body = gp.tupledict()
        for i in range(Num_Loops):
            latency_Critical[i] = model.addVar(lb=LB_Process[i+1], ub=UB_Critical[i], vtype=GRB.CONTINUOUS, name=f"latency_Critical_({i})")
            for op, op_name in enumerate(['I','W','O']):
                latency_Process[i,op] = model.addVar(lb=LB_Process[i], ub=UB_latencyLevel[i], vtype=GRB.CONTINUOUS, name=f"latency_Process_({i},{op_name})")

                # Per-memory McCormick: active = loop2Mem AND xMem; Transfer = Σ active × EffTrans
                _contrib = 0
                for m in range(1, acc.Num_mem):
                    if acc.mappingArray[op][m]:
                        _active = var_AandB(model, indic_loop2Mem[i,op,m], indic_hasTransfer[i,op],
                                             name=f"active_Transfer_({i},{op_name},{acc.mem2dict(m)})")
                        _ub_eff_m = max(UB_transLatency[m,op], LAT_UNIT) + LAT_UNIT
                        _contrib += var_mul01(model, _active, effectiveTransLatency[m,op],
                                               A_ub=_ub_eff_m, var_ub=_ub_eff_m,
                                               name=f"latency_Transfer_c_({i},{op_name},{acc.mem2dict(m)})")
                latency_Transfer[i,op] = model.addVar(lb=0, ub=self.MAX_TRANS[op]+LAT_UNIT,
                                                       vtype=GRB.CONTINUOUS, name=f"latency_Transfer_({i},{op_name})")
                model.addConstr(latency_Transfer[i,op] == _contrib,
                                name=f"C_latency_Transfer_({i},{op_name})")
                if i == Num_Loops-1:
                    latency_Process[Num_Loops,op] = t_MAC

        # ─── L2: Body(i) = max_λ { Process(i+1,λ) − dbl_overlap } ────────────────────────────────────────────
        for i in range(Num_Loops):
            _ub_body = UB_latencyLevel[i + 1] if i + 1 < Num_Loops else t_MAC
            latency_Body[i] = model.addVar(lb=LB_Process[i+1], ub=_ub_body, vtype=GRB.CONTINUOUS, name=f"latency_Body_({i})")
            for op, op_name in enumerate(['I','W','O']):
                if i < Num_Loops - 1 and op < 2:   # I/W 双缓冲时首迭代传输与父级流水重叠
                    indic_dblOverlap = var_mul01(model, indic_doubleLoop[i+1,op], latency_Transfer[i+1,op],
                                           A_ub=self.MAX_TRANS[op]+LAT_UNIT, var_ub=self.MAX_TRANS[op]+LAT_UNIT,
                                           name=f"indic_dblOverlap_({i},{op_name})")
                    model.addConstr(latency_Body[i] >= latency_Process[i+1,op] - indic_dblOverlap,
                                     name=f"C_Body_({i},{op_name})")
                else:
                    model.addConstr(latency_Body[i] >= latency_Process[i+1,op],
                                     name=f"C_Body_({i},{op_name})")

        # Cut: Σ Transfer ≥ transLatency for each used memory (strengthening)
        # 排除Macro Weight：CIM权重静态驻留，xMem被强制为0（line 509），Transfer路径不适用
        for op, op_name in enumerate(['I','W','O']):
            sum_latency_transfer = quicksum(latency_Transfer[i,op] for i in range(Num_Loops))
            for m in range(1, acc.Num_mem):
                if acc.mappingArray[op][m] == 0:
                    continue
                if op == 1 and m == acc.Macro2mem:
                    continue  # Macro Weight: 静态驻留，无per-iteration Transfer
                model.addConstr(sum_latency_transfer >= var_mul01(model, indic_usedMem[m,op], transLatency[m,op],
                                                                     name=f"tmp_sum_latencyTransfer_({acc.mem2dict(m)},{op_name})"),
                                    name=f"Cut_sum_latencyTransfer_({acc.mem2dict(m)},{op_name})")

        # ─── L3: Critical(i) — 单次稳态迭代瓶颈 ───────────────────────────────────────────────────────────────────
        for i in range(Num_Loops):
            model.addConstr(latency_Critical[i] >= latency_Body[i], name=f"C_latency_cp_child_({i})")
            for op, op_name in enumerate(['I','W','O']):
                coeff_rw = 2 if op == 2 else 1
                # 双缓冲：Transfer 与 Body 并行 → 瓶颈取 max
                model.addConstr(latency_Critical[i] >= coeff_rw * latency_Transfer[i,op], name=f"C_latency_cp_transfer_({i},{op_name})")
                # 单缓冲：Transfer 与 Body 串行 → 瓶颈取 sum (tight big-M: M = max possible c*Transfer)
                _M_tight = coeff_rw * (self.MAX_TRANS[op] + LAT_UNIT)
                model.addConstr(latency_Critical[i] >= coeff_rw * latency_Transfer[i,op] + latency_Body[i]
                                - _M_tight * indic_doubleLoop[i,op],
                                 name=f"C_latency_cp_transfer+child_({i},{op_name})")

        # ─── L4: Process(i,λ) = c(λ)×Transfer + child + (F(i)−1)×Critical ─────────────────────────────────────────
        # Z_crit[i,p] = indic_loop2Factor[i,p] × Critical[i]
        # Adaptive: McCormick (tight LP) at inner levels where UB_Critical is small,
        # GenConstrIndicator (exact) at outer levels where UB_Critical * IntFeasTol
        # causes near-integer indicators to produce large Z_crit errors.
        _intfeastol = model.Params.IntFeasTol
        Z_crit = {}
        for i in range(Num_Loops):
            _UB_C = UB_Critical[i]
            _max_zcrit_err = _UB_C * _intfeastol
            _use_indicator = (_max_zcrit_err > 0.01 * LB_Process[i + 1])
            for p in range(len(UNIQUE_FACTOR)):
                Z_crit[i, p] = model.addVar(lb=0, ub=_UB_C, vtype=GRB.CONTINUOUS,
                                             name=f"Z_crit_({i},{p})")
                # McCormick envelope — convex-hull LP relaxation
                model.addConstr(Z_crit[i, p] <= _UB_C * indic_loop2Factor[i, p],
                                name=f"C_McCormick_Z_ub1_({i},{p})")
                model.addConstr(Z_crit[i, p] >= latency_Critical[i] - _UB_C * (1 - indic_loop2Factor[i, p]),
                                name=f"C_McCormick_Z_lb1_({i},{p})")
                model.addConstr(Z_crit[i, p] <= latency_Critical[i],
                                name=f"C_McCormick_Z_ub2_({i},{p})")
                model.addConstr(Z_crit[i, p] >= LB_Process[i + 1] * indic_loop2Factor[i, p],
                                name=f"C_McCormick_Z_lb2_({i},{p})")
                if _use_indicator:
                    # GenConstrIndicator supplements McCormick at outer levels where
                    # UB_Critical * IntFeasTol causes near-integer exploitation of lb1.
                    # McCormick is retained for LP relaxation quality; indicators ensure
                    # integer-solution exactness via Gurobi's internal integrality rounding.
                    model.addGenConstrIndicator(indic_loop2Factor[i, p], True,
                        Z_crit[i, p] == latency_Critical[i],
                        name=f"C_Indicator_Z_true_({i},{p})")
                    model.addGenConstrIndicator(indic_loop2Factor[i, p], False,
                        Z_crit[i, p] == 0,
                        name=f"C_Indicator_Z_false_({i},{p})")

        for i in range(Num_Loops):
            for op, op_name in enumerate(['I','W','O']):
                coeff_rw = 2 if op == 2 else 1
                model.addConstr(
                    latency_Process[i, op] >= coeff_rw * latency_Transfer[i, op]
                        + latency_Process[i + 1, op]
                        + quicksum((UNIQUE_FACTOR[p] - 1) * Z_crit[i, p]
                                   for p in range(len(UNIQUE_FACTOR))),
                    name=f"C_latency_process_McCormick_({i},{op_name})")

                # Cut: valid hierarchical decay bound
                if op == 2:  # O: always valid since Body >= Process[i+1,O] (no dbl subtraction)
                    model.addConstr(latency_Process[i, op] >= 2 * latency_Process[i + 1, op],
                                    name=f"Cut_Hierarchical_Decay_({i},{op_name})")
                else:  # I/W: weaker but always valid (dbl_overlap may invalidate 2× bound)
                    model.addConstr(latency_Process[i, op] >= latency_Process[i + 1, op] + LB_Process[i + 1],
                                    name=f"Cut_Hierarchical_Decay_({i},{op_name})")

        # Cut: Transfer cascade bounds for all operands
        for op, op_name in enumerate(['I','W','O']):
            coeff_rw = 2 if op == 2 else 1
            model.addConstr(latency_Process[0, op] >= MIN_INNER_PROD[0] * t_MAC
                            + quicksum(coeff_rw * MIN_OUTER_PROD[i] * latency_Transfer[i, op]
                                       for i in range(Num_Loops)),
                            name=f"Cut_Transfer_Cascade_({op_name})")

        # ─── L5-L6: MaxStartup & res_latency ───────────────────────────────────────────────────────────────────────
        # Bootstrap: 最外层loop不在DRAM时，需要一次性加载全部数据。
        # Output精度绑定holdPsum[DRAM]：bootstrap生效 ⟹ 无Output loop在DRAM ⟹ holdPsum[DRAM]=0 ⟹ 精度=final。
        # 因此Output bootstrap直接使用precision_final（不使用worst_prec的psum上界）。
        def dram_prec(op):
            """DRAM级bootstrap精度：I/W用标准精度，O用final（bootstrap生效时DRAM无psum）"""
            if op != 2:
                return acc.precision[acc.Dram2mem, op]
            return acc.precision_final

        latency_BootstrapRead = {}
        latency_SumTransfer = {}
        for op, op_name in enumerate(['I','W','O']):
            latency_BootstrapRead[op] = MAX_SIZE[op] * dram_prec(op) / acc.bw[acc.Dram2mem] / CONST.SCALE_LATENCY
            latency_SumTransfer[op] = quicksum(latency_Transfer[i,op] for i in range(Num_Loops))

        latency_MaxStartup = model.addVar(lb=0, ub=UB_latencyLevel[0], vtype=GRB.CONTINUOUS, name="latency_MaxStartup")
        for op, op_name in enumerate(['I','W','O']):
            model.addConstr(latency_MaxStartup >= latency_BootstrapRead[op] * (1 - indic_loop2Mem[0,op,acc.Dram2mem])
                                                  + latency_SumTransfer[op],
                            name=f"C_MaxStartup_{op_name}")

        for op, op_name in enumerate(['I','W','O']):
            latency_BootstrapWrite = latency_BootstrapRead[op] * (1 - indic_loop2Mem[0,op,acc.Dram2mem]) if op == 2 else 0
            model.addConstr(res_latency >= latency_MaxStartup - latency_SumTransfer[op]
                                           + latency_Process[0,op] + latency_BootstrapWrite,
                             name=f"C_Res_Latency_CrossOp_({op_name})")

        # ─── Hardware-prior valid inequalities ────────────────────────────────────────────────────────────────────
        # Cut: DRAM bandwidth lower bound on res_latency (all data must pass through DRAM)
        # Output DRAM流量：write总是final精度，read在holdPsum[DRAM]=1时为psum精度
        for op, op_name in enumerate(['I','W','O']):
            coeff_rw = 2 if op == 2 else 1
            if op != 2:
                _dram_time = coeff_rw * MAX_SIZE[op] * acc.precision[acc.Dram2mem, op] / acc.bw[acc.Dram2mem] / CONST.SCALE_LATENCY
                model.addConstr(res_latency >= _dram_time, name=f"Cut_DRAM_BW_({op_name})")
            else:
                # Output: write + read 精度均由holdPsum[DRAM]决定（与simulator一致）
                _base_rw = 2 * MAX_SIZE[op] * acc.precision_final / acc.bw[acc.Dram2mem] / CONST.SCALE_LATENCY
                _psum_extra_rw = 2 * MAX_SIZE[op] * (acc.precision_psum - acc.precision_final) / acc.bw[acc.Dram2mem] / CONST.SCALE_LATENCY
                model.addConstr(res_latency >= _base_rw + _psum_extra_rw * indic_holdPartialSum[acc.Dram2mem],
                                name=f"Cut_DRAM_BW_({op_name})")

        # Cut: per-memory constant-floor transfer sum (spatial-unrolling minimum)
        for op, op_name in enumerate(['I','W','O']):
            _sum_trans = quicksum(latency_Transfer[i,op] for i in range(Num_Loops))
            for m in range(1, acc.Num_mem):
                if not acc.mappingArray[op][m]: continue
                if op == 1 and m == acc.Macro2mem: continue
                _lb_c = LB_transLatency_const.get((m, op), LAT_UNIT)
                if _lb_c > LAT_UNIT + 1e-9:
                    model.addConstr(_sum_trans >= _lb_c * indic_usedMem[m, op],
                                    name=f"Cut_TransFloor_const_({acc.mem2dict(m)},{op_name})")

        # Finite, provably-valid upper bound on res_energy (companion to the
        # res_latency bound above; both are needed for a finite EDP McCormick).
        # res_energy = energy_expr_rw + energy_expr_comp + energy_expr_leakage at
        # optimum; energy_expr_rw <= Sum of the per-memory energy var caps
        # (energy_usedMem <= energy_perMem <= UB_energy_perMem, indic<=1),
        # energy_expr_comp is constant, leakage <= leakage_per_cycle*SCALE*res_lat_ub.
        # Holds for every feasible mapping -> never cuts a mapping -> sound.
        _ub_energy_rw = sum(UB_energy_perMem[m, op]
                            for m in range(1, acc.Num_mem) for op in range(3)
                            if acc.mappingArray[op][m] and (m, op) in UB_energy_perMem)
        res_energy.ub = (_ub_energy_rw + energy_expr_comp
                         + acc.leakage_per_cycle * CONST.SCALE_LATENCY * res_latency.ub)
        model.addConstr(res_energy >= energy_expr_rw + energy_expr_comp + energy_expr_leakage, name="C_Res_Energy_Summation")
        if CONST.FLAG_OPT == "EDP":
            model.addConstr(res_EDP >= res_latency * res_energy * self.edp_scaling_factor, name="C_Res_EDP_Multiplication")

        # Tighten metric_ub from cross-worker shared state (if available).
        self._refresh_metric_ub()

        match CONST.FLAG_OPT:
            case "Latency":
                if self._has_finite_metric_ub():
                    model.addConstr(res_latency <= self.metric_ub / CONST.SCALE_LATENCY, name="C_metric_ub_latency")
                    self.profile.metric_upper_bound_applied = True
            case "Energy" | "EDP":
                pass
            case _:
                Logger.warning("Undefined Optimization Objective, No Upper Bound Applied.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#            
        Logger.info('* '*20 + "Start Running MIP Solver" + ' *'*20)
        model.ModelSense = GRB.MINIMIZE
        model.Params.TimeLimit = 20  # Feasibility prescreen
        model.Params.MIPFocus = 1
        model.update()
        self.profile.model_build_time_sec = time.time() - run_start_time
        self.profile.num_vars = model.NumVars
        self.profile.num_constrs = model.NumConstrs
        self.profile.num_sos = model.NumSOS
        self.profile.num_genconstrs = model.NumGenConstrs
        self.profile.num_bin_vars = sum(1 for var in model.getVars() if var.VType == GRB.BINARY)
        feasibility_start_time = time.time()
        model.optimize()
        self.profile.feasibility_time_sec = time.time() - feasibility_start_time
        if model.SolCount == 0:
            self.profile.status = int(model.Status)
            self.profile.sol_count = 0
            self.profile.total_time_sec = time.time() - run_start_time
            self.result = [CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS]
            return 1

        # Energy/EDP metric bounds come only from MIREDO's internal incumbent and
        # are applied after the feasibility prescreen so external baselines cannot
        # turn a feasible scheme into a prescreen failure.
        self._refresh_metric_ub()
        match CONST.FLAG_OPT:
            case "Energy":
                if self._has_finite_metric_ub():
                    model.addConstr(res_energy <= self.metric_ub, name="C_metric_ub_energy")
                    self.profile.metric_upper_bound_applied = True
            case "EDP":
                if self._has_finite_metric_ub():
                    # metric_ub is raw physical EDP; res_EDP objective units are
                    # raw * edp_scaling_factor / SCALE_LATENCY.
                    model.addConstr(res_EDP <= self.metric_ub * self.edp_scaling_factor / CONST.SCALE_LATENCY, name="C_metric_ub_EDP")
                    self.profile.metric_upper_bound_applied = True
                    # Cutoff-derived bound tightening — sound ONLY together with the
                    # cutoff just added, hence applied here (NOT during build, so the
                    # cutoff-free feasibility prescreen is not wrongly restricted). Any
                    # res_EDP<=metric_ub mapping has L*E<=metric_ub; the model enforces
                    # E>=energy_expr_comp and L(model)>=MIN_INNER_PROD[0]*t_MAC, so
                    # res_latency<=(metric_ub/SCALE)/E_floor, res_energy<=(metric_ub/SCALE)/L_floor
                    # drop no counterexample (certificate stays sound) while removing the
                    # huge-bound ill-conditioning of the bilinear res_EDP>=L*E.
                    _mub_model = self.metric_ub / CONST.SCALE_LATENCY
                    _Ef = energy_expr_comp if isinstance(energy_expr_comp, (int, float)) else None
                    if _Ef and _Ef > 0:
                        res_latency.ub = min(res_latency.ub, _mub_model / _Ef)
                    _Lf = MIN_INNER_PROD[0] * t_MAC
                    if _Lf > 0:
                        res_energy.ub = min(res_energy.ub, _mub_model / _Lf)
                    model.update()
            case _:
                pass

        ####################################################################  Set Constraint Flag ###################################################################
        model.discardMultiobjEnvs()

        model.setParam('TimeLimit', CONST.TIMELIMIT)
        model.setParam('MIPFocus', CONST.MIPFOCUS)

        match CONST.FLAG_OPT:
            case "Latency":
                model.setObjectiveN(res_latency, 0, priority=3, name='Latency')
                env0 = model.getMultiobjEnv(0)
                env0.setParam('TimeLimit', CONST.TIMELIMIT * 0.7)
                env0.setParam('ImproveStartTime', CONST.TIMELIMIT * 0.7 * 0.3)
                env0.setParam('MIPFocus', CONST.MIPFOCUS)
                model.setObjectiveN(res_energy, 1, priority=2, reltol=0.0, abstol=0.0, name='Energy')
                env1 = model.getMultiobjEnv(1)
                env1.setParam('TimeLimit', CONST.TIMELIMIT * 0.3)
                env1.setParam('MIPFocus', CONST.MIPFOCUS)
            case "Energy":
                model.setObjectiveN(res_energy, 0, priority=3, name='Energy')
                env0 = model.getMultiobjEnv(0)
                env0.setParam('TimeLimit', CONST.TIMELIMIT * 0.7)
                env0.setParam('ImproveStartTime', CONST.TIMELIMIT * 0.7 * 0.3)
                env0.setParam('MIPFocus', CONST.MIPFOCUS)
                model.setObjectiveN(res_latency, 1, priority=2, reltol=0.0, abstol=0.0, name='Latency')
                env1 = model.getMultiobjEnv(1)
                env1.setParam('TimeLimit', CONST.TIMELIMIT * 0.3)
                env1.setParam('MIPFocus', CONST.MIPFOCUS)
            case "EDP":
                model.setObjective(res_EDP, GRB.MINIMIZE)
                model.setParam('TimeLimit', CONST.TIMELIMIT)
                model.setParam('ImproveStartTime', CONST.TIMELIMIT * 0.3)
                model.setParam('MIPFocus', CONST.MIPFOCUS)
            case _:
                model.setObjective(0)
                model.setParam("TimeLimit", CONST.TIMELIMIT * 0.1)

        ####################################################################  Optimization    #######################################################################

        model.setParam('TimeLimit', CONST.TIMELIMIT)
        model.update()
        if FLAG.DEBUG:
            model.write(os.path.join(self.outputdir, "debug_model.lp"))

        # Build Gurobi callback for cross-worker bound propagation
        _cb = None
        if self.shared_ub is not None:
            match CONST.FLAG_OPT:
                case "Latency":
                    _ub_to_obj = 1.0 / CONST.SCALE_LATENCY
                case "EDP":
                    # shared_ub holds raw physical EDP; res_EDP objective units are
                    # raw * edp_scaling_factor / SCALE_LATENCY.
                    _ub_to_obj = self.edp_scaling_factor / CONST.SCALE_LATENCY
                case _:
                    _ub_to_obj = 1.0

            _shared = self.shared_ub
            _state = [0, 0, _shared.value * _ub_to_obj]  # [obj_idx, node_count, cached_ub]
            _REFRESH_NODES = 500  # refresh shared memory every N B&B nodes

            def _cb(model, where):
                if where == GRB.Callback.MULTIOBJ:
                    _state[0] = model.cbGet(GRB.Callback.MULTIOBJ_OBJCNT)
                elif where == GRB.Callback.MIPSOL and _state[0] == 0:
                    # Worker found a new incumbent — propagate to shared state
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    result_val = obj_val / _ub_to_obj
                    _shared.update_min(result_val)
                    _state[2] = _shared.value * _ub_to_obj
                    self.profile.callback_mipsol_updates += 1
                elif where == GRB.Callback.MIP and _state[0] == 0:
                    _state[1] += 1
                    if _state[1] >= _REFRESH_NODES:
                        _state[1] = 0
                        _state[2] = _shared.value * _ub_to_obj
                        obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
                        if obj_bound > _state[2]:
                            self.profile.callback_dynamic_terminations += 1
                            model.terminate()

        # ── EDP warm-start ────────────────────────────────────────────────
        # The EDP objective min res_EDP s.t. res_EDP >= res_latency*res_energy
        # is non-convex (bilinear); within the time budget the solver otherwise
        # stalls at a garbage heuristic incumbent (diagnosed: on a fixed spatial
        # scheme the EDP solve returned EDP ~1000x the achievable optimum that
        # the linear Latency/Energy solves find on the SAME scheme). Seed it from
        # cheap linear proxies: solve latency-min and energy-min, keep whichever
        # feasible mapping has the lower raw EDP, and inject it as the MIP start.
        # The EDP minimization then begins from a good incumbent and can only
        # improve, so the returned EDP <= min(latency-opt, energy-opt) EDP for the
        # scheme — eliminating the dominated "EDP-opt worse on both axes" result.
        # Guarded by FLAG.EDP_WARMSTART (default on) for A/B and rollback.
        if CONST.FLAG_OPT == "EDP" and getattr(FLAG, "EDP_WARMSTART", True):
            _ws_each = 0.15         # 2 proxy solves @15% each, EDP gets ~70%
            _best_start = None
            _best_raw_edp = float("inf")
            for _proxy in (res_latency, res_energy):
                model.setObjective(_proxy, GRB.MINIMIZE)
                model.setParam('TimeLimit', max(5.0, CONST.TIMELIMIT * _ws_each))
                model.setParam('MIPFocus', 1)
                model.update()
                model.optimize()
                if model.SolCount > 0:
                    _raw_edp = float(res_latency.X) * float(res_energy.X)
                    if _raw_edp < _best_raw_edp:
                        _best_raw_edp = _raw_edp
                        # Snapshot ONLY the discrete decision vars. Pinning the
                        # continuous res_* vars too makes Gurobi reject the MIP
                        # start as inconsistent, so the EDP solve was not actually
                        # bounded by the seed; integer-only starts are accepted and
                        # Gurobi derives the continuous vars from them.
                        _best_start = {_v: _v.X for _v in model.getVars()
                                       if _v.VType in (GRB.BINARY, GRB.INTEGER)}
            if _best_start is not None:
                for _v, _val in _best_start.items():
                    try:
                        _v.Start = _val
                    except Exception:
                        pass
                Logger.critical(f"EDP warm-start: seeded MIP start from best linear proxy "
                                f"(raw EDP={_best_raw_edp:.3e}, {len(_best_start)} int vars)")
            else:
                Logger.warning("EDP warm-start: no proxy incumbent found; proceeding cold")
            model.setObjective(res_EDP, GRB.MINIMIZE)
            model.setParam('TimeLimit', max(5.0, CONST.TIMELIMIT * (1 - 2 * _ws_each)))
            model.setParam('ImproveStartTime', CONST.TIMELIMIT * (1 - 2 * _ws_each) * 0.3)
            model.setParam('MIPFocus', CONST.MIPFOCUS)
            model.update()

        start_time = time.time()
        model.optimize(_cb)
        end_time = time.time()
        self.profile.optimize_time_sec = end_time - start_time
        self.profile.total_time_sec = end_time - run_start_time
        self.profile.status = int(model.Status)
        self.profile.sol_count = int(model.SolCount)
        self.profile.node_count = float(getattr(model, "NodeCount", 0.0))
        try:
            self.profile.mip_gap = float(model.MIPGap)
        except Exception:
            self.profile.mip_gap = None
        try:
            _ob = float(model.ObjBound)
            # ObjBound is in scaled model units; for EDP convert to raw physical
            # EDP so best_bound stays comparable across layers and carries no
            # solve-time conditioner (mirrors result[2]). Latency/Energy keep
            # their existing objective-unit convention.
            if CONST.FLAG_OPT == "EDP":
                _ob = _ob * CONST.SCALE_LATENCY / self.edp_scaling_factor
            self.profile.best_bound = _ob
        except Exception:
            self.profile.best_bound = None

        ####################################################################  Debug & Output  #######################################################################

        time_optimization = end_time - start_time
        Logger.critical(f"Optimizing time: {'%.3f' %(time_optimization)}s")

        def set_dataflow():
            Logger.info("# Set Dataflow Loops:")
            loops = LoopNest(acc=acc,ops=ops)
            # loops.tm
            for i in range(Num_Loops):
                for d in range(1, ops.Num_dim):
                    if factors[d] == [1]:     # DimSize == 1
                        continue
                    for f in range(len(factors[d])):
                        if (round(indic_factor2Loop[d,f,i].x) == 1):
                            loop2Mem = [0,0,0]
                            for m in range(1,acc.Num_mem):
                                for op in range(3):
                                    if acc.mappingArray[op][m] == 0:
                                        continue
                                    if round(indic_loop2Mem[i,op,m].x) == 1:
                                        loop2Mem[op] = m
                            loops.tm.append(Mapping(dim=d, 
                                                    dimSize=factors[d][f],
                                                    mem=[loop2Mem[op] for op in range(3)]))
            
            Logger.debug(self.su)
            for u in range(acc.Num_SpUr):
                for d in range(1, ops.Num_dim):
                    if self.su[u][d] > 1:
                        loops.sm.append(Mapping(dim=d, 
                                                dimSize=self.su[u][d],
                                                mem=[acc.SpUr2Mem[u,op] for op in range(3)]))

            double_tag = [[1 for _ in range(3)] for __ in range(acc.Num_mem+1)]
            for m in range(1,acc.Num_mem+1):
                for op, op_name in enumerate(['I','W','O']):
                    if m==acc.Num_mem:
                        # opMAC row: no buffering decision; the macro weight's
                        # flag is read from indic_doubleMem[Macro2mem, W] below.
                        double_tag[m][op] = 0
                        continue
                    if acc.double_config[m][op]:
                        val = indic_doubleMem[m,op]
                        double_tag[m][op] = round(val.x) if isinstance(val, gp.Var) else int(val)
                    else:
                        double_tag[m][op] = 0
            loops.usr_defined_double_flag = double_tag

            loops.psum_flag = {}
            for m in range(1, acc.Num_mem):
                state = indic_holdPartialSum[m]
                if isinstance(state, gp.Var):
                    loops.psum_flag[m] = bool(round(state.x))
                else:
                    loops.psum_flag[m] = bool(state)

            self.dataflow = loops
            
        if model.SolCount > 0:
            Logger.critical("MIP Solved successfully !!!")
            solved_latency = res_latency.x * CONST.SCALE_LATENCY
            solved_energy = res_energy.x
            # Report raw physical EDP (latency*energy). res_EDP carries the
            # solve-time conditioner edp_scaling_factor, which must not leak into
            # the reported metric, mirroring how solved_latency strips SCALE_LATENCY.
            solved_edp = solved_latency * solved_energy
            self.result = [solved_latency, solved_energy, solved_edp]
            set_dataflow()
            if FLAG.DEBUG:
                model.write(os.path.join(self.outputdir, "solution.sol"))
            match CONST.FLAG_OPT:
                case "Latency":
                    Logger.debug(f"Get best Latency= {solved_latency}")
                case "Energy":
                    Logger.debug(f"Get best Energy= {solved_energy}")
                case "EDP":
                    Logger.debug(f"Get best EDP= {solved_edp}")
                case _:
                    Logger.debug(f"Get simple solution, L={solved_latency}, E={solved_energy}")
        else:
            self.result = [CONST.MAX_POS, CONST.MAX_POS, CONST.MAX_POS]
            return 1

    def close(self):
        if self.model is not None:
            self.model.dispose()
            self.model = None
        if self._owns_env and self.env is not None:
            self.env.dispose()
            self.env = None
