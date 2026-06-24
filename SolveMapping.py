# this file is prepared for project 511
# Created by iboxl

from utils.Workload import WorkLoad
from utils.Tools import append_scheme_summary, detect_parallel_config, auto_parallel_config, SharedUB
from utils.SolverTSS import Solver, SolverProfile
import gurobipy as gp
from Simulator.Simulax import tranSimulator
from utils.GlobalUT import *
from Architecture.ArchSpec import CIM_Acc
import pickle
from utils.UtilsFunction.ToolFunction import prepare_save_dir, get_Spatial_Unrolling
from utils.UtilsFunction.SchemeFunction import score_scheme, scheme_objective_lb, dominance_filter
import shutil
import time, math, os, struct
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field

@dataclass
class MappingRunProfile:
    objective: str = ""
    dominance_pruning_enabled: bool = False
    initial_metric_bound: float = 0.0
    num_schemes_initial: int = 0
    num_schemes_after_dominance: int = 0
    num_schemes_after_static_lb: int = 0
    num_schemes_after_top_k: int = 0
    num_schemes_dynamic_lb_pruned: int = 0
    num_schemes_after_dynamic_lb: int = 0
    num_schemes_submitted: int = 0
    num_schemes_with_solution: int = 0
    num_schemes_no_solution: int = 0
    best_scheme_count: int = 0
    best_scheme_origin_index: int = 0
    best_metric: float = 0.0
    timing_enumeration_sec: float = 0.0
    timing_scoring_pruning_sec: float = 0.0
    timing_mip_wall_sec: float = 0.0
    timing_mip_cumulative_sec: float = 0.0
    timing_total_sec: float = 0.0
    parallel_config: dict = field(default_factory=dict)
    best_solver_profile: SolverProfile | None = None

_worker_env = None  # cached Gurobi Env, set by _init_worker


def _finite_metric_bound(metric_bound):
    return metric_bound is not None and metric_bound < CONST.MAX_POS * 0.5

def _init_worker(threads_per_worker, runtime_config):
    """ProcessPoolExecutor initializer: create one Gurobi Env per worker process."""
    global _worker_env
    if runtime_config is not None:
        for key, value in runtime_config["CONST"].items():
            setattr(CONST, key, value)
        for key, value in runtime_config["FLAG"].items():
            setattr(FLAG, key, value)
    _worker_env = gp.Env(empty=True)
    _worker_env.setParam('OutputFlag', FLAG.GUROBI_OUTPUT)
    _worker_env.setParam('ThreadLimit', max(1, int(threads_per_worker)))
    _worker_env.start()


def solve_scheme_worker(count:int, origin_index:int, scheme, acc:CIM_Acc, ops:WorkLoad, metric_ub:float, outputdir_root:str, outputdir_scheme:str, threads_per_worker:int, soft_mem_limit_gb:float, runtime_config=None, shared_ub_name=None):
    if runtime_config is None:
        # Sequential mode: apply config directly (no initializer)
        pass
    # In parallel mode, config was already applied by _init_worker

    # Open cross-worker shared memory for metric upper bound
    shared_ub = None
    _shm_handle = None
    if shared_ub_name is not None:
        _shm_handle = SharedMemory(name=shared_ub_name)
        shared_ub = SharedUB(_shm_handle)

    prepare_save_dir(outputdir_scheme)

    spatial_unrolling = [math.prod(col) for col in zip(*scheme)]
    temporal_unrolling = [math.ceil(x / y) for x, y in zip(ops.dim2bound, spatial_unrolling)]

    # Provable lower-bound pruning: skip model building entirely if the
    # analytical bound already exceeds the current best objective.
    effective_ub = CONST.MAX_POS if metric_ub is None else metric_ub
    if shared_ub is not None:
        effective_ub = min(effective_ub, shared_ub.value)
    lat_lb, eng_lb, edp_lb = scheme_objective_lb(acc, ops, scheme, temporal_unrolling)
    obj_lb = {"Latency": lat_lb, "Energy": eng_lb, "EDP": edp_lb}.get(CONST.FLAG_OPT)
    if obj_lb is not None and obj_lb > effective_ub:
        if _shm_handle is not None:
            _shm_handle.close()
        return {
            "count": count, "origin_index": origin_index,
            "solver_result": [CONST.MAX_POS]*3, "sim_l": CONST.MAX_POS,
            "sim_e": CONST.MAX_POS, "profile": None, "dataflow": None,
            "has_solution": False, "outputdir_root": outputdir_root,
            "skip_reason": "dynamic_lb",
            "objective_lb": obj_lb,
            "solver_profile": None,
        }

    solver = Solver(
        acc=acc,
        ops=ops,
        tu=temporal_unrolling,
        su=scheme,
        metric_ub=metric_ub,
        outputdir=outputdir_scheme,
        threads=threads_per_worker,
        soft_mem_limit_gb=soft_mem_limit_gb,
        shared_ub=shared_ub,
        env=_worker_env,
    )

    sim_l, sim_e, profile = CONST.MAX_POS, CONST.MAX_POS, None
    try:
        solver.run()

        has_solution = solver.profile.sol_count > 0
        if has_solution:
            # Immediately propagate solution to shared state for cross-worker pruning
            if shared_ub is not None:
                metric_index = {"Latency": 0, "Energy": 1, "EDP": 2}.get(CONST.FLAG_OPT)
                if metric_index is not None:
                    shared_ub.update_min(solver.result[metric_index])

            if FLAG.SIMU:
                simu = tranSimulator(acc=acc, ops=ops, dataflow=solver.dataflow, DEBUG_SIMU=FLAG.DEBUG_SIMU)
                sim_l, sim_e = simu.run()
                profile = simu.PD
            else:
                sim_l, sim_e = solver.result[0], solver.result[1]
        elif os.path.exists(outputdir_scheme):
            shutil.rmtree(outputdir_scheme)

        return {
            "count": count,
            "origin_index": origin_index,
            "solver_result": solver.result,
            "sim_l": sim_l,
            "sim_e": sim_e,
            "profile": profile,
            "dataflow": solver.dataflow if has_solution else None,
            "has_solution": has_solution,
            "outputdir_root": outputdir_root,
            "skip_reason": None if has_solution else "no_solution",
            "objective_lb": obj_lb,
            "solver_profile": solver.profile,
        }
    finally:
        solver.close()
        if _shm_handle is not None:
            _shm_handle.close()


def update_best(result_pack:dict, best_metric:float, result, best_count:int, best_dataflow, solCount:int,
                best_solver_profile=None, best_origin_index:int=0):
    count = result_pack["count"]
    origin_index = result_pack["origin_index"]
    solver_result = result_pack["solver_result"]
    sim_l = result_pack["sim_l"]
    sim_e = result_pack["sim_e"]
    profile = result_pack["profile"]
    has_solution = result_pack["has_solution"]
    skip_reason = result_pack.get("skip_reason")
    solver_profile = result_pack.get("solver_profile")

    if has_solution:
        solCount += 1
        result_msg = (
            f"Scheme {count:<3} End: Latency-{round(sim_l,3):<15}, Energy-{round(sim_e,3):<15}, "
            f"EDP-{round(sim_l * sim_e,3):<15}"
        )
        lat_msg = (
            f"      |--- Latency Relative Error: {round(abs(solver_result[0]-sim_l)/sim_l*100,2)}%, "
            f"Solver-{round(solver_result[0]):<10} and Simu-{round(sim_l,3):<10}"
        )
        eng_msg = (
            f"      |--- Energy  Relative Error: {round(abs(solver_result[1]-sim_e)/sim_e*100,2)}%, "
            f"Solver-{round(solver_result[1]):<10} and Simu-{round(sim_e,3):<10}"
        )

        append_scheme_summary(result_pack["outputdir_root"], result_msg)
        append_scheme_summary(result_pack["outputdir_root"], lat_msg)
        append_scheme_summary(result_pack["outputdir_root"], eng_msg)

        Logger.info(result_msg)
        Logger.info(lat_msg)
        Logger.info(eng_msg)
    elif skip_reason == "dynamic_lb":
        result_msg = f"Scheme {count:<3} End: PRUNED_BY_DYNAMIC_LB"
        append_scheme_summary(result_pack["outputdir_root"], result_msg)
        Logger.info(result_msg)
    else:
        result_msg = f"Scheme {count:<3} End: NO BETTER SOLUTION"
        append_scheme_summary(result_pack["outputdir_root"], result_msg)
        Logger.info(result_msg)

    assert CONST.FLAG_OPT in ["Latency", "Energy", "EDP"], "No Such Metric for Optimization, Please Check CONST.FLAG_OPT"
    metric_index = {"Latency": 0, "Energy": 1, "EDP": 2}[CONST.FLAG_OPT]
    # Record MIREDO's own best independently from any external baseline.  The
    # pruning bound `best_metric` is only tightened by MIREDO incumbents.
    if has_solution and solver_result[metric_index] < result[metric_index]:
        result = solver_result + [sim_l, sim_e] + [profile]
        best_count = count
        best_dataflow = result_pack["dataflow"]
        best_solver_profile = solver_profile
        best_origin_index = origin_index
    if has_solution and solver_result[metric_index] < best_metric:
        best_metric = solver_result[metric_index]

    return best_metric, result, best_count, best_dataflow, solCount, best_solver_profile, best_origin_index

def SolveMapping(acc:CIM_Acc, ops:WorkLoad, bestMetric:int, outputdir:str, singleIter=False, **kwargs):
    time_begin = time.time()
    return_profile = kwargs.get("return_profile", False)
    initial_metric_bound = CONST.MAX_POS
    run_profile = MappingRunProfile(
        objective=CONST.FLAG_OPT,
        dominance_pruning_enabled=kwargs.get("dominance_pruning", False),
        initial_metric_bound=initial_metric_bound,
        best_metric=initial_metric_bound,
    )

    if FLAG.INPUT_STATIONARY and (acc.core.size_input_buffer * acc.num_core >= ops.dim_M * ops.dim_K * ops.input.bitwidth):
        Logger.debug("Sufficient Buffer Resources for Input")

    count, solCount = 0, 0
    best_metric = initial_metric_bound
    best_count = 0
    best_origin_index = 0
    best_dataflow = None
    best_solver_profile = None
    result = [CONST.MAX_POS] * 5 + [None]
    enumeration_begin = time.time()

    # ── Step 1: enumerate & prune spatial schemes ──────────────────────────
    if singleIter:
        scheme = kwargs["Spatial_unrolling"]
        assert scheme is not None, "Single Iteration Mode Requires Spatial Unrolling Input as Scheme."
        scheme_records = [{
            "origin_index": 1,
            "scheme": scheme,
            "meta": score_scheme(acc=acc, ops=ops, scheme=scheme),
        }]
        run_profile.num_schemes_initial = 1
        run_profile.num_schemes_after_dominance = 1
        run_profile.num_schemes_after_static_lb = 1
    else:
        scheme_records = []
        for origin_index, scheme in enumerate(
            get_Spatial_Unrolling(ops.dim2bound, acc.mappingRule, acc.SpUnrolling),
            start=1,
        ):
            meta = score_scheme(acc=acc, ops=ops, scheme=scheme)
            scheme_records.append({
                "origin_index": origin_index,
                "scheme": scheme,
                "meta": meta,
            })

        total_candidates = len(scheme_records)
        run_profile.num_schemes_initial = total_candidates
        run_profile.timing_enumeration_sec = time.time() - enumeration_begin
        pruning_begin = time.time()

        # ── Dominance pruning (optional, off by default) ─────────────────
        # NOT provably safe: a dominator with prime temporal factors may have
        # fewer loop levels than the dominated scheme, limiting memory hierarchy
        # tiling flexibility.  ~44% of dominance pairs exhibit this factor-count
        # asymmetry.  Experimentally validated (never loses optimum on tested
        # workloads), but cannot guarantee zero optimality loss in general.
        # Enable via SolveMapping(..., dominance_pruning=True) for ablation.
        if kwargs.get("dominance_pruning", False):
            dominance_safe = True
            for m in range(1, acc.Num_mem):
                if acc.mem2dict(m) in ('OReg', 'IReg', 'Macro'):
                    continue
                if acc.shareMemory[m]:
                    combined_volume = 0
                    for op in range(3):
                        if not acc.mappingArray[op][m]: continue
                        max_spur = math.prod(acc.SpUnrolling[u] for u in range(acc.Num_SpUr) if m <= acc.SpUr2Mem[u, op])
                        prec = acc.precision_psum if op == 2 else acc.precision[m, op]
                        combined_volume += max_spur * prec
                    if combined_volume > acc.memSize[m]:
                        dominance_safe = False; break
                else:
                    for op in range(3):
                        if not acc.mappingArray[op][m]: continue
                        max_spur = math.prod(acc.SpUnrolling[u] for u in range(acc.Num_SpUr) if m <= acc.SpUr2Mem[u, op])
                        prec = acc.precision_psum if op == 2 else acc.precision[m, op]
                        if max_spur * prec > acc.memSize[m]:
                            dominance_safe = False; break
                    if not dominance_safe: break

            if dominance_safe:
                scheme_records = dominance_filter(scheme_records)
                if len(scheme_records) < total_candidates:
                    Logger.info(f"Dominance pruning: {total_candidates} -> {len(scheme_records)} schemes.")
            else:
                Logger.info(f"Dominance pruning skipped: spatial tile exceeds buffer capacity.")
        run_profile.num_schemes_after_dominance = len(scheme_records)

        # ── Sort by utilization score ──────────────────────────────────────
        # High-utilization first → finds good bounds early → tightens metric_ub
        # → accelerates downstream analytical-LB pruning.
        scheme_records.sort(key=lambda r: (r["meta"]["sort_key"], -r["origin_index"]), reverse=True)

        # ── Analytical lower-bound screening (Propositions 1-3) ────────────
        # Discard schemes whose provable LB exceeds the initial internal metric UB.
        # The dynamic version in solve_scheme_worker re-checks against the
        # live shared_ub during parallel execution.
        if _finite_metric_bound(best_metric):
            before = len(scheme_records)
            survived = []
            for rec in scheme_records:
                lat_lb, eng_lb, edp_lb = scheme_objective_lb(acc, ops, rec["scheme"], rec["meta"]["temporal_unrolling"])
                obj_lb = {"Latency": lat_lb, "Energy": eng_lb, "EDP": edp_lb}.get(CONST.FLAG_OPT)
                if obj_lb is None or obj_lb <= best_metric:
                    survived.append(rec)
            if len(survived) < before:
                Logger.info(f"Analytical LB screening: {before} -> {len(survived)} schemes (LB <= internal incumbent).")
                scheme_records = survived
        run_profile.num_schemes_after_static_lb = len(scheme_records)
        run_profile.timing_scoring_pruning_sec = time.time() - pruning_begin

    # ── Optional top-K truncation (EXP-7f knob) ───────────────────────────
    # Applied AFTER analytical-LB screening (static LB stage) and BEFORE
    # scout/sweep dispatch.  Activates only when FLAG.ACCEL_TOP_K is a
    # positive integer; None/0/unset leaves the pipeline byte-identical.
    # Relies on the utilization-LB sort already done at line 309 — the
    # highest-utility candidates are at the front of scheme_records.
    _top_k = getattr(FLAG, "ACCEL_TOP_K", None)
    if _top_k and isinstance(_top_k, int) and _top_k > 0 and len(scheme_records) > _top_k:
        Logger.info(f"Top-K truncation: {len(scheme_records)} -> {_top_k} schemes (ACCEL_TOP_K={_top_k}).")
        scheme_records = scheme_records[:_top_k]
    run_profile.num_schemes_after_top_k = len(scheme_records)

    if singleIter:
        run_profile.timing_enumeration_sec = 0.0
        run_profile.timing_scoring_pruning_sec = 0.0

    num_schemes = len(scheme_records)

    for count, scheme_record in enumerate(scheme_records, start=1):
        scheme_record["count"] = count
        scheme_record["outputdir_scheme"] = outputdir if singleIter else os.path.join(outputdir, "SolPool", str(count))

    # ── Step 2: two-phase adaptive parallel configuration ─────────────────
    parallel_config = detect_parallel_config()
    usable = parallel_config["usable_cores"]
    avail_mem = parallel_config["available_mem_gb"]
    runtime_config = {"CONST": dict(vars(CONST)), "FLAG": dict(vars(FLAG))}

    # Two-phase adaptive parallel config: scout (thread-tilted, high-utility
    # schemes) then sweep (worker-tilted, remaining schemes). See
    # utils.Tools.auto_parallel_config docstring for the design rationale.
    parallel_cfg = auto_parallel_config(usable, avail_mem, num_schemes)
    scout_threads, scout_workers = parallel_cfg["scout"]
    sweep_threads, sweep_workers = parallel_cfg["sweep"]
    scout_size = parallel_cfg["scout_size"]

    use_parallel = (not singleIter) and usable >= 2
    if not use_parallel:
        scout_threads = usable
        scout_workers = 1
        sweep_threads = usable
        sweep_workers = 1
        scout_size = num_schemes

    soft_mem_scout = max(1.0, avail_mem * 0.8 / scout_workers)
    soft_mem_sweep = max(1.0, avail_mem * 0.8 / sweep_workers)

    Logger.critical(
        f"Auto Parallel Config: physical={parallel_config['physical_cores']}, logical={parallel_config['logical_cores']}, "
        f"usable={usable}, scout={scout_workers}w×{scout_threads}t(top-{scout_size}), "
        f"sweep={sweep_workers}w×{sweep_threads}t(rest-{max(0,num_schemes-scout_size)}), "
        f"schemes={num_schemes}"
    )
    run_profile.parallel_config = {
        "physical_cores": parallel_config["physical_cores"],
        "logical_cores": parallel_config["logical_cores"],
        "usable_cores": usable,
        "scout_workers": scout_workers,
        "scout_threads": scout_threads,
        "sweep_workers": sweep_workers,
        "sweep_threads": sweep_threads,
        "scout_size": scout_size,
        "num_schemes": num_schemes,
    }

    scheme_iter = iter(scheme_records)
    mip_stage_begin = time.time()

    if not use_parallel:
        for scheme_record in scheme_iter:
            count = scheme_record["count"]
            meta = scheme_record["meta"]
            append_scheme_summary(
                outputdir,
                f"Scheme {count:<3} : "
                f"util_prod={meta['util_product']:.4f}, min_util={meta['util_min']:.3f}, "
                f"avg_util={meta['util_avg']:.3f}] "
                f"Beginning: SpUr-{meta['spatial_unrolling']}, TpUr-{meta['temporal_unrolling']}"
            )
            worker_args = (
                count,
                scheme_record["origin_index"],
                scheme_record["scheme"],
                acc,
                ops,
                best_metric,
                outputdir,
                scheme_record["outputdir_scheme"],
                usable,
                soft_mem_scout,
            )
            run_profile.num_schemes_submitted += 1
            result_pack = solve_scheme_worker(*worker_args)
            if result_pack.get("skip_reason") == "dynamic_lb":
                run_profile.num_schemes_dynamic_lb_pruned += 1
            elif result_pack["has_solution"]:
                run_profile.num_schemes_with_solution += 1
            else:
                run_profile.num_schemes_no_solution += 1
            solver_profile = result_pack.get("solver_profile")
            if solver_profile is not None:
                run_profile.timing_mip_cumulative_sec += solver_profile.total_time_sec
            best_metric, result, best_count, best_dataflow, solCount, best_solver_profile, best_origin_index = update_best(
                result_pack, best_metric, result, best_count, best_dataflow, solCount, best_solver_profile, best_origin_index
            )
    else:
        _shm = SharedMemory(create=True, size=8)
        struct.pack_into('d', _shm.buf, 0, best_metric)
        _shm_name = _shm.name

        def _run_phase(executor, records, threads, soft_mem, max_workers):
            nonlocal count, best_metric, result, best_count, best_dataflow, solCount, best_solver_profile, best_origin_index
            pending = {}
            rec_iter = iter(records)
            inflight_limit = max_workers * 2  # bound pending futures to limit memory

            def submit(rec):
                nonlocal count, best_metric
                scheme_count = rec["count"]
                meta = rec["meta"]
                append_scheme_summary(
                    outputdir,
                    f"Scheme {scheme_count:<3} : "
                    f"util_prod={meta['util_product']:.4f}, min_util={meta['util_min']:.3f}, "
                    f"avg_util={meta['util_avg']:.3f}] "
                    f"Beginning: SpUr-{meta['spatial_unrolling']}, TpUr-{meta['temporal_unrolling']}"
                )
                args = (
                    scheme_count, rec["origin_index"], rec["scheme"],
                    acc, ops, best_metric, outputdir, rec["outputdir_scheme"],
                    threads, soft_mem, runtime_config, _shm_name,
                )
                run_profile.num_schemes_submitted += 1
                future = executor.submit(solve_scheme_worker, *args)
                pending[future] = scheme_count
                count = scheme_count

            # Fill pipeline
            for rec in rec_iter:
                submit(rec)
                if len(pending) >= inflight_limit:
                    break

            # Drain + refill
            while pending:
                done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    result_pack = future.result()
                    if result_pack.get("skip_reason") == "dynamic_lb":
                        run_profile.num_schemes_dynamic_lb_pruned += 1
                    elif result_pack["has_solution"]:
                        run_profile.num_schemes_with_solution += 1
                    else:
                        run_profile.num_schemes_no_solution += 1
                    solver_profile = result_pack.get("solver_profile")
                    if solver_profile is not None:
                        run_profile.timing_mip_cumulative_sec += solver_profile.total_time_sec
                    best_metric, result, best_count, best_dataflow, solCount, best_solver_profile, best_origin_index = update_best(
                        result_pack, best_metric, result, best_count, best_dataflow, solCount, best_solver_profile, best_origin_index
                    )
                    SharedUB(_shm).update_min(best_metric)
                for rec in rec_iter:
                    submit(rec)
                    if len(pending) >= inflight_limit:
                        break

        mp_context = mp.get_context("spawn")
        scout_records = scheme_records[:scout_size]
        sweep_records = scheme_records[scout_size:]

        try:
            # Phase 1: Scout — more threads, top-ranked schemes
            with ProcessPoolExecutor(max_workers=scout_workers, mp_context=mp_context,
                                     initializer=_init_worker, initargs=(scout_threads, runtime_config)) as executor:
                _run_phase(executor, scout_records, scout_threads, soft_mem_scout, scout_workers)

            # Phase 2: Sweep — more workers, remaining schemes with tight shared_ub
            if sweep_records:
                with ProcessPoolExecutor(max_workers=sweep_workers, mp_context=mp_context,
                                         initializer=_init_worker, initargs=(sweep_threads, runtime_config)) as executor:
                    _run_phase(executor, sweep_records, sweep_threads, soft_mem_sweep, sweep_workers)
        finally:
            _shm.close()
            _shm.unlink()
    run_profile.timing_mip_wall_sec = time.time() - mip_stage_begin
    
    if count == 0:
        raise ValueError("No feasible spatial scheme found after pre-screening")
    if solCount == 0:
        raise ValueError("SOLVER IIS")

    time_end = time.time()
    if not singleIter:
        Logger.info(f"Total valid loop nest found: {count}, best_count: {best_count}")
        Logger.info(f"Solving Time within whole layer: {round(time_end - time_begin,1)}s")

    file_name = os.path.join(outputdir, "Dataflow.pkl")
    with open(file_name, 'wb') as file:
        pickle.dump(best_dataflow, file)

    run_profile.num_schemes_after_dynamic_lb = max(
        0,
        run_profile.num_schemes_after_static_lb - run_profile.num_schemes_dynamic_lb_pruned,
    )
    run_profile.best_scheme_count = best_count
    run_profile.best_scheme_origin_index = best_origin_index
    run_profile.best_metric = best_metric
    run_profile.best_solver_profile = best_solver_profile
    run_profile.timing_total_sec = time_end - time_begin

    if return_profile:
        return result, run_profile
    return result

if __name__ == "__main__":
    
    import uuid
    import time
    outFolder = os.path.join("output",f"#SolveMappingTest_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid1().hex[:8]}")
    prepare_save_dir(outFolder)

    start_time = time.time()

    Logger.setcfg(setcritical=False, setDebug=True, STD=False, file=os.path.join(outFolder,'112.log'), nofile=False)

    from Architecture.templates.default import default_spec
    accelerator = CIM_Acc.from_spec(default_spec())

    Logger.debug("Running SolveMapping for debugging and testing Solver (MIP model), only one iteration with given scheme")

    CONST.FLAG_OPT="Latency"
    # CONST.FLAG_OPT="Energy"
    # CONST.FLAG_OPT="EDP"
    
    CONST.MIPFOCUS = 1
    # CONST.MIPFOCUS = 2
    # CONST.MIPFOCUS = 3
    
    CONST.TIMELIMIT = 77
    # CONST.TIMELIMIT = 600

    # # # # # RestNet-layer-0
    # ops = WorkLoad(loopDim={'R': 7, 'S': 7, 'C': 3, 'K':64, 'P': 112, 'Q': 112, 'G': 1, 'B': 1, 'H': 224, 'W': 224, 'Stride': 2, 'Padding': 3})
    # Spatial_unrolling = [[1,1,1,2,1,1,4,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,1,7,1,1,3,1,1],
    #                      [1,1,1,1,1,1,16,1]]

    # # # # # RestNet-layer-1
    ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 64, 'K':64, 'P': 56, 'Q': 56, 'G': 1, 'B': 1, 'H': 56, 'W': 56, 'Stride': 1, 'Padding': 1})
    Spatial_unrolling =    [[1,1,1,2,1,1,4,1],
                        #   [-,R,S,P,Q,C,K,G],
                            [1,1,1,1,1,32,1,1],
                            [1,1,1,1,1,1,16,1]]

    # # # # # RestNet-layer-12
    # ops = WorkLoad(loopDim={'R': 1, 'S': 1, 'C': 128, 'K':256, 'P': 14, 'Q': 14, 'G': 1, 'B': 1, 'H': 28, 'W': 28, 'Stride': 2, 'Padding': 0})
    # Spatial_unrolling = [[1,1,1,2,2,1,2,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,1,1,1,1,32,1,1],
    #                      [1,1,1,1,1,1,16,1]]

    # # # # # RestNet-layer-15
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 256, 'K':512, 'P': 7, 'Q': 7, 'G': 1, 'B': 1, 'H': 14, 'W': 14, 'Stride': 2, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,1,1,8,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,1,1,1,1,32,1,1],
    #                      [1,1,1,1,1,1,16,1]]

    # # # # # RestNet-layer-16
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 512, 'K':512, 'P': 7, 'Q': 7, 'G': 1, 'B': 1, 'H': 7, 'W': 7, 'Stride': 1, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,1,1,8,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,1,1,1,1,32,1,1],
    #                      [1,1,1,1,1,1,16,1]]

    # # # # # RestNet-layer-17
    # ops = WorkLoad(loopDim={'R': 1, 'S': 1, 'C': 256, 'K':512, 'P': 7, 'Q': 7, 'G': 1, 'B': 1, 'H': 14, 'W': 14, 'Stride': 2, 'Padding': 0})
    # Spatial_unrolling = [[1,1,1,1,1,1,8,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,1,1,1,1,32,1,1],
    #                      [1,1,1,1,1,1,16,1]]

    # # # # # MobileNetV2-depthwise Conv_1  (G=32,  P=Q=112, stride=1, SpUr from full-model Scheme4)
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 1, 'K': 1, 'P': 112, 'Q': 112, 'G': 32, 'B': 1, 'H': 112, 'W': 112, 'Stride': 1, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,8,1,1,1],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,3,3,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1]]

    # # # # # MobileNetV2-depthwise Conv_10 (G=144, P=Q=28,  stride=2, SpUr from full-model Scheme2)
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 1, 'K': 1, 'P': 28, 'Q': 28, 'G': 144, 'B': 1, 'H': 56, 'W': 56, 'Stride': 2, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,2,1,1,4],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,3,3,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1]]

    # # # # # MobileNetV2-depthwise Conv_40 (G=576, P=Q=7,   stride=2, SpUr from full-model Scheme1)
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 1, 'K': 1, 'P': 7, 'Q': 7, 'G': 576, 'B': 1, 'H': 14, 'W': 14, 'Stride': 2, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,1,1,1,8],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,3,3,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1]]

    # # # # # MobileNetV2-depthwise Conv_43 (G=960, P=Q=7,   stride=1, SpUr from full-model Scheme1)
    # ops = WorkLoad(loopDim={'R': 3, 'S': 3, 'C': 1, 'K': 1, 'P': 7, 'Q': 7, 'G': 960, 'B': 1, 'H': 7, 'W': 7, 'Stride': 1, 'Padding': 1})
    # Spatial_unrolling = [[1,1,1,1,1,1,1,8],
    #                  #   [-,R,S,P,Q,C,K,G],
    #                      [1,3,3,1,1,1,1,1],
    #                      [1,1,1,1,1,1,1,1]]

    lat, eng, edp, c_lat, c_eng, ds = SolveMapping(acc=accelerator, ops=ops, bestMetric=CONST.MAX_POS, outputdir=outFolder, singleIter=True, Spatial_unrolling=Spatial_unrolling)

    end_time = time.time()
    Logger.critical(f"SingleIter-Running SolveMapping Cost: {round(end_time - start_time,1)}s")
