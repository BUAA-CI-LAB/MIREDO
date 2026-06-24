from utils.Tools import *
from utils.Workload import WorkLoad
from SolveMapping import SolveMapping
import argparse
from utils.GlobalUT import *
import uuid
from utils.UtilsFunction.OnnxParser import extract_loopdims
from utils.UtilsFunction.ToolFunction import prepare_save_dir
from Architecture.ArchSpec import CIM_Acc
from Architecture.templates.default import default_spec
import time, copy


def normalize_loopdim_for_solver(loopdim):
    normalized = copy.deepcopy(loopdim)
    for dim_char in ["P", "Q", "H", "W"]:
        if normalized[dim_char] % 2 == 1 and normalized[dim_char] > 15:
            normalized[dim_char] += 1
    return normalized


def make_accelerator():
    spec = default_spec()
    return CIM_Acc.from_spec(spec)


def get_Args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", nargs="?", const=True, default=False, help="Enable debug mode.")
    parser.add_argument("--logger", nargs="?", const=True, default=False, help="Just print Logger")
    parser.add_argument("--srun", nargs="?", const=True, default=False, help="batch srun with critical message")
    parser.add_argument("--noLogFile", nargs="?", const=True, default=False, help="No log file")
    parser.add_argument("--IS", nargs="?", const=True, default=False, help="FLAG: Buffer/Input stationary")
    parser.add_argument("--NoPreSolve", nargs="?", const=True, default=False, help="dont search presolve by alpha&beta")
    parser.add_argument("--SIMU", nargs="?", const=True, default=False, help="using simulator calc")

    parser.add_argument('-m', '--model', dest='model', required=False,
                        type=str, default='resnet18', help='NN model Name')
    parser.add_argument('-log', '--log_file', dest='log', required=False,
                        type=str, default='miredo.log', help='Log file Name')
    parser.add_argument('-opt', '--flag_opt', dest='opt', choices=["Latency", "Energy", "EDP"], required=False,
                        type=str, default="Feasible", help='Optimization: Feasible, Latency, Energy, EDP')
    parser.add_argument('-f', '--mipFocus', dest='mipFocus', choices=[0, 1, 2, 3], required=False,
                        type=int, default=1, help='0=balanced, 1=feasibility, 2=optimality, 3=best bound')
    parser.add_argument('-class', '--num_classes', dest='classes', choices=[10, 1000], required=False,
                        type=int, default=1000, help='10=CIFAR 1000=ImageNet')
    parser.add_argument('-t', '--time', dest='time_limit', required=False,
                        type=int, default=CONST.TIMELIMIT, help='time limitation for solving gurobi model')
    parser.add_argument('-o', '--outputdir', dest='output_dir', required=False,
                        type=str, default=f'test_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid1().hex[:8]}',
                        help='save output files in folder')
    args = parser.parse_args()

    return args


def __main__(**kwargs):

    args = get_Args()
    start_time = time.time()
    outFolder = os.path.join("output", args.output_dir)
    prepare_save_dir(outFolder)

    Logger.setcfg(setcritical=args.srun, setDebug=args.debug, STD=args.logger,
                  file=os.path.join(outFolder, args.log), nofile=args.noLogFile)

    CONST.FLAG_OPT              = args.opt
    CONST.TIMELIMIT             = args.time_limit
    CONST.MIPFOCUS              = args.mipFocus
    FLAG.INPUT_STATIONARY       = args.IS
    FLAG.DEBUG_SIMU             = args.SIMU
    FLAG.PRESOLVE_SEARCH        = not args.NoPreSolve
    FLAG.DEBUG_PER_LAYER_DETAIL = False

    Logger.info("* " * 50)
    Logger.info(f"model={args.model}, Optimization_Flag={CONST.FLAG_OPT}, MIPFOCUS={CONST.MIPFOCUS}")
    Logger.info("* " * 50)

    model = f"model/{args.model}.onnx"
    convs, loopdims = extract_loopdims(model)
    assert len(convs) == len(loopdims)

    accelerator_template = make_accelerator()
    total_latency, total_energy = 0, 0

    for i, (Conv, loopdim) in enumerate(zip(convs, loopdims)):

        Logger.info('\n\n' + '* ' * 20 + f"Layer {i}" + ' *' * 20)
        ops = WorkLoad(loopDim=loopdim)
        Logger.info(ops)

        newdim = normalize_loopdim_for_solver(loopdim)
        outputdir_layer = os.path.join(outFolder, Conv)
        prepare_save_dir(outputdir_layer)
        Logger.changeFile(new_file=os.path.join(outputdir_layer, "Layer.log"), mode="w")
        Logger.info(ops)
        Logger.info('\n' + '* ' * 30 + '\n')

        accelerator = copy.deepcopy(accelerator_template)

        l_solver, e_solver, edp_solver, l_simu, e_simu, profile = SolveMapping(
            acc=accelerator,
            ops=WorkLoad(loopDim=newdim),
            bestMetric=CONST.MAX_POS,
            outputdir=outputdir_layer,
        )

        pstr = '\n'
        pstr += f"* * * MIREDO Result * * *  Latency:{round(l_simu, 3):<15}, Energy:{round(e_simu, 3):<20}, EDP:{round(l_simu * e_simu, 3):.5e}" + '\n'
        pstr += f"MIP Solver Latency Relative Error: {round(abs(l_solver - l_simu) / l_simu * 100, 2)}%    (Simu){round(l_simu, 3):<15} (Solver){round(l_solver, 3):<15}" + '\n'
        pstr += f"MIP Solver Energy  Relative Error: {round(abs(e_solver - e_simu) / e_simu * 100, 2)}%    (Simu){round(e_simu, 3):<15} (Solver){round(e_solver, 3):<15}" + '\n'

        Logger.info(pstr)
        Logger.changeFile(new_file=os.path.join(outFolder, args.log))
        Logger.info(pstr)

        total_latency += l_simu
        total_energy += e_simu

    Logger.info("* " * 50)
    Logger.info('\n\n' + '* ' * 20 + f"The WHOLE Model" + ' *' * 20)
    Logger.info(f"* * * MIREDO Result * * *  Latency:{round(total_latency, 3):<15}, Energy:{round(total_energy, 3):<15}, EDP:{round(total_latency * total_energy, 3):.5e}")

    end_time = time.time()
    Logger.critical(f"Solving The Whole Model Cost: {round(end_time - start_time, 1)}s")
    Logger.recover_stdout()


if __name__ == "__main__":
    __main__()
