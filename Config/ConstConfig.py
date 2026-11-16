import os

class _const_num_config():
    def __init__(self):
        self.PLACEHOLD = 5

        self.MAX_POS = 1e15 - 1

        self.FLAG_OPT = "Feasible"

        self.MIPFOCUS = 1

        self.WEIGHT_LATENCY = 0.8

        self.MAX_BLOCK_N = 35

        self.MAX_BLOCK_M = 15

        self.GAP_THRESHOLD = 0.2

        self.TIMELIMIT = 60




        self.SCALINGFACTOR = 1e-6

        self.SCALE_LATENCY = 1e3


        self.EPS = 1e-5

        self.ExpOption = "FuncPieces=-2 FuncPieceError=0.01"

class _constraint_flag_config():
    def __init__(self):
        self.DOUBLE_BUFFER = False

        self.LOADING_CONGESTION = True

        self.REMAIN_MULITCAST = True

        self.PARALLEL_INPUT_AND_WEIGHT = False

        self.PIPELINE_LOAD_AND_COMPUTATION = False

        self.PIPELINE_STORE_AND_MERGE = False

        self.FULLY_MAPPED = True

        self.MULIT_OP_EXIST_SIMULTANEOUSLY = False

        self.MACRO_UNDERUTILIZED = False

        self.OBUFFER_MERGE_WRITEBACK = False

        self.GBUFFER_MERGE_WRITEBACK = False

        self.BUFFER_FIFO = False

        self.STATIC_POWER = False


        self.BLOCK_N = False

        self.BLOCK_M = False

        self.OPS_EXCHANGE = False

        self.GAMMA = False

        self.INPUT_STATIONARY = False

        self.ROW_STATIONARY = False

        self.OUTPUT_STATIONARY = False

        self.ABLATION_FIXED_DOUBLE_BUFFER = False

        self.ABLATION_FORCE_DOUBLE_BUFFER = False

        self.ABLATION_SIMPLIFIED_PIPELINE = False

        self.ABLATION_PSUM_CAPACITY_ONLY = False

        self.ABLATION_DISABLE_LB_PRUNING = False

        self.ABLATION_DISABLE_FLEXFACT = os.environ.get("MIREDO_DISABLE_FLEXFACT", "").strip().lower() not in ("", "0", "false", "no")

        self.ACCEL_TOP_K = None

        self.PRESOLVE_SEARCH =    False

        self.LOAD_SOLUTION =    False

        self.DEBUG = False

        self.ASSERT_DEBUG = True

        self.GUROBI_OUTPUT = True

        self.DEBUG_PER_LAYER_DETAIL = False

        self.SIMU = True

        self.DEBUG_SIMU = False
