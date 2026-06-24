# this file is prepared for project 511
# Created by iboxl

import os

class _const_num_config():
    def __init__(self):
        self.PLACEHOLD = 5

        self.MAX_POS = 1e15 - 1                 #large RHS WARNING!!!
        
        self.FLAG_OPT = "Feasible"

        self.MIPFOCUS = 1                               # MIPFocus: 0=balanced, 1=feasibility, 2=optimality, 3=best bound

        self.WEIGHT_LATENCY = 0.8

        self.MAX_BLOCK_N = 35                   # 太大会导致某些操作数维度infeasible

        self.MAX_BLOCK_M = 15

        self.GAP_THRESHOLD = 0.2

        self.TIMELIMIT = 60

        # self.TIMELIMIT = 15

        # self.IMPROVESTART = 5

        # self.TIMELIMIT_AFTER_TLE = 10

        self.SCALINGFACTOR = 1e-6

        self.SCALE_LATENCY = 1e3    # latency 以千周期为单位
        
        # self.SCALE_BW = 1024            # bandwidth 以 KB/cycle 为单位

        self.EPS = 1e-5

        self.ExpOption = "FuncPieces=-2 FuncPieceError=0.01"

class _constraint_flag_config():
    def __init__(self):
        self.DOUBLE_BUFFER = False                          # 和 FLAG_PIPELINE_LOAD_AND_COMPUTATION 的联系、相关性

        self.LOADING_CONGESTION = True                      # True:输入间拥塞等待; False:不同操作数并行输入

        self.REMAIN_MULITCAST = True                        # 强制组播                  # bug 是否需要强制组播? 组播是线性空间的子集

        self.PARALLEL_INPUT_AND_WEIGHT = False              # 同时载入input和weight

        self.PIPELINE_LOAD_AND_COMPUTATION = False          # 使用双端SRAM或ping-pong buffer 需要考虑传输速率和计算速率差距

        self.PIPELINE_STORE_AND_MERGE = False               # output + accumulation & Psum merge

        self.FULLY_MAPPED = True                            # 包括多个层次：buffer(I)、macro compartment(W/C)、core_num(task)、
                                                            # W.T.D. 会大大增加模型复杂度

        self.MULIT_OP_EXIST_SIMULTANEOUSLY = False

        self.MACRO_UNDERUTILIZED = False                    # 约束范围小于 FLAG_FULLY_MAPPED
                                                            # 不充分占用不等同于多操作数同时，涉及控制等多因素 multi_num = size // utilize
                                                            # 细粒度的元操作数是否会导致搜索空间指数增长

        self.OBUFFER_MERGE_WRITEBACK = False                # True or False ?

        self.GBUFFER_MERGE_WRITEBACK = False

        self.BUFFER_FIFO = False                            # 数据消耗后不占用空间

        self.STATIC_POWER = False                            # 是否计算静态泄露功耗

        ####################################    优化策略标识            ###################################

        self.BLOCK_N = False                                # dim_N是否分块

        self.BLOCK_M = False                                # dim_M是否分块

        self.OPS_EXCHANGE = False                           # 交换操作数优化

        self.GAMMA = False                                  # 启用Gamma优化

        self.INPUT_STATIONARY = False                       # 当输入规模较小时避免反复调度输入

        self.ROW_STATIONARY = False
        
        self.OUTPUT_STATIONARY = False                      # 部分和累加在固定的位置完成-即一个output只存在于local OR global

        self.ABLATION_FIXED_DOUBLE_BUFFER = False       # 消融：禁用双缓冲决策（强制全部单缓冲）

        self.ABLATION_FORCE_DOUBLE_BUFFER = False       # 消融：强制双缓冲（在硬件/config允许处 beta=used，Reg等不支持处仍为单缓冲）

        self.ABLATION_SIMPLIFIED_PIPELINE = False       # 消融：移除流水线重叠建模（传输与计算永远串行）

        self.ABLATION_PSUM_CAPACITY_ONLY = False    # 消融：移除partial-sum容量耦合（保留精度翻转语义）

        self.ABLATION_DISABLE_LB_PRUNING = False    # 消融：禁用静态+动态分析下界剪枝（验证无损性）

        self.ABLATION_DISABLE_FLEXFACT = os.environ.get("MIREDO_DISABLE_FLEXFACT", "").strip().lower() not in ("", "0", "false", "no")      # 消融：禁用柔性分解（env MIREDO_DISABLE_FLEXFACT=1 开启，强制素因子分解；默认 False，环境变量未设时与原行为一致）

        self.ACCEL_TOP_K = None                     # EXP-7f knob: after static-LB screening, keep only the top-K
                                                    # candidates (sorted by utilization score). None/0 = disabled.

        self.PRESOLVE_SEARCH =    False                  # 预搜索MN寻找简单初始解

        self.LOAD_SOLUTION =    False                           # 读入solution.sol

        ####################################    程序 Debug 标识         ###########################
        self.DEBUG = False

        self.ASSERT_DEBUG = True

        self.GUROBI_OUTPUT = True

        self.DEBUG_PER_LAYER_DETAIL = False

        self.SIMU = True
        
        self.DEBUG_SIMU = False
