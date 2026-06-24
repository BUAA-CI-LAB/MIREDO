#  this file is prepared for project 419
#  Created by iboxl
#  Modified in project 112

from utils.Workload import WorkLoad, LoopNest
from Architecture.ArchSpec import CIM_Acc
import math
from utils.GlobalUT import *
from dataclasses import dataclass, field

@dataclass
class TransCost():
    r:float
    w:float
    t:int       

@dataclass
class ProfilingDetail():
    dynamic_power:float = 0
    leakage_power:float = 0
    dynamic_power_onChip:float = 0
    memCost:list[float] = field(default_factory=list)
    macEnergy:float = 0
    macLatency:int = 0
    latency:int = 0
    Energy_OnChip:float = 0
    Energy_OffChip:float = 0
    doubleFlag:list[list[bool]] = field(default_factory=list)
    transfer_cycles:list[float] = field(default_factory=lambda: [0, 0, 0])
    mode_switch_cycles:list[float] = field(default_factory=lambda: [0, 0, 0])
    mismatch_cycles:list[float] = field(default_factory=lambda: [0, 0, 0])
    output_writeback_cycles:float = 0
    bootstrap_cycles:float = 0
    mode_switch_stall:float = 0
    mismatch_stall:float = 0
    writeback_stall:float = 0
    idle_cycles:float = 0
    count_mac: int = 0
    peak_mac_per_cycle: int = 0
    bytes_read: list = field(default_factory=list)    # per memory level, in bytes
    bytes_written: list = field(default_factory=list) # per memory level, in bytes

class tranSimulator():
    def __init__(self, acc:CIM_Acc, ops:WorkLoad, dataflow:LoopNest, DEBUG_SIMU=False):
        Logger.info('* - '*8 + "Deploying Dataflow Onto Trans-simulator" + ' - *'*8)
        self.acc = acc
        self.ops = ops
        self.dataflow = dataflow
        self.DEBUG_SIMU = DEBUG_SIMU

        self.dataflow.preprogress()
        # print(self.dataflow)
        Logger.info(self.dataflow)

        # dim_tp[mem][t][dim]   
        dim_tp = [ [ [ 1 for dim in range(ops.Num_dim)] for t in range(3) ] for mem in range(acc.Num_mem)]
        # dim_sp[mem][t][dim]
        dim_sp = [ [ [ 1 for dim in range(ops.Num_dim)] for t in range(3) ] for mem in range(acc.Num_mem)]
        for mem in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                for mapping in dataflow.tm:
                    if mem <= mapping.mem[op] and acc.mappingArray[op][mem]==1:
                        dim_tp[mem][op][mapping.dim] *= mapping.dimSize
                for mapping in dataflow.sm:
                    if mem <= mapping.mem[op] and acc.mappingArray[op][mem]==1:
                        dim_sp[mem][op][mapping.dim] *= mapping.dimSize
        self.dim_tp, self.dim_sp = dim_tp, dim_sp
        
        dataSize = {}
        tmp_dim = [_ for _ in range(ops.Num_dim)]
        for op, op_name in enumerate(['I','W','O']):
            dataSize[acc.Num_mem, op] = 1
        for mem in range(1,acc.Num_mem):
            op = 0  # Input
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                dataSize[mem,op] = ops.get_operand_size(tmp_dim, op)
            else:
                dataSize[mem,op] = 0

            op = 1  # Weight
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                dataSize[mem,op] = ops.get_operand_size(tmp_dim, op)
            else:
                dataSize[mem,op] = 0

            op = 2  # Output
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                dataSize[mem,op] = ops.get_operand_size(tmp_dim, op)
            else:
                dataSize[mem,op] = 0
        for op, op_name in enumerate(['I','W','O']):
            for mem in range(1,acc.Num_mem):
                if self.dataflow.bypassMem[mem][op]:
                    dataSize[mem,op] = 0
        
        Logger.info(f"Utilization:") 
        for mem in range(1,acc.Num_mem):
            sum_mem = 0
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][mem]:
                    sum_mem += (dataSize[mem,op] + dataSize[mem,op] * dataflow.usr_defined_double_flag[mem][op]) * self._prec(mem, op)
            if sum_mem > acc.memSize[mem]:
                Logger.error(f"OverSize Error:\n{acc.mem2dict(mem)}:{acc.memSize[mem]}(Size) less than {sum_mem}(Used):"
                                + f"I-{dataSize[mem,0]*self._prec(mem,0)}({dataflow.usr_defined_double_flag[mem][0]}) "
                                + f"W-{dataSize[mem,1]*self._prec(mem,1)}({dataflow.usr_defined_double_flag[mem][1]}) "
                                + f"O-{dataSize[mem,2]*self._prec(mem,2)}({dataflow.usr_defined_double_flag[mem][2]})")
                raise ValueError("Dataflow Over MemSize Error")
            else:
                Logger.info('-'*4 + f" {acc.mem2dict(mem)}({round(sum_mem/acc.memSize[mem]*100,2):>3}%): {sum_mem}/{acc.memSize[mem]} - {acc.bw[mem]} bit/cc")
        self.dataSize = dataSize

        # dim_tp[mem][t][dim]   
        trans_dim_tp = [ [ [ 1 for dim in range(ops.Num_dim)] for t in range(3) ] for mem in range(acc.Num_mem)]
        # dim_sp[mem][t][dim]
        trans_dim_sp = [ [ [ 1 for dim in range(ops.Num_dim)] for t in range(3) ] for mem in range(acc.Num_mem)]
        for mem in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                for mapping in dataflow.tm:
                    if mem < mapping.mem[op] and acc.mappingArray[op][mem]==1:        #  <= -> <
                        trans_dim_tp[mem][op][mapping.dim] *= mapping.dimSize
                for mapping in dataflow.sm:
                    if mem <= mapping.mem[op] and acc.mappingArray[op][mem]==1:
                        trans_dim_sp[mem][op][mapping.dim] *= mapping.dimSize
        self.trans_dim_tp, self.trans_dim_sp = trans_dim_tp, trans_dim_sp
        
        tileSize = {}
        tmp_dim = [_ for _ in range(ops.Num_dim)]
        for op, op_name in enumerate(['I','W','O']):
            tileSize[acc.Num_mem, op] = 1
        for mem in range(1,acc.Num_mem):
            op = 0  # Input
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = trans_dim_sp[mem][op][dim] * trans_dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                tileSize[mem,op] = ops.get_operand_size(tmp_dim, op)
            else:
                tileSize[mem,op] = 0

            op = 1  # Weight
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = trans_dim_sp[mem][op][dim] * trans_dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                tileSize[mem,op] = ops.get_operand_size(tmp_dim, op)
            else:
                tileSize[mem,op] = 0

            op = 2  # Output
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = trans_dim_sp[mem][op][dim] * trans_dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                tileSize[mem,op] = ops.get_operand_size(tmp_dim, op)
            else:
                tileSize[mem,op] = 0
        for op, op_name in enumerate(['I','W','O']):
            for mem in range(1,acc.Num_mem):
                if self.dataflow.bypassMem[mem][op]:
                    tileSize[mem,op] = 0
        self.tileSize = tileSize

        self.instance_Used_count = {}
        for op, op_name in enumerate(['I','W','O']):
            for m in range(1, acc.Num_mem):
                cnt = 1
                for mapping in dataflow.sm:
                    if m > mapping.mem[op] and acc.mappingArray[op][m] == 1:
                        cnt *= mapping.dimSize
                self.instance_Used_count[m, op] = cnt
        for op, op_name in enumerate(['I','W','O']):
            self.instance_Used_count[acc.Num_mem, op] = self.instance_Used_count[acc.lastMem[1], 1]
        
        self.count_mac = 0

        self.timer = {}
        for mem in range(acc.Num_mem+1):
            for op, op_name in enumerate(['I','W','O']):
                self.timer[mem,op] = 0
        
        self.memCost = {}
        for m in range(1,acc.Num_mem+1):
            self.memCost[m] = TransCost(r=0, w=0, t=0)
        
        self.PD = ProfilingDetail()
        self.PD.memCost.append(0)

        self.PD.doubleFlag.append([])
        for mem in range(1,acc.Num_mem):
            tmp_list = []
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][mem] == 0:
                    tmp_list.append('-') 
                else:
                    tmp_list.append(round(self.dataflow.usr_defined_double_flag[mem][op]))
            self.PD.doubleFlag.append(tmp_list)

    def _prec(self, mem, op):
        if hasattr(self.dataflow, "get_precision"):
            return self.dataflow.get_precision(mem, op)
        return self.acc.precision[mem, op]


    def ptimer(self, idx, n):
        if self.DEBUG_SIMU == False:
            return
        pstr = f"Loop {idx} iter {n}:\n"
        timer = self.timer
        for mem in range(1, self.acc.Num_mem+1):
            pstr += f"{self.acc.mem2dict(mem):<13}:"
            for op, op_name in enumerate(['I','W','O']):
                pstr += f"{op_name}({timer[mem,op]:^8})  "
            pstr += '\n'
        print(pstr)
            

    def loopExecution(self, loopidx:int):

        uppmem = self.dataflow.uppermem
        nxtmem = self.dataflow.nxtmem
        xMem   = self.dataflow.xMem

        if loopidx == len(self.dataflow.tm):
            mac_mem = self.acc.Num_mem
            # data_input_rdy = max(self.timer[mac_mem,0], self.timer[mac_mem,1])
            data_input_rdy = max(self.timer[mac_mem,op] for op in range(3))
            data_output_rdy = self.timer[mac_mem,2]
            self.ptimer(loopidx, 0)

            for op, op_name in enumerate(['I','W','O']):
                self.timer[mac_mem, op] = max(data_input_rdy + self.acc.t_MAC, data_output_rdy)

            for op, op_name in enumerate(['I','W','O']):                                # bit-serial  Macro/IOReg should be defined by user
                # if op_name == 'O':                          # Psum dont using bit-serial
                #     self.timer[mac_mem, op] += 1            # Accumulation in the merge unit 
                if self.dataflow.usr_defined_double_flag[uppmem[mac_mem,op]][op] == 0:
                    self.timer[uppmem[mac_mem,op], op] = self.timer[mac_mem, op]
            self.count_mac += 1
            
            self.ptimer(loopidx, 1)
            
        else:
            mem_cur = [self.dataflow.tm[loopidx].mem[op] for op in range(3)]
            mem_nxt = [nxtmem[mem_cur[op],op] for op in range(3)]

            nxtSize = [self.dataSize[mem_nxt[op],op] if xMem[loopidx,op]==1 else 0 for op in range(3)]
            tileSize = [self.tileSize[mem_cur[op],op] if xMem[loopidx,op]==1 else 0 for op in range(3)]

            trans = [math.ceil(tileSize[op]*self._prec(mem_cur[op], op)/self.acc.bw[mem_cur[op]]) for op in range(3)]
            '''
            如果将来 r_bw ≠ w_bw 或内层 bw < 外层 bw 就会出错 但当前不需要修 MIP 模型用的同一个
            '''
            for op, op_name in enumerate(['I','W','O']):     
                if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                    trans[op] += 1                      # Not only 1, consider others
            op, op_name = 1,'W'     # Weight
            if mem_cur[op] == self.acc.Macro2mem:
                trans[op] = 0
            Cons = [(trans[op] if xMem[loopidx,op]==1 else 0) for op in range(3)]

            dflag = [self.dataflow.usr_defined_double_flag[mem_nxt[op]][op] for op in range(3)]

            self.ptimer(loopidx, 0)

            N = self.dataflow.tm[loopidx].dimSize
            for i in range(N):

                for op, op_name in enumerate(['I','W','O']):
                    if dflag[op] == 1:      # double-buffer: start immediately
                        self.timer[mem_cur[op],op] = self.timer[mem_cur[op],op] + Cons[op]
                        stall = max(0,max(self.timer[mem_nxt[op],op], self.timer[mem_cur[op],op]) - self.timer[mem_nxt[op],op])
                        self.timer[mem_nxt[op],op] = max(self.timer[mem_nxt[op],op], self.timer[mem_cur[op],op])
                        self.PD.transfer_cycles[op] += Cons[op]
                        self.PD.mismatch_cycles[op] += stall
                    else:                   # single-buffer: wait for nxt
                        self.timer[mem_cur[op],op] = max(self.timer[mem_cur[op],op], self.timer[mem_nxt[op],op]) + Cons[op]
                        self.timer[mem_nxt[op],op] = self.timer[mem_cur[op],op]
                        stall = Cons[op]
                        self.PD.transfer_cycles[op] += Cons[op]
                        self.PD.mode_switch_cycles[op] += Cons[op]
                    self.memCost[mem_cur[op]].t += stall

                    if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                        self.memCost[self.lastMappingMem[op]].r += self.acc.cost_r[self.lastMappingMem[op]] * tileSize[op] * self._prec(self.lastMappingMem[op], op)
                        self.memCost[self.acc.lastMem[op]].w += self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op]  * self._prec(self.acc.lastMem[op], op)

                        self.memCost[self.acc.lastMem[op]].r += self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op]  * self._prec(self.acc.lastMem[op], op)
                        self.memCost[mem_nxt[op]].w += self.acc.cost_w[mem_nxt[op]] * nxtSize[op]  * self._prec(mem_nxt[op], op)
                    else:
                        self.memCost[mem_cur[op]].r += self.acc.cost_r[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)
                        self.memCost[mem_nxt[op]].w += self.acc.cost_w[mem_nxt[op]] * nxtSize[op]  * self._prec(mem_nxt[op], op)

                # Double-buffer O write-back of previous iteration (delayed).
                # Placed after pre-read so PR(k) and WB(k-1) are serial at mem_cur.
                # max(cur, nxt) enforces data dependency on previous child.
                # nxt NOT updated: WB at mem_cur port does not block child at mem_nxt.
                op = 2  # O
                if i > 0 and Cons[op] > 0 and dflag[op] == 1:
                    self.timer[mem_cur[op],op] = max(self.timer[mem_cur[op],op], self.timer[mem_nxt[op],op]) + Cons[op]
                    self.PD.transfer_cycles[op] += Cons[op]
                    self.PD.output_writeback_cycles += Cons[op]

                    if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                        self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op]  * self._prec(mem_nxt[op], op)
                        self.memCost[self.acc.lastMem[op]].w += self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op] * self._prec(self.acc.lastMem[op], op)

                        self.memCost[self.acc.lastMem[op]].r += self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op]  * self._prec(self.acc.lastMem[op], op)
                        self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)
                    else:
                        self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op]  * self._prec(mem_nxt[op], op)
                        self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)

                    self.memCost[mem_cur[op]].t += Cons[op]

                self.loopExecution(loopidx=loopidx+1)

                # Single-buffer O write-back: after child (no pipeline benefit)
                op = 2  # O
                if dflag[op] != 1:
                    self.timer[mem_nxt[op],op] = max(self.timer[mem_cur[op],op], self.timer[mem_nxt[op],op]) + Cons[op]
                    self.timer[mem_cur[op],op] = self.timer[mem_nxt[op],op]
                    self.PD.transfer_cycles[op] += Cons[op]
                    self.PD.output_writeback_cycles += Cons[op]

                    if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                        self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op]  * self._prec(mem_nxt[op], op)
                        self.memCost[self.acc.lastMem[op]].w += self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op] * self._prec(self.acc.lastMem[op], op)

                        self.memCost[self.acc.lastMem[op]].r += self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op]  * self._prec(self.acc.lastMem[op], op)
                        self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)
                    else:
                        self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op]  * self._prec(mem_nxt[op], op)
                        self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)

                    self.memCost[mem_cur[op]].t += Cons[op]

                self.ptimer(loopidx, i+1)

            # Flush: last double-buffer O write-back
            op = 2  # O
            if N > 0 and Cons[op] > 0 and dflag[op] == 1:
                self.timer[mem_cur[op],op] = max(self.timer[mem_cur[op],op], self.timer[mem_nxt[op],op]) + Cons[op]
                self.PD.transfer_cycles[op] += Cons[op]
                self.PD.output_writeback_cycles += Cons[op]

                if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                    self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op]  * self._prec(mem_nxt[op], op)
                    self.memCost[self.acc.lastMem[op]].w += self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op] * self._prec(self.acc.lastMem[op], op)

                    self.memCost[self.acc.lastMem[op]].r += self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op]  * self._prec(self.acc.lastMem[op], op)
                    self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)
                else:
                    self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op]  * self._prec(mem_nxt[op], op)
                    self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)

                self.memCost[mem_cur[op]].t += Cons[op]

    # ------------------------------------------------------------------
    # Analytical evaluation path.
    #
    # 动机：CoSA / CIMLoop 产生的 mapping 经常把大的 trip count 放在内层，
    # 每层 `for i in range(N)` 展开后 leaf 调用数会放大成全部 MAC 事件数（VGG 单层
    # 可到 1e9 量级），原 tile-walk `loopExecution` 会跑数小时。
    #
    # 策略：对每个非叶层 loopExecution(loopidx) 的 N 次循环做如下等价展开
    #   iter0 显式执行（跳过 `if i > 0` 分支）
    #   iter1 显式执行（带 `if i > 0` 分支）
    #   iter2 显式执行；记录 (状态before→after) 作稳态 delta
    #   对 (N-3) 次追加迭代按稳态 delta 线性累加
    #   flush（若 dflag[O]==1）
    # 其中 iter0/iter1/iter2 都递归调用 `loopExecutionAnalytical(loopidx+1)`，
    # 因此 child 每层被调用 3 次。总调用复杂度  O(3^len(tm))，对典型 baseline
    # （len(tm) ≤ ~15）在秒级。
    #
    # 为什么需要 3 个显式 iter（而非 2 个）：
    #   iter0 消化初始 transient；iter1 消化 child analytical 内部 max-absorb 的
    #   次级 transient（child 对两次不同 entry 的 inner iter0 可能产生不同 delta，
    #   它在 parent 眼中表现为 iter1 delta ≠ iter2 delta）。iter2 及以后 entry
    #   pattern 稳定，child delta 固定，可做线性外推。
    #
    # bit-exact 前提（与 run() 逐字段一致）：
    #   (a) pre-read 累加项（transfer_cycles / mode_switch / mismatch / memCost）
    #       每迭代贡献恒定，直接线性 × (N-3)。
    #   (b) `if i > 0` 分支（lines 297-312 in run()）从 iter1 起每轮执行一次；
    #       它已包含在 iter1/iter2 body 内。
    #   (c) flush（lines 339-355，仅 dflag[O]==1）在 N 次循环之后执行一次。
    #   (d) timer 的 `max(t, t_nxt)` 递推从 iter2 起达到稳态 delta。
    # 若某种特殊 mapping 破坏 (d)，会在 Verify_simulax_equivalence 上暴露为非 bit-exact
    # — 当作 bug 修复而不是近似。能耗/memCost 因浮点累加顺序会有 ≤1e-12 相对误差，属
    # 可接受的 FP 噪声而非语义偏差。
    # ------------------------------------------------------------------

    def loopExecutionAnalytical(self, loopidx:int):
        uppmem = self.dataflow.uppermem
        nxtmem = self.dataflow.nxtmem
        xMem   = self.dataflow.xMem

        if loopidx == len(self.dataflow.tm):
            mac_mem = self.acc.Num_mem
            data_input_rdy = max(self.timer[mac_mem, op] for op in range(3))
            data_output_rdy = self.timer[mac_mem, 2]
            for op in range(3):
                self.timer[mac_mem, op] = max(data_input_rdy + self.acc.t_MAC, data_output_rdy)
            for op in range(3):
                if self.dataflow.usr_defined_double_flag[uppmem[mac_mem, op]][op] == 0:
                    self.timer[uppmem[mac_mem, op], op] = self.timer[mac_mem, op]
            self.count_mac += 1
            return

        mem_cur = [self.dataflow.tm[loopidx].mem[op] for op in range(3)]
        mem_nxt = [nxtmem[mem_cur[op], op] for op in range(3)]

        nxtSize = [self.dataSize[mem_nxt[op], op] if xMem[loopidx, op] == 1 else 0 for op in range(3)]
        tileSize = [self.tileSize[mem_cur[op], op] if xMem[loopidx, op] == 1 else 0 for op in range(3)]

        trans = [math.ceil(tileSize[op] * self._prec(mem_cur[op], op) / self.acc.bw[mem_cur[op]]) for op in range(3)]
        for op in range(3):
            if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                trans[op] += 1
        if mem_cur[1] == self.acc.Macro2mem:
            trans[1] = 0
        Cons = [(trans[op] if xMem[loopidx, op] == 1 else 0) for op in range(3)]

        dflag = [self.dataflow.usr_defined_double_flag[mem_nxt[op]][op] for op in range(3)]

        N = self.dataflow.tm[loopidx].dimSize
        if N == 0:
            return

        # === iter 0 === （跳过 i>0 分支）
        self._analytical_iter_body(
            loopidx=loopidx, i_gt_0=False,
            mem_cur=mem_cur, mem_nxt=mem_nxt,
            tileSize=tileSize, nxtSize=nxtSize,
            Cons=Cons, dflag=dflag,
        )

        if N == 1:
            self._analytical_flush(
                loopidx=loopidx,
                mem_cur=mem_cur, mem_nxt=mem_nxt,
                tileSize=tileSize, nxtSize=nxtSize,
                Cons=Cons, dflag=dflag,
            )
            return

        # iter 1 带 i>0 分支
        self._analytical_iter_body(
            loopidx=loopidx, i_gt_0=True,
            mem_cur=mem_cur, mem_nxt=mem_nxt,
            tileSize=tileSize, nxtSize=nxtSize,
            Cons=Cons, dflag=dflag,
        )

        if N == 2:
            self._analytical_flush(
                loopidx=loopidx,
                mem_cur=mem_cur, mem_nxt=mem_nxt,
                tileSize=tileSize, nxtSize=nxtSize,
                Cons=Cons, dflag=dflag,
            )
            return

        # === iter 2 === snapshot 在前，取 iter2 delta 作为稳态（由 iter1 已收敛到稳定
        # 的相对 gap；iter2 的 max-pattern 与 iter1 相同，作为线性外推基准）
        snap_timer = dict(self.timer)
        snap_mem_r = {m: self.memCost[m].r for m in self.memCost}
        snap_mem_w = {m: self.memCost[m].w for m in self.memCost}
        snap_mem_t = {m: self.memCost[m].t for m in self.memCost}
        snap_tc = list(self.PD.transfer_cycles)
        snap_ms = list(self.PD.mismatch_cycles)
        snap_mode = list(self.PD.mode_switch_cycles)
        snap_owb = self.PD.output_writeback_cycles
        snap_cm = self.count_mac

        self._analytical_iter_body(
            loopidx=loopidx, i_gt_0=True,
            mem_cur=mem_cur, mem_nxt=mem_nxt,
            tileSize=tileSize, nxtSize=nxtSize,
            Cons=Cons, dflag=dflag,
        )

        scale = N - 3
        if scale > 0:
            for k in snap_timer:
                self.timer[k] += scale * (self.timer[k] - snap_timer[k])
            for m in snap_mem_r:
                self.memCost[m].r += scale * (self.memCost[m].r - snap_mem_r[m])
                self.memCost[m].w += scale * (self.memCost[m].w - snap_mem_w[m])
                self.memCost[m].t += scale * (self.memCost[m].t - snap_mem_t[m])
            for op in range(3):
                self.PD.transfer_cycles[op]    += scale * (self.PD.transfer_cycles[op]    - snap_tc[op])
                self.PD.mismatch_cycles[op]    += scale * (self.PD.mismatch_cycles[op]    - snap_ms[op])
                self.PD.mode_switch_cycles[op] += scale * (self.PD.mode_switch_cycles[op] - snap_mode[op])
            self.PD.output_writeback_cycles += scale * (self.PD.output_writeback_cycles - snap_owb)
            self.count_mac += scale * (self.count_mac - snap_cm)

        self._analytical_flush(
            loopidx=loopidx,
            mem_cur=mem_cur, mem_nxt=mem_nxt,
            tileSize=tileSize, nxtSize=nxtSize,
            Cons=Cons, dflag=dflag,
        )


    def _analytical_iter_body(self, loopidx, i_gt_0, mem_cur, mem_nxt,
                              tileSize, nxtSize, Cons, dflag):
        # pre-read (与 loopExecution lines 267-290 对齐)
        for op in range(3):
            if dflag[op] == 1:
                self.timer[mem_cur[op], op] = self.timer[mem_cur[op], op] + Cons[op]
                stall = max(0, max(self.timer[mem_nxt[op], op], self.timer[mem_cur[op], op]) - self.timer[mem_nxt[op], op])
                self.timer[mem_nxt[op], op] = max(self.timer[mem_nxt[op], op], self.timer[mem_cur[op], op])
                self.PD.transfer_cycles[op] += Cons[op]
                self.PD.mismatch_cycles[op] += stall
            else:
                self.timer[mem_cur[op], op] = max(self.timer[mem_cur[op], op], self.timer[mem_nxt[op], op]) + Cons[op]
                self.timer[mem_nxt[op], op] = self.timer[mem_cur[op], op]
                stall = Cons[op]
                self.PD.transfer_cycles[op] += Cons[op]
                self.PD.mode_switch_cycles[op] += Cons[op]
            self.memCost[mem_cur[op]].t += stall

            if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                self.memCost[self.lastMappingMem[op]].r += self.acc.cost_r[self.lastMappingMem[op]] * tileSize[op] * self._prec(self.lastMappingMem[op], op)
                self.memCost[self.acc.lastMem[op]].w += self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op] * self._prec(self.acc.lastMem[op], op)
                self.memCost[self.acc.lastMem[op]].r += self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op] * self._prec(self.acc.lastMem[op], op)
                self.memCost[mem_nxt[op]].w += self.acc.cost_w[mem_nxt[op]] * nxtSize[op] * self._prec(mem_nxt[op], op)
            else:
                self.memCost[mem_cur[op]].r += self.acc.cost_r[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)
                self.memCost[mem_nxt[op]].w += self.acc.cost_w[mem_nxt[op]] * nxtSize[op] * self._prec(mem_nxt[op], op)

        # double-buffer O 的 pre-child write（仅 i > 0；lines 297-312）
        op = 2
        if i_gt_0 and Cons[op] > 0 and dflag[op] == 1:
            self.timer[mem_cur[op], op] = max(self.timer[mem_cur[op], op], self.timer[mem_nxt[op], op]) + Cons[op]
            self.PD.transfer_cycles[op] += Cons[op]
            self.PD.output_writeback_cycles += Cons[op]

            if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op] * self._prec(mem_nxt[op], op)
                self.memCost[self.acc.lastMem[op]].w += self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op] * self._prec(self.acc.lastMem[op], op)
                self.memCost[self.acc.lastMem[op]].r += self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op] * self._prec(self.acc.lastMem[op], op)
                self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)
            else:
                self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op] * self._prec(mem_nxt[op], op)
                self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)

            self.memCost[mem_cur[op]].t += Cons[op]

        # child 递归一次
        self.loopExecutionAnalytical(loopidx=loopidx + 1)

        # 单缓冲 O 的 post-child write（每 iter 都跑；lines 317-334）
        op = 2
        if dflag[op] != 1:
            self.timer[mem_nxt[op], op] = max(self.timer[mem_cur[op], op], self.timer[mem_nxt[op], op]) + Cons[op]
            self.timer[mem_cur[op], op] = self.timer[mem_nxt[op], op]
            self.PD.transfer_cycles[op] += Cons[op]
            self.PD.output_writeback_cycles += Cons[op]

            if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op] * self._prec(mem_nxt[op], op)
                self.memCost[self.acc.lastMem[op]].w += self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op] * self._prec(self.acc.lastMem[op], op)
                self.memCost[self.acc.lastMem[op]].r += self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op] * self._prec(self.acc.lastMem[op], op)
                self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)
            else:
                self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op] * self._prec(mem_nxt[op], op)
                self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)

            self.memCost[mem_cur[op]].t += Cons[op]


    def _analytical_flush(self, loopidx, mem_cur, mem_nxt,
                          tileSize, nxtSize, Cons, dflag):
        # double-buffer O 末尾 flush（lines 339-355）
        op = 2
        N = self.dataflow.tm[loopidx].dimSize
        if N > 0 and Cons[op] > 0 and dflag[op] == 1:
            self.timer[mem_cur[op], op] = max(self.timer[mem_cur[op], op], self.timer[mem_nxt[op], op]) + Cons[op]
            self.PD.transfer_cycles[op] += Cons[op]
            self.PD.output_writeback_cycles += Cons[op]

            if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op] * self._prec(mem_nxt[op], op)
                self.memCost[self.acc.lastMem[op]].w += self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op] * self._prec(self.acc.lastMem[op], op)
                self.memCost[self.acc.lastMem[op]].r += self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op] * self._prec(self.acc.lastMem[op], op)
                self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)
            else:
                self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op] * self._prec(mem_nxt[op], op)
                self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self._prec(mem_cur[op], op)

            self.memCost[mem_cur[op]].t += Cons[op]


    def run_analytical(self):
        return self._run_impl(use_analytical=True)


    def run(self):
        return self._run_impl(use_analytical=False)


    def _run_impl(self, use_analytical: bool):
        Logger.critical("Evaluation by running translation simulator"
                        + (" [analytical]" if use_analytical else ""))

        self.firstMemDram = {}
        self.firstMappingMem = {}
        self.lastMemReg = {}
        self.lastMappingMem = {}
        for op, op_name in enumerate(['I','W','O']):
            self.firstMappingMem[op] = self.dataflow.tm[0].mem[op]
            self.firstMemDram[op] = True if self.dataflow.tm[0].mem[op] == self.acc.Dram2mem else False
            self.lastMappingMem[op] = self.dataflow.tm[-1].mem[op]
            self.lastMemReg[op] = True if self.lastMappingMem[op] == self.acc.lastMem[op] else False


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # Start Running Simulator  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        for op, op_name in enumerate(['I','W','O']):
            if self.firstMemDram[op] is False:
                dram_prec = self._prec(self.acc.Dram2mem, op)
                bootstrap_cycles = self.ops.size[op] * dram_prec / self.acc.bw[self.acc.Dram2mem]
                self.timer[self.acc.Dram2mem, op]        += bootstrap_cycles
                self.timer[self.firstMappingMem[op], op] += bootstrap_cycles
                self.PD.bootstrap_cycles += bootstrap_cycles

        if use_analytical:
            self.loopExecutionAnalytical(0)
        else:
            self.loopExecution(0)

        for op, op_name in enumerate(['I','W','O']):
            if op_name == 'O' and self.firstMemDram[op] is False:
                dram_prec = self._prec(self.acc.Dram2mem, op)
                bootstrap_cycles = self.ops.size[op] * dram_prec / self.acc.bw[self.acc.Dram2mem]
                self.timer[self.acc.Dram2mem, op]        += bootstrap_cycles
                self.timer[self.firstMappingMem[op], op] += bootstrap_cycles
                self.PD.bootstrap_cycles += bootstrap_cycles

        res_Latency = max(self.timer[mem, op] for op in range(3) for mem in range(self.acc.Num_mem))

        for op, op_name in enumerate(['I','W','O']):
            if self.firstMemDram[op] is False:
                # DRAM supplies the full operand in bootstrap (matches line 376's latency accounting).
                # firstMappingMem's write side stays per-instance; line 402-404 scales it by instance count.
                self.memCost[self.acc.Dram2mem].r += self.ops.size[op] * \
                                                    self.acc.cost_r[self.acc.Dram2mem] * self._prec(self.acc.Dram2mem, op)
                self.memCost[self.firstMappingMem[op]].w += self.dataSize[self.firstMappingMem[op],op] * \
                                                       self.acc.cost_w[self.firstMappingMem[op]] * self._prec(self.firstMappingMem[op], op)
        
        for m in range(1, self.acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if self.acc.mappingArray[op][m] == 1 and self.instance_Used_count[m,op] > 1:
                    self.memCost[m].r *= self.instance_Used_count[m,op]
                    self.memCost[m].w *= self.instance_Used_count[m,op]
                    continue

        self.memCost[self.acc.Macro2mem].r = 0

        cost_r, cost_w = 0, 0
        for m in range(1,self.acc.Num_mem):
            self.PD.memCost.append(0)
            cost_r += self.memCost[m].r
            cost_w += self.memCost[m].w
            self.PD.memCost[m] += self.memCost[m].r + self.memCost[m].w

        cost_m = self.acc.cost_ActMacro * self.count_mac * self.instance_Used_count[4,0]
                                                           # the number of multi-core is equal with the number of input buffer instance

        cost_s = res_Latency * self.acc.leakage_per_cycle

        res_Energy = cost_r+cost_w+cost_m+cost_s
        
        Logger.info(f"* * SimuLax-Running  * *  Latency:{res_Latency}, MAC-times:{self.count_mac}, MAC-Latency:{self.count_mac*self.acc.t_MAC}, MAC-Energy:{cost_m:.3e} nJ")
        Logger.info(f"* * SimuLax-Running  * *  EnergyCost:{round(res_Energy,3)}, Dynamic({cost_r+cost_w+cost_m}), Leakage:{round(cost_s,3)}")
        # Logger.info(f"* * SimuLax-Running  * *  readOUT:{round(cost_r,3)}, writeIN:{round(cost_w,3)}, MAC:{round(cost_m,3)}")
        Logger.info(f"* * SimuLax-Running  * *  Memory Cost:") 
        for m in range(1,self.acc.Num_mem):
            Logger.info('-'*4 + f" {self.acc.mem2dict(m)}: {(self.memCost[m].r + self.memCost[m].w):.3e} nJ" +
                        f" {self.memCost[m].r:.3e}(Read) + {self.memCost[m].w:.3e}(Write), "
                        f" ({round((self.memCost[m].r + self.memCost[m].w)/res_Energy*100,2)}%)")
        Logger.info(f"* * SimuLax-Running  * *  Energy BreakDown: " + 
                    f"MemoryAccess(On/Off):{round((cost_r+cost_w-(self.memCost[1].r+self.memCost[1].w))/res_Energy*100,2)}%/{round((self.memCost[1].r+self.memCost[1].w)/res_Energy*100,2)}%, " +
                    f"MAC:{round(cost_m/res_Energy*100,2)}%, " + 
                    f"Leakage:{round(cost_s/res_Energy*100,2)}%" ) 
        Logger.info(f"* * SimuLax-Running  * *  EDP:{((res_Energy)*res_Latency):.3e}")
        Logger.info('* '*50+'\n')

        self.PD.dynamic_power = cost_r+cost_w+cost_m
        self.PD.dynamic_power_onChip = self.PD.dynamic_power - self.PD.memCost[1]
        self.PD.leakage_power = cost_s
        self.PD.macEnergy = cost_m

        self.PD.latency = res_Latency
        self.PD.macLatency = self.count_mac * self.acc.t_MAC
        self.PD.mode_switch_stall = sum(self.PD.mode_switch_cycles)
        self.PD.mismatch_stall = sum(self.PD.mismatch_cycles)
        self.PD.writeback_stall = self.PD.output_writeback_cycles
        self.PD.idle_cycles = max(
            0,
            self.PD.latency
            - self.PD.macLatency
            - self.PD.mode_switch_stall
            - self.PD.mismatch_stall
            - self.PD.writeback_stall,
        )

        self.PD.count_mac = self.count_mac
        self.PD.peak_mac_per_cycle = self.acc.Num_core * self.acc.dimX * self.acc.dimY

        # Fallback byte accumulation: derive from memCost energy / cost_per_bit.
        # Direct per-access accumulation would require touching all 30+ memCost +=
        # sites; energy-inversion is bit-exact for constant cost_r/cost_w per level.
        # Guard against zero cost (CIM Macro read is zeroed at line ~659).
        self.PD.bytes_read = []
        self.PD.bytes_written = []
        for m in range(1, self.acc.Num_mem):
            cr = self.acc.cost_r[m]
            cw = self.acc.cost_w[m]
            br = (self.memCost[m].r / cr / 8) if cr and cr > 0 else 0.0
            bw = (self.memCost[m].w / cw / 8) if cw and cw > 0 else 0.0
            self.PD.bytes_read.append(br)
            self.PD.bytes_written.append(bw)

        return res_Latency, res_Energy

    def debugLog(self):
        acc = self.acc
        ops = self.ops
        pstr = "\n"
        
        str_tpsp = "dim_tp & dim_sp \n"
        for mem in range(1,acc.Num_mem):
            # str_tpsp += '- ' * 10 + f"{acc.mem2dict(mem)}: dim_tp & dim_sp " + '- ' * 10 + '\n'
            str_tpsp += '- - - - - tp - - - - - ' + f"{acc.mem2dict(mem):^15}" + '- - - - - sp - - - - - ' + '\n'
            for op, op_name in enumerate(['I','W','O']):
                str_tpsp += f"{op_name}:"
                for dim in range(1,ops.Num_dim):
                    str_tpsp += f"{ops.dim2Dict[dim]}:{self.dim_tp[mem][op][dim]} "
                str_tpsp += '    |    '
                for dim in range(1,ops.Num_dim):
                    str_tpsp += f"{ops.dim2Dict[dim]}:{self.dim_sp[mem][op][dim]} "
                str_tpsp += '\n'
        # pstr += str_tpsp

        
        str_mem = "Data element size:\n"
        for mem in range(1,acc.Num_mem):
            str_mem += f"{acc.mem2dict(mem):<15}:"
            for t in range(3):
                str_mem += f"{self.dataSize[mem,t]:>8}"
            str_mem += '\n'
        pstr += str_mem
        
        str_relat_mem = "\n"
        for mem in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):    
                if self.dataflow.bypassMem[mem][op] == 0:
                    str_relat_mem += f"{acc.mem2dict(mem)}({op_name}): "
                    str_relat_mem += f"next_mem is {acc.mem2dict(self.dataflow.nxtmem[mem,op])}"
                    str_relat_mem += '\n'
        # pstr += str_relat_mem


        return pstr
