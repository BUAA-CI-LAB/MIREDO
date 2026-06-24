# this file is prepared for project 026
# Created by iboxl
# Modified in project 112

import math
from utils.UtilsFunction.ToolFunction import getDivisors
from Architecture.ArchSpec import CIM_Acc
from dataclasses import dataclass
from utils.GlobalUT import *

class Operand():
    def __init__(self):
        self.bitwidth :int = None

class Operands():   
    def __init__(self, config, ori_M:int, ori_K:int, ori_N:int, multi:int=1):
        
        self.dim_M = ori_M
        self.dim_K = ori_K
        self.dim_N = ori_N
        self.multi = multi
        # GEMM: [M, K]x[K, N]
        self.weight = Operand()
        self.input = Operand()
        self.output = Operand()

        self.weight.bitwidth = config.getint("Workload", "weight_bit_width")  
        self.input.bitwidth = config.getint("Workload", "input_bit_width")  
        # """     放在计算内部求得 ?
        if self.weight.bitwidth == 8 and self.input.bitwidth == 8 :
            self.output.bitwidth = int(math.pow(2, math.ceil(math.log2(
                (self.weight.bitwidth + self.input.bitwidth) + math.ceil(math.log(ori_K, 2))
            )))   )               #  位数保持为2的x次方
        else:
            self.output.bitwidth = (self.weight.bitwidth + self.input.bitwidth) + math.ceil(math.log(5000, 2))
    
    def exchange(self):
        
        tmp = self.dim_M
        self.dim_M = self.dim_N
        self.dim_N = tmp

        tmp = self.input
        self.input = self.weight
        self.weight = tmp

        return self

class WorkLoad(): 
    def __init__(self, loopDim):
        self.weight = Operand()
        self.input = Operand()
        self.output = Operand()

        loop_dim_keys = ['R', 'S', 'C', 'K', 'P', 'Q', 'G', 'B', 'H', 'W', 'Stride', 'Padding']
        for key in loop_dim_keys:
            setattr(self, key, loopDim[key])

        self.dim2Dict = ['-','R', 'S', 'P', 'Q', 'C', 'K', 'G']
        loopDim['-'] = 1
        self.dim2bound = [loopDim[xx] for xx in self.dim2Dict]

        self.Num_dim = len(self.dim2Dict)

        self.relevance = [      # relevance[t][dim]
            [0,1,1,1,1,1,0,1],      # Input
            [0,1,1,0,0,1,1,1],      # weight
            [0,0,0,1,1,0,1,1]       # Output
        ]

        self.Divisors = [getDivisors(d) for d in self.dim2bound]

        self.size = [self.get_operand_size(self.dim2bound, op) for op, op_name in enumerate(['I','W','O'])]

        Num_MAC = 1
        for dim in range(1, self.Num_dim):
            Num_MAC *= self.dim2bound[dim]
        self.Num_MAC = Num_MAC
        
        assert self.P == (self.H + 2 * self.Padding - self.R)//self.Stride + 1
        assert self.Q == (self.W + 2 * self.Padding - self.S)//self.Stride + 1 

    def _get_input_extent(self, output_size, kernel_size, input_bound):
        unique_extent = kernel_size + (output_size - 1) * min(self.Stride, kernel_size)
        return min(unique_extent, input_bound)

    def _get_input_hw(self, dim_size):
        r_size = dim_size[self.dict2Dim('R')]
        s_size = dim_size[self.dict2Dim('S')]
        p_size = dim_size[self.dict2Dim('P')]
        q_size = dim_size[self.dict2Dim('Q')]

        tmp_h = self._get_input_extent(p_size, r_size, self.H)
        tmp_w = self._get_input_extent(q_size, s_size, self.W)

        return tmp_h, tmp_w

    def get_operand_size(self, dim_size, op):
        if op == 0:
            h_size, w_size = self._get_input_hw(dim_size)
            return h_size * w_size * dim_size[self.dict2Dim('C')] * dim_size[self.dict2Dim('G')]
        if op == 1:
            return (
                dim_size[self.dict2Dim('R')] *
                dim_size[self.dict2Dim('S')] *
                dim_size[self.dict2Dim('C')] *
                dim_size[self.dict2Dim('K')] *
                dim_size[self.dict2Dim('G')]
            )
        if op == 2:
            return (
                dim_size[self.dict2Dim('P')] *
                dim_size[self.dict2Dim('Q')] *
                dim_size[self.dict2Dim('K')] *
                dim_size[self.dict2Dim('G')]
            )
        raise ValueError(f"Unsupported operand index: {op}")
    
    def dict2Dim(self, dimChar):
        try:
            return self.dim2Dict.index(dimChar)
        except ValueError:
            # 如果字符不在列表中，返回一个错误提示或特殊值
            raise ValueError("Error dim char")

    def __repr__(self):
        attrs = ['R', 'S', 'C', 'K', 'P', 'Q', 'G', 'B', 'H', 'W', 'Stride', 'Padding']
        pstr = ", ".join(f"{k}:{getattr(self, k)}" for k in attrs)
        return pstr

@dataclass
class Mapping():
    dim:int
    dimSize:int
    mem:list[int]

class LoopNest():
    def __init__(self, acc:CIM_Acc, ops:WorkLoad):
        self.acc = acc
        self.ops = ops

        self.tm:list[Mapping] = [] # Temporal Mapping List from outer to inner 

        self.sm:list[Mapping] = [] # yin

        self.usr_defined_double_flag = None
        self.psum_flag = None
        
    
    def preprogress(self):
        unrollingSize = [1 for _ in self.ops.dim2Dict]
        for mapping in self.tm+self.sm:
            for op, op_name in enumerate(['I','W','O']):
                if self.acc.mappingArray[op][mapping.mem[op]] == 0:
                    raise ValueError(f"Illegal Memory mapping in LoopNest: Operand-{op_name} in {self.acc.mem2dict(mapping.mem[op])}")
            if 0 <= mapping.dimSize <= self.ops.dim2bound[mapping.dim]:
                pass
            else:
                raise ValueError(f"Illegal dimension {self.ops.dim2Dict[mapping.dim]} bound in LoopNest")
            unrollingSize[mapping.dim] *= mapping.dimSize

            
        if len(self.usr_defined_double_flag) != self.acc.Num_mem+1:
            raise ValueError("Mismatch with usr_difined_double_flag")

        tm = self.tm

        bypassMem = [[1 for op in range(3)] for _ in range(self.acc.Num_mem+1)]    # bypass[mem][op]
        for mapping in tm:
            for op, op_name in enumerate(['I','W','O']):
                bypassMem[mapping.mem[op]][op] = 0
        self.bypassMem = bypassMem

        nxtmem = {}
        uppermem = {}
        for op, op_name in enumerate(['I','W','O']):
            for i in range(len(tm)-1):
                # nxtmem[i,op] = acc.Num_mem
                for j in range(i+1,len(tm)):
                    if tm[i].mem[op] < tm[j].mem[op]:
                        nxtmem[tm[i].mem[op],op] = tm[j].mem[op]
                        break
            nxtmem[tm[-1].mem[op],op] = self.acc.Num_mem
        # Safety net: orphan levels (used but lack a strict greater successor
        # and not equal to tm[-1].mem[op]) get promoted to DRAM-equivalent.
        # MIREDO's own mappings never trigger this; adapter-produced tm may.
        for op, op_name in enumerate(['I','W','O']):
            for mem in range(self.acc.Num_mem+1):
                if bypassMem[mem][op] == 0 and (mem, op) not in nxtmem:
                    Logger.warning(
                        f"preprogress: orphan tm level mem={mem} op={op_name}; "
                        f"defaulting nxtmem to Num_mem={self.acc.Num_mem}"
                    )
                    nxtmem[mem, op] = self.acc.Num_mem
        for op, op_name in enumerate(['I','W','O']):
            for mem in range(self.acc.Num_mem+1):
                if bypassMem[mem][op] == 0:
                    uppermem[nxtmem[mem,op],op] = mem
            uppermem[1,op] = 0
        self.nxtmem = nxtmem
        self.uppermem = uppermem

        xMem = {}
        for op, op_name in enumerate(['I','W','O']):    
            for i in range(len(tm)-1):
                if self.acc.mappingArray[op][tm[i].mem[op]] == 1:
                    if tm[i+1].mem[op] == nxtmem[tm[i].mem[op],op]:
                        xMem[i,op] = 1
                    else:
                        xMem[i,op] = 0
            xMem[len(tm)-1,op] = 1
            for i in range(len(tm)-1,-1,-1):            # fix bottem-ir reuse
                if xMem[i,op] == 1:
                    for j in range(i,-1,-1):
                        if tm[j].mem[op] != tm[i].mem[op]:
                            j += 1
                            break
                        elif self.ops.relevance[op][tm[j].dim] == 0:
                            xMem[j,op] = 0
                        else:
                            break
                    xMem[j,op] = 1
        for i in range(len(tm)):
            if tm[i].mem[1] == self.acc.Macro2mem:
                xMem[i,1] = 0
        self.xMem = xMem

        output_mems = [m for m in range(1, self.acc.Num_mem) if self.acc.mappingArray[2][m] == 1]
        if self.psum_flag is None:
            ir_at_mem = {}
            for mapping in tm:
                if self.ops.relevance[2][mapping.dim] == 0 and mapping.dimSize > 1:
                    ir_at_mem[mapping.mem[2]] = True

            psum_flag = {}
            has_ir_outer = False
            for m in output_mems:
                if ir_at_mem.get(m, False):
                    has_ir_outer = True
                psum_flag[m] = has_ir_outer
            self.psum_flag = psum_flag

        for m in output_mems:
            self.psum_flag.setdefault(m, False)

        self._prec_table = {}
        for mem in range(self.acc.Num_mem + 1):
            for op in range(3):
                if op == 2:
                    if self.psum_flag is not None and self.psum_flag.get(mem, False):
                        self._prec_table[(mem, op)] = self.acc.precision_psum
                    else:
                        self._prec_table[(mem, op)] = self.acc.precision_final
                else:
                    self._prec_table[(mem, op)] = self.acc.precision[mem, op] if (mem, op) in self.acc.precision else 0

        for i,_ in enumerate(self.ops.dim2Dict):
            if abs(unrollingSize[i]-self.ops.dim2bound[i]) > 1:
                Logger.warning(f"Dataflow LoopNest used greedy mapping ({unrollingSize[i]}) for dimension {self.ops.dim2Dict[i]}({self.ops.dim2bound[i]})")  
            if unrollingSize[i] < self.ops.dim2bound[i]:
                Logger.error(self.__repr__())
                raise ValueError(f"Dimension {self.ops.dim2Dict[i]}({self.ops.dim2bound[i]}) unrolling not fully({unrollingSize[i]}) in LoopNest")

    def __repr__(self) -> str:
        pstr = ""
        pstr += ' '*42 + "Spatial Mapping" + ' '*42 + '\n' + "* "*50 + '\n'
        for i,mp in enumerate(self.sm):
            pstr += ' '*i + f"Sp_for {self.ops.dim2Dict[mp.dim]} in {mp.dimSize:<{17-i}}"
            for op in range(3):
                pstr += f"{self.acc.mem2dict(mp.mem[op]):<22}"
            pstr += '\n'
            pstr += "- "*50 + '\n'
        pstr += ' '*42 + "Temporal Mapping" + ' '*42 + '\n' + "* "*50 + '\n'
        for i,mp in enumerate(self.tm):
            pstr += f"{i:<2}" + ' '*i + f"for {self.ops.dim2Dict[mp.dim]} in {mp.dimSize:<{20-i}}"
            for op in range(3):
                pstr += "X| " if self.xMem[i,op]==1 else "   "
                pstr += f"{self.acc.mem2dict(mp.mem[op]):<22}"
            pstr += '\n'
            pstr += "- "*50 + '\n'
        return pstr

    def get_output_precision(self, mem):
        if self.psum_flag is not None and self.psum_flag.get(mem, False):
            return self.acc.precision_psum
        return self.acc.precision_final

    def get_precision(self, mem, op):
        table = getattr(self, "_prec_table", None)
        if table is not None:
            return table[(mem, op)]
        if op == 2:
            return self.get_output_precision(mem)
        return self.acc.precision[mem, op]



        
