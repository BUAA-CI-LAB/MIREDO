import math

from utils.GlobalUT import Logger as Logger

def convert_mapping_to_next(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    nxtMem = [[-1 for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        last_valid_layer = -1
        for col in range(cols):
            if matrix[row][col] == 1:
                next_layer = -1
                for next_col in range(col + 1, cols):
                    if matrix[row][next_col] == 1:
                        next_layer = next_col
                        break
                if next_layer == -1:
                    next_layer = cols
                nxtMem[row][col] = next_layer
                last_valid_layer = col
    return nxtMem

def find_lastMem_index(mapArray, idx):
    count = 0
    for i in range(len(mapArray) - 1, -1, -1):
        if mapArray[i] == 1:
            count += 1
            if count == idx:
                return i
    return -1

def find_FirstMem_index(mapArray, idx):
    count = 0
    for i in range(1,len(mapArray)):
        if mapArray[i] == 1:
            count += 1
            if count == idx:
                return i
    return -1

class CIM_Acc():

    def mem2dict(self,x):
        if x>=0 and x<self.Num_mem:
            return self._mem2dict[int(x)]
        elif x == self.Num_mem:
            return 'opMAC'
        else:
            return '---'

    @property
    def placement_depth(self):
        return max(sum(self.mappingArray[op][m] for m in range(1, self.Num_mem))
                   for op in range(3))

    @classmethod
    def from_spec(cls, spec) -> "CIM_Acc":
        Logger.critical("Create CIM Acc (from_spec)")
        self = cls.__new__(cls)
        self._mem2dict = ['-P-']
        self.memSize = [-1]
        self.bw = [-1]
        self.cost_r = [-1]
        self.cost_w = [-1]
        self.mappingArray = [[-1], [-1], [-1]]

        I = spec.macro.precision.I
        W = spec.macro.precision.W
        psum = spec.macro.precision.psum
        O_final = spec.macro.precision.O_final

        op_name_to_idx = {'I': 0, 'W': 1, 'O': 2}

        for level in spec.memory_hierarchy:
            self._mem2dict.append(level.name)
            self.bw.append(level.r_bw_bits_per_cycle)
            self.cost_r.append(level.r_cost_per_bit_pJ)
            self.cost_w.append(level.w_cost_per_bit_pJ)
            self.memSize.append(level.size_bits)
            for t_name in ['I', 'W', 'O']:
                t = op_name_to_idx[t_name]
                self.mappingArray[t].append(1 if t_name in level.operands else 0)

        self.Num_mem = len(self._mem2dict)
        self.cost_r.append(0)
        self.cost_w.append(0)

        self.IReg2mem  = find_lastMem_index(self.mappingArray[0], 1)
        self.Macro2mem = find_lastMem_index(self.mappingArray[1], 1)
        self.OReg2mem  = find_lastMem_index(self.mappingArray[2], 1)
        self.Dram2mem  = 1
        self.Global2mem = self._mem2dict.index("Global_buffer")

        depth = spec.macro.compartment_depth
        if depth < 1:
            raise ValueError(f"compartment_depth ({depth}) must be >= 1.")
        expected_macro_bits = depth * W
        if self.memSize[self.Macro2mem] != expected_macro_bits:
            raise ValueError(
                f"Macro memory level size_bits ({self.memSize[self.Macro2mem]}) "
                f"!= compartment_depth * W ({depth} * {W} = {expected_macro_bits}). "
                f"These must be consistent; update default.py / spec dict."
            )

        self.lastMem = {0: self.IReg2mem, 1: self.Macro2mem, 2: self.OReg2mem}

        self.double_Macro = 1 if bool(getattr(spec.macro, "weight_double_buffer", False)) else 0
        self.double_config = [[0 for _ in range(3)] for _ in range(self.Num_mem)]
        for m in range(1, self.Num_mem):
            for op in range(3):
                if self.mappingArray[op][m] == 1:
                    self.double_config[m][op] = 1
        self.double_config[self.Macro2mem][1] = self.double_Macro
        self.double_config[self.IReg2mem][0] = 0
        self.double_config[self.OReg2mem][2] = 0

        self.shareMemory = [False for _ in range(self.Num_mem)]
        for m in range(1, self.Num_mem):
            opShared = sum(self.mappingArray[op][m] for op in range(3))
            if opShared > 1:
                self.shareMemory[m] = True
            elif opShared == 1:
                self.shareMemory[m] = False
            else:
                raise ValueError(f"Memory {self._mem2dict[m]} have no operands.")

        self.nxtMem = convert_mapping_to_next(self.mappingArray)

        self.precision = {}
        for mem in range(1, self.Num_mem + 1):
            self.precision[mem, 0] = I
            self.precision[mem, 1] = W
            self.precision[mem, 2] = O_final
        self.precision[self.OReg2mem, 2] = psum
        self.precision_final = O_final
        self.precision_psum = psum

        self.Num_core = spec.cores
        assert spec.macro.input_bit_per_cycle == 1

        self.dimX = spec.macro.dimX
        self.dimY = spec.macro.dimY

        self.SpUnrolling = [self.Num_core, self.dimX, self.dimY]
        self.Num_SpUr = len(self.SpUnrolling)

        dim_names = ['_', 'R', 'S', 'P', 'Q', 'C', 'K', 'G']
        dim_idx_map = {n: i for i, n in enumerate(dim_names)}
        self.mappingRule = []
        for axis in spec.macro.spatial_axes:
            row = [0] * len(dim_names)
            for loop in axis.allowed_loops:
                row[dim_idx_map[loop]] = 1
            self.mappingRule.append(row)

        self.SpUr2Mem = {}
        mem_name_to_idx = {name: i for i, name in enumerate(self._mem2dict)}
        for ai, axis in enumerate(spec.macro.spatial_axes):
            for op_name, mem_name in axis.source_memory_per_operand.items():
                self.SpUr2Mem[ai, op_name_to_idx[op_name]] = mem_name_to_idx[mem_name]

        self.minBW = [1e9, 1e9, 1e9]
        for m in range(1, self.Num_mem):
            for op in range(3):
                if self.mappingArray[op][m]:
                    self.minBW[op] = min(self.minBW[op], self.bw[m])

        self.t_MAC = math.ceil(I / spec.macro.input_bit_per_cycle)

        mult_1b = spec.macro.logic_energies_pJ.mult_1b
        adder_1b = spec.macro.logic_energies_pJ.adder_1b
        reg_1b = spec.macro.logic_energies_pJ.reg_1b

        precision_Psum = I + W + math.ceil(math.log2(self.dimX))
        num_bit_MAC_in_Macro = W * spec.macro.input_bit_per_cycle * self.dimX * self.dimY
        num_bit_ADD_in_addTree = self.dimX * (I + 1) - (I + math.ceil(math.log2(I)) + 1) * self.dimY

        CIMArray_energy = mult_1b * num_bit_MAC_in_Macro * self.t_MAC
        Adder_tree_energy = adder_1b * num_bit_ADD_in_addTree * self.t_MAC
        Merge_energy = (adder_1b + reg_1b) * precision_Psum * self.t_MAC * self.dimY

        cost_ActMacro_pJ = CIMArray_energy + Adder_tree_energy + Merge_energy

        def pj_to_nj(value):
            if isinstance(value, list):
                return [x * 1e-3 for x in value]
            return value * 1e-3

        self.cost_ActMacro = pj_to_nj(cost_ActMacro_pJ)
        self.cost_r = pj_to_nj(self.cost_r)
        self.cost_w = pj_to_nj(self.cost_w)
        self.leakage_per_cycle = spec.leakage_per_cycle_nJ

        self.source_spec = spec

        return self
