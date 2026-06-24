# Self-contained CACTI 7.0 Python driver for MIREDO.
#
# Originally adapted from ZigZag-IMC's get_cacti_cost.py (zigzag/classes/hardware/
# architecture/get_cacti_cost.py). Copied + lifted so MIREDO does not depend on
# ZigZag at runtime — CACTI sits under utils/Cacti_wrapper/cacti/ and is invoked
# through this module exclusively.
#
# API:
#   get_cacti_cost(tech_node, mem_type, mem_size_in_byte, bw, hd_hash=...)
#     -> (access_time_ns, area_mm2, r_cost_pJ_per_access, w_cost_pJ_per_access)
#
# The tech_node=0.028 (28nm) case is handled by calling CACTI at its nearest
# supported planar CMOS node (32nm) and applying a (0.9/1.0)^2 = 0.81 voltage-
# squared scaling factor across all outputs. This is the same approximation
# ZigZag-IMC uses; we keep the convention so the default-spec frozen values in
# Architecture/templates/default.py remain reproducible byte-for-byte.

import os
import platform
from pathlib import Path


_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CACTI_DIR = _MODULE_DIR / "cacti"


class CactiConfig:

    def __init__(self):
        self.baseline_config = ['# power gating\n',
                                '-Array Power Gating - "false"\n',
                                '-WL Power Gating - "false"\n',
                                '-CL Power Gating - "false"\n',
                                '-Bitline floating - "false"\n',
                                '-Interconnect Power Gating - "false"\n',
                                '-Power Gating Performance Loss 0.01\n',
                                '\n',
                                '# following three parameters are meaningful only for main memories\n',
                                '-page size (bits) 8192 \n',
                                '-burst length 8\n',
                                '-internal prefetch width 8\n',
                                '\n',
                                '# following parameter can have one of five values -- (itrs-hp, itrs-lstp, itrs-lop, lp-dram, comm-dram)\n',
                                '-Data array cell type - "itrs-hp"\n',
                                '//-Data array cell type - "itrs-lstp"\n',
                                '//-Data array cell type - "itrs-lop"\n',
                                '\n',
                                '# following parameter can have one of three values -- (itrs-hp, itrs-lstp, itrs-lop)\n',
                                '-Data array peripheral type - "itrs-hp"\n',
                                '//-Data array peripheral type - "itrs-lstp"\n',
                                '//-Data array peripheral type - "itrs-lop"\n',
                                '\n',
                                '# following parameter can have one of five values -- (itrs-hp, itrs-lstp, itrs-lop, lp-dram, comm-dram)\n',
                                '-Tag array cell type - "itrs-hp"\n',
                                '//-Tag array cell type - "itrs-lstp"\n',
                                '//-Tag array cell type - "itrs-lop"\n',
                                '\n',
                                '# following parameter can have one of three values -- (itrs-hp, itrs-lstp, itrs-lop)\n',
                                '-Tag array peripheral type - "itrs-hp"\n',
                                '//-Tag array peripheral type - "itrs-lstp"\n',
                                '//-Tag array peripheral type - "itrs-lop\n',
                                '\n',
                                '\n',
                                '// 300-400 in steps of 10\n',
                                '-operating temperature (K) 360\n',
                                '\n',
                                '# to model special structure like branch target buffers, directory, etc. \n',
                                '# change the tag size parameter\n',
                                '# if you want cacti to calculate the tagbits, set the tag size to "default"\n',
                                '-tag size (b) "default"\n',
                                '//-tag size (b) 22\n',
                                '\n',
                                '# fast - data and tag access happen in parallel\n',
                                '# sequential - data array is accessed after accessing the tag array\n',
                                '# normal - data array lookup and tag access happen in parallel\n',
                                '#          final data block is broadcasted in data array h-tree \n',
                                '#          after getting the signal from the tag array\n',
                                '//-access mode (normal, sequential, fast) - "fast"\n',
                                '-access mode (normal, sequential, fast) - "normal"\n',
                                '//-access mode (normal, sequential, fast) - "sequential"\n',
                                '\n',
                                '\n',
                                '# DESIGN OBJECTIVE for UCA (or banks in NUCA)\n',
                                '-design objective (weight delay, dynamic power, leakage power, cycle time, area) 0:0:0:100:0\n',
                                '\n',
                                '# Percentage deviation from the minimum value \n',
                                '# Ex: A deviation value of 10:1000:1000:1000:1000 will try to find an organization\n',
                                '# that compromises at most 10% delay. \n',
                                '# NOTE: Try reasonable values for % deviation. Inconsistent deviation\n',
                                '# percentage values will not produce any valid organizations. For example,\n',
                                '# 0:0:100:100:100 will try to identify an organization that has both\n',
                                '# least delay and dynamic power. Since such an organization is not possible, CACTI will\n',
                                '# throw an error. Refer CACTI-6 Technical report for more details\n',
                                '-deviate (delay, dynamic power, leakage power, cycle time, area) 20:100000:100000:100000:100000\n',
                                '\n',
                                '# Objective for NUCA\n',
                                '-NUCAdesign objective (weight delay, dynamic power, leakage power, cycle time, area) 100:100:0:0:100\n',
                                '-NUCAdeviate (delay, dynamic power, leakage power, cycle time, area) 10:10000:10000:10000:10000\n',
                                '\n',
                                '# Set optimize tag to ED or ED^2 to obtain a cache configuration optimized for\n',
                                '# energy-delay or energy-delay sq. product\n',
                                '# Note: Optimize tag will disable weight or deviate values mentioned above\n',
                                '# Set it to NONE to let weight and deviate values determine the \n',
                                '# appropriate cache configuration\n',
                                '//-Optimize ED or ED^2 (ED, ED^2, NONE): "ED"\n',
                                '-Optimize ED or ED^2 (ED, ED^2, NONE): "ED^2"\n',
                                '//-Optimize ED or ED^2 (ED, ED^2, NONE): "NONE"\n',
                                '\n',
                                '-Cache model (NUCA, UCA)  - "UCA"\n',
                                '//-Cache model (NUCA, UCA)  - "NUCA"\n',
                                '\n',
                                '# In order for CACTI to find the optimal NUCA bank value the following\n',
                                '# variable should be assigned 0.\n',
                                '-NUCA bank count 0\n',
                                '\n',
                                '# NOTE: for nuca network frequency is set to a default value of \n',
                                '# 5GHz in time.c. CACTI automatically\n',
                                '# calculates the maximum possible frequency and downgrades this value if necessary\n',
                                '\n',
                                '# By default CACTI considers both full-swing and low-swing \n',
                                '# wires to find an optimal configuration. However, it is possible to \n',
                                '# restrict the search space by changing the signaling from "default" to \n',
                                '# "fullswing" or "lowswing" type.\n',
                                '-Wire signaling (fullswing, lowswing, default) - "Global_30"\n',
                                '//-Wire signaling (fullswing, lowswing, default) - "default"\n',
                                '//-Wire signaling (fullswing, lowswing, default) - "lowswing"\n',
                                '\n',
                                '//-Wire inside mat - "global"\n',
                                '-Wire inside mat - "semi-global"\n',
                                '//-Wire outside mat - "global"\n',
                                '-Wire outside mat - "semi-global"\n',
                                '\n',
                                '-Interconnect projection - "conservative"\n',
                                '//-Interconnect projection - "aggressive"\n',
                                '\n',
                                '# Contention in network (which is a function of core count and cache level) is one of\n',
                                '# the critical factor used for deciding the optimal bank count value\n',
                                '# core count can be 4, 8, or 16\n',
                                '//-Core count 4\n',
                                '-Core count 8\n',
                                '//-Core count 16\n',
                                '-Cache level (L2/L3) - "L3"\n',
                                '\n',
                                '-Add ECC - "true"\n',
                                '\n',
                                '//-Print level (DETAILED, CONCISE) - "CONCISE"\n',
                                '-Print level (DETAILED, CONCISE) - "DETAILED"\n',
                                '\n',
                                '# for debugging\n',
                                '-Print input parameters - "true"\n',
                                '//-Print input parameters - "false"\n',
                                '# force CACTI to model the cache with the \n',
                                '# following Ndbl, Ndwl, Nspd, Ndsam,\n',
                                '# and Ndcm values\n',
                                '//-Force cache config - "true"\n',
                                '-Force cache config - "false"\n',
                                '-Ndwl 1\n',
                                '-Ndbl 1\n',
                                '-Nspd 0\n',
                                '-Ndcm 1\n',
                                '-Ndsam1 0\n',
                                '-Ndsam2 0\n',
                                '\n',
                                '\n',
                                '\n',
                                '#### Default CONFIGURATION values for baseline external IO parameters to DRAM. More details can be found in the CACTI-IO technical report (), especially Chapters 2 and 3.\n',
                                '\n',
                                '# Memory Type (D3=DDR3, D4=DDR4, L=LPDDR2, W=WideIO, S=Serial). Additional memory types can be defined by the user in extio_technology.cc, along with their technology and configuration parameters.\n',
                                '\n',
                                '-dram_type "DDR3"\n',
                                '//-dram_type "DDR4"\n',
                                '//-dram_type "LPDDR2"\n',
                                '//-dram_type "WideIO"\n',
                                '//-dram_type "Serial"\n',
                                '\n',
                                '# Memory State (R=Read, W=Write, I=Idle  or S=Sleep) \n',
                                '\n',
                                '//-io state  "READ"\n',
                                '-io state "WRITE"\n',
                                '//-io state "IDLE"\n',
                                '//-io state "SLEEP"\n',
                                '\n',
                                '#Address bus timing. To alleviate the timing on the command and address bus due to high loading (shared across all memories on the channel), the interface allows for multi-cycle timing options. \n',
                                '\n',
                                '//-addr_timing 0.5 //DDR\n',
                                '-addr_timing 1.0 //SDR (half of DQ rate)\n',
                                '//-addr_timing 2.0 //2T timing (One fourth of DQ rate)\n',
                                '//-addr_timing 3.0 // 3T timing (One sixth of DQ rate)\n',
                                '\n',
                                '# Memory Density (Gbit per memory/DRAM die)\n',
                                '\n',
                                '-mem_density 4 Gb //Valid values 2^n Gb\n',
                                '\n',
                                '# IO frequency (MHz) (frequency of the external memory interface).\n',
                                '\n',
                                '-bus_freq 800 MHz //As of current memory standards (2013), valid range 0 to 1.5 GHz for DDR3, 0 to 533 MHz for LPDDR2, 0 - 800 MHz for WideIO and 0 - 3 GHz for Low-swing differential. However this can change, and the user is free to define valid ranges based on new memory types or extending beyond existing standards for existing dram types.\n',
                                '\n',
                                '# Duty Cycle (fraction of time in the Memory State defined above)\n',
                                '\n',
                                '-duty_cycle 1.0 //Valid range 0 to 1.0\n',
                                '\n',
                                '# Activity factor for Data (0->1 transitions) per cycle (for DDR, need to account for the higher activity in this parameter. E.g. max. activity factor for DDR is 1.0, for SDR is 0.5)\n',
                                ' \n',
                                '-activity_dq 1.0 //Valid range 0 to 1.0 for DDR, 0 to 0.5 for SDR\n',
                                '\n',
                                '# Activity factor for Control/Address (0->1 transitions) per cycle (for DDR, need to account for the higher activity in this parameter. E.g. max. activity factor for DDR is 1.0, for SDR is 0.5)\n',
                                '\n',
                                '-activity_ca 0.5 //Valid range 0 to 1.0 for DDR, 0 to 0.5 for SDR, 0 to 0.25 for 2T, and 0 to 0.17 for 3T\n',
                                '\n',
                                '# Number of DQ pins \n',
                                '\n',
                                '-num_dq 72 //Number of DQ pins. Includes ECC pins.\n',
                                '\n',
                                '# Number of DQS pins. DQS is a data strobe that is sent along with a small number of data-lanes so the source synchronous timing is local to these DQ bits. Typically, 1 DQS per byte (8 DQ bits) is used. The DQS is also typucally differential, just like the CLK pin. \n',
                                '\n',
                                '-num_dqs 18 //2 x differential pairs. Include ECC pins as well. Valid range 0 to 18. For x4 memories, could have 36 DQS pins.\n',
                                '\n',
                                '# Number of CA pins \n',
                                '\n',
                                '-num_ca 25 //Valid range 0 to 35 pins.\n',
                                '\n',
                                '# Number of CLK pins. CLK is typically a differential pair. In some cases additional CLK pairs may be used to limit the loading on the CLK pin. \n',
                                '\n',
                                '-num_clk  2 //2 x differential pair. Valid values: 0/2/4.\n',
                                '\n',
                                '# Number of Physical Ranks\n',
                                '\n',
                                '-num_mem_dq 2 //Number of ranks (loads on DQ and DQS) per buffer/register. If multiple LRDIMMs or buffer chips exist, the analysis for capacity and power is reported per buffer/register. \n',
                                '\n',
                                '# Width of the Memory Data Bus\n',
                                '\n',
                                '-mem_data_width 8 //x4 or x8 or x16 or x32 memories. For WideIO upto x128.\n',
                                '\n',
                                '# RTT Termination Resistance\n',
                                '\n',
                                '-rtt_value 10000\n',
                                '\n',
                                '# RON Termination Resistance\n',
                                '\n',
                                '-ron_value 34\n',
                                '\n',
                                '# Time of flight for DQ\n',
                                '\n',
                                '-tflight_value\n',
                                '\n',
                                '# Parameter related to MemCAD\n',
                                '\n',
                                '# Number of BoBs: 1,2,3,4,5,6,\n',
                                '-num_bobs 1\n',
                                '\t\n',
                                '# Memory System Capacity in GB\n',
                                '-capacity 80\t\n',
                                '\t\n',
                                '# Number of Channel per BoB: 1,2. \n',
                                '-num_channels_per_bob 1\t\n',
                                '\n',
                                '# First Metric for ordering different design points\t\n',
                                '-first metric "Cost"\n',
                                '#-first metric "Bandwidth"\n',
                                '#-first metric "Energy"\n',
                                '\t\n',
                                '# Second Metric for ordering different design points\t\n',
                                '#-second metric "Cost"\n',
                                '-second metric "Bandwidth"\n',
                                '#-second metric "Energy"\n',
                                '\n',
                                '# Third Metric for ordering different design points\t\n',
                                '#-third metric "Cost"\n',
                                '#-third metric "Bandwidth"\n',
                                '-third metric "Energy"\t\n',
                                '\t\n',
                                '\t\n',
                                '# Possible DIMM option to consider\n',
                                '#-DIMM model "JUST_UDIMM"\n',
                                '#-DIMM model "JUST_RDIMM"\n',
                                '#-DIMM model "JUST_LRDIMM"\n',
                                '-DIMM model "ALL"\n',
                                '\n',
                                '#if channels of each bob have the same configurations\n',
                                '#-mirror_in_bob "T"\n',
                                '-mirror_in_bob "F"\n',
                                '\n',
                                '#if we want to see all channels/bobs/memory configurations explored\t\n',
                                '#-verbose "T"\n',
                                '#-verbose "F"\n',
                                '\n',
                                '=======USER DEFINE======= \n']

        self.config_options = {}
        self.config_options['cache_size'] = {'string': '-size (bytes) ',
                                             'option': [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                                                        65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
                                                        8388608, 16777216, 33554432, 134217728, 67108864,
                                                        1073741824],
                                             'default': 64}
        self.config_options['line_size'] = {'string': '-block size (bytes) ',
                                            'option': [8, 16, 24],
                                            'default': 64}
        self.config_options['IO_bus_width'] = {'string': '-output/input bus width ',
                                               'option': [4, 8, 16, 24, 32, 64, 128],
                                               'default': 64}
        self.config_options['associativity'] = {'string': '-associativity ',
                                                'option': [0, 1, 2, 4],
                                                'default': 1}
        self.config_options['rd_wr_port'] = {'string': '-read-write port ',
                                             'option': [0, 1, 2, 3, 4],
                                             'default': 1}
        self.config_options['ex_rd_port'] = {'string': '-exclusive read port ',
                                             'option': [0, 1, 2, 3, 4],
                                             'default': 0}
        self.config_options['ex_wr_port'] = {'string': '-exclusive write port ',
                                             'option': [0, 1, 2, 3, 4],
                                             'default': 0}
        self.config_options['single_rd_port'] = {'string': '-single ended read ports ',
                                                 'option': [0, 1, 2, 3, 4],
                                                 'default': 0}
        self.config_options['bank_count'] = {'string': '-UCA bank count ',
                                             'option': [1, 2, 4, 8, 16],
                                             'default': 1}
        self.config_options['technology'] = {'string': '-technology (u) ',
                                             'option': [0.022, 0.028, 0.040, 0.032, 0.065, 0.090],
                                             'default': 0.065}
        self.config_options['mem_type'] = {'string': '-cache type ',
                                           'option': ['"cache"', '"ram"', '"main memory"'],
                                           'default': '"ram"'}
        self.config_options['temperature'] = {'string': '-operating temperature (K) ',
                                              'option': [300, 310, 320, 330],
                                              'default': 300}

    def change_default_value(self, name_list, new_value_list):
        for idx, name in enumerate(name_list):
            self.config_options[name]['default'] = new_value_list[idx]

    def write_config(self, user_config, path):
        f = open(path, "w+")
        f.write(''.join(self.baseline_config))
        f.write(''.join(user_config))
        f.close()

    def call_cacti(self, path):
        stream = os.popen('./cacti -infile %s &> /dev/null' % path)
        output = stream.readlines()
        return output

    def cacti_auto(self, user_input, path):
        '''
        user_input format can be 1 out of these 3:
        user_input = ['default']
        user_input = ['single', [['mem_type', 'technology', ...], ['"ram"', 0.028, ...]]
        user_input = ['sweep', ['IO_bus_width'/'']]
        '''
        user_config = []
        if user_input[0] == 'default':
            for itm in self.config_options.keys():
                user_config.append(self.config_options[itm]['string'] + str(self.config_options[itm]['default']) + '\n')
            self.write_config(user_config, path)
            self.call_cacti(path)

        if user_input[0] == 'single':
            for itm in self.config_options.keys():
                if itm in user_input[1][0]:
                    ii = user_input[1][0].index(itm)
                    user_config.append(self.config_options[itm]['string'] + str(user_input[1][1][ii]) + '\n')
                else:
                    user_config.append(self.config_options[itm]['string'] + str(self.config_options[itm]['default']) + '\n')
            self.write_config(user_config, path)
            self.call_cacti(path)

        if user_input[0] == 'sweep':
            common_part = []
            for itm in self.config_options.keys():
                if itm not in user_input[1]:
                    common_part.append(self.config_options[itm]['string'] + str(self.config_options[itm]['default']) + '\n')

            for itm in user_input[1]:
                for va in self.config_options[itm]['option']:
                    user_config.append([self.config_options[itm]['string'] + str(va) + '\n'])

            for ii in range(len(user_config)):
                user_config[ii] += common_part

            for ii in range(len(user_config)):
                self.write_config(user_config[ii], path)
                self.call_cacti(path)


def get_cacti_cost(tech_node, mem_type, mem_size_in_byte, bw, hd_hash="a", cacti_path=None):
    '''
    Run CACTI and extract (access_time, area, r_cost, w_cost).

    :param tech_node:           technology node in um (natively supported: 0.022, 0.032, 0.040,
                                0.045, 0.065, 0.090; 0.028 remapped to 0.032 + 0.81 scaling).
    :param mem_type:            'sram' or 'dram'.
    :param mem_size_in_byte:    memory capacity in bytes (CACTI minimum: 64 B, minimum_rows: 32).
    :param bw:                  memory IO bitwidth.
    :param hd_hash:             suffix for the generated cache_{hd_hash}.cfg(.out) filenames —
                                use a unique value per call to avoid races in parallel sweeps.
    :param cacti_path:          root directory holding the `cacti` binary. Defaults to the copy
                                vendored under utils/Cacti_wrapper/cacti/.

    Returns (access_time_ns, area_mm2, r_cost_pJ_per_access, w_cost_pJ_per_access).
    '''
    import logging as _logging
    _logging_level = _logging.CRITICAL
    _logging_format = '%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s'
    _logging.basicConfig(level=_logging_level, format=_logging_format)

    if cacti_path is None:
        cacti_path = str(DEFAULT_CACTI_DIR)

    # 28nm is not a directly-supported CACTI tech_params node; use 32nm as anchor
    # and apply (0.9/1.0)^2 = 0.81 scaling on outputs to approximate 28nm@0.9V.
    if tech_node == 0.028:
        tech = 0.032
        scaling_factor = 0.9 * 0.9
    else:
        tech = tech_node
        scaling_factor = 1

    if mem_type == 'dram':
        mem = '"main memory"'
    elif mem_type == 'sram':
        mem = '"ram"'
    else:
        raise ValueError(f'mem_type can only be dram or sram. Now it is: {mem_type}')

    # CACTI area estimation grows nonlinearly when bw>32; clamp IO width to 32 and
    # scale the reported cost back up proportionally for consistency.
    if bw > 32:
        rows = mem_size_in_byte * 8 / bw
        line_size = int(32 / 8)
        IO_bus_width = 32
        mem_size_in_byte_adjust = rows * 32 / 8
    else:
        rows = mem_size_in_byte * 8 / bw
        line_size = int(bw / 8)
        IO_bus_width = bw
        mem_size_in_byte_adjust = mem_size_in_byte

    file_path = './self_gen'
    cwd = os.getcwd()
    # CACTI binary expects to be invoked from the directory containing its
    # `tech_params/`, `self_gen/`, etc. — try/finally so unexpected exceptions
    # (subprocess crash, parse failure, KeyboardInterrupt) cannot leave the
    # caller's cwd pointing inside the CACTI tree.
    os.chdir(cacti_path)
    try:
        os.makedirs(file_path, exist_ok=True)

        C = CactiConfig()
        C.cacti_auto(
            ['single',
             [['technology', 'cache_size', 'line_size', 'IO_bus_width', 'mem_type'],
              [tech, mem_size_in_byte_adjust, line_size, IO_bus_width, mem]]],
            f"{file_path}/cache_{hd_hash}.cfg"
        )

        try:
            with open(f'{file_path}/cache_{hd_hash}.cfg.out', 'r') as f:
                raw_result = f.readlines()
        except FileNotFoundError:
            _logging.critical(
                f'CACTI failed. [current setting] rows: {rows}, bw: {bw}, mem size (byte): {mem_size_in_byte}'
            )
            _logging.critical(
                '[CACTI minimal requirement] rows: >= 32, bw: >= 8, mem size (byte): >=64'
            )
            raise

        result = {}
        for ii, each_line in enumerate(raw_result):
            if ii == 0:
                attribute_list = each_line.split(',')
                for each_attribute in attribute_list:
                    result[each_attribute] = []
            else:
                for jj, each_value in enumerate(each_line.split(',')):
                    try:
                        result[attribute_list[jj]].append(float(each_value))
                    except Exception:
                        pass

        try:
            access_time = scaling_factor * float(result[' Access time (ns)'][-1])
            if bw > 32:
                area = scaling_factor * float(result[' Area (mm2)'][-1]) * 2 * bw / 32
                r_cost = scaling_factor * float(result[' Dynamic read energy (nJ)'][-1]) * bw / 32
                w_cost = scaling_factor * float(result[' Dynamic write energy (nJ)'][-1]) * bw / 32
            else:
                area = scaling_factor * float(result[' Area (mm2)'][-1]) * 2
                r_cost = scaling_factor * float(result[' Dynamic read energy (nJ)'][-1])
                w_cost = scaling_factor * float(result[' Dynamic write energy (nJ)'][-1])
        except KeyError:
            _logging.critical(f'**KeyError** in result, current result: {result}')
            raise
    finally:
        os.chdir(cwd)

    area = round(area, 7)
    r_cost *= 1000  # nJ → pJ
    w_cost *= 1000

    return access_time, area, r_cost, w_cost


if __name__ == '__main__':
    # Smoke test: 32 rows × 32 cols SRAM, 32-bit IO, 28nm.
    mem_size = 32 * 32 / 8  # bytes
    bw = 32
    access_time, area, r_cost, w_cost = get_cacti_cost(
        tech_node=0.028, mem_type='sram',
        mem_size_in_byte=mem_size, bw=bw, hd_hash='smoke',
    )
    print(f'access_time={access_time:.4f} ns, area={area} mm^2, '
          f'r_cost={r_cost*1000/bw:.4f} pJ/bit, w_cost={w_cost*1000/bw:.4f} pJ/bit')
