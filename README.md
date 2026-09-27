# MIREDO: Mixed-Integer Programming for System-Level Dataflow Optimization in SRAM-CIM Accelerators

MIREDO maps each DNN layer onto an SRAM computing-in-memory (CIM) accelerator by solving a mixed-integer program.
For every spatial mapping candidate, the program chooses the temporal tiling, the loop order, and the buffering of each operand at every memory level, minimizing latency, energy, or EDP.
The formulation models the critical path of a CIM macro whose weight loading and computation are mutually exclusive, or overlapped on a ping-pong-capable macro.
A cycle-level simulator evaluates each selected mapping.

## Installation

1. Obtain a [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/).
2. Create the environment:

```bash
conda env create -f environment.yml
conda activate MIREDO
```

## Usage

```bash
python run.py -m resnet18 -opt EDP
```

`-m <name>` reads `model/<name>.onnx`.
Convolution layers are mapped by default; `--matmul` also maps MatMul and Gemm layers.
The scripts in `model/` export the Transformer workloads, for example:

```bash
python model/BERT_Base.py
python run.py -m bert_base -a transformer --matmul -opt EDP
```

`model/GPT2_Medium_block.py` and `model/TinyLlama_block.py` export one decoder block with a 1024-token prompt, and `model/Decode_blocks.py` exports single-token decode steps of the same blocks (for example `gpt2_medium_decode_b1_t1024`).

| Option | Meaning |
| --- | --- |
| `-m` | Workload name, `model/<name>.onnx` |
| `-a` | Hardware template: `default` or `transformer` |
| `--matmul` | Also map MatMul and Gemm layers |
| `-opt` | Objective: `Latency`, `Energy`, or `EDP` (default) |
| `-t` | Time limit in seconds for each MIP solve, one per spatial candidate (default 60) |
| `-o` | Output folder under `output/` |

Each run writes a folder under `output/` with one subfolder per layer.
The log reports, for every layer and for the whole model, the simulated latency, energy, and EDP of the selected mapping, together with the MIP's prediction error against the simulator.
The whole-model EDP is the product of total latency and total energy.

## Hardware

`Architecture/templates/default.py` describes an 8-core digital SRAM-CIM accelerator at 28 nm: a 32×128 macro array per core with a storage-to-compute ratio of 8:1, 8 KB input and 16 KB output buffers per core, a 256 KB global buffer, and 1 GB of DRAM at 64 bit/cycle.
`Architecture/templates/transformer.py` scales it to 16 cores with a 64×256 macro array, 16 KB and 32 KB per-core buffers, a 4 MB global buffer, and 4 GB of DRAM at 256 bit/cycle.
Both templates store precomputed energies, so they run without building CACTI.

`Architecture/HardwareVariants.py` derives variants of a template through `build_hardware_variant(spec, parameter, value)`, with the parameters `core_count`, `buffer_capacity`, `gbuf_core_bw`, `dram_bw`, `operand_scratchpad`, `compartment_depth`, `macro_spec`, and `macro_weight_double_buffer`.
The last one turns the macro into a ping-pong-capable weight store that loads one bank while computing on the other, at unchanged total capacity.
Variants recompute memory energies with CACTI, which must be built first:

```bash
cd utils/Cacti_wrapper/cacti && make
```

## Code Structure

| Path | Content |
| --- | --- |
| `run.py` | Command-line entry |
| `SolveMapping.py` | Spatial-candidate enumeration, lower-bound pruning, and parallel MIP solves |
| `utils/SolverTSS.py` | MIP formulation |
| `Simulator/Simulax.py` | Cycle-level simulator |
| `Architecture/` | Hardware specification, templates, and variants |
| `utils/Cacti_wrapper/` | CACTI interface for memory energy |
| `model/` | Workload definitions and ONNX exporters |
