# MIREDO: Mixed-Integer Programming for System-Level Dataflow Optimization in SRAM-CIM Accelerators

MIREDO is a system-level dataflow optimization framework for SRAM-based Computing-in-Memory (CIM) accelerators.
It jointly optimizes tiling, ordering, and buffering via Mixed-Integer Programming (MIP) to minimize latency, energy, or EDP.

## Installation

1. Obtain a [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/).

2. Set up the environment:
```bash
conda env create -f environment.yml
conda activate MIREDO
```

## Quick Start

```bash
python run.py -m resnet18 -opt Latency
```

Supported objectives: `Latency`, `Energy`, `EDP`. Run `python run.py -h` for all options.

## Hardware Configuration

The default template (`Architecture/templates/default.py`) defines an 8-core digital SRAM CIM accelerator at 28nm with pre-computed energy parameters.

To customize hardware parameters (core count, buffer sizes, macro dimensions, etc.), use `Architecture/HardwareVariants.py`. This requires building CACTI:

```bash
cd utils/Cacti_wrapper/cacti && make
```
