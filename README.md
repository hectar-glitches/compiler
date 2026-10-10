# mlacc — an ML compiler for an accelerator

This project traces a model graph, applies compiler transformations, lowers
the result to a Load/Store/Compute instruction set, and evaluates execution
with functional simulation and a hardware cost model. Its structure follows
the compiler stages used by ML accelerator software, including graph
rewriting, code generation, buffer management, and performance analysis.

## The pipeline

```
PyTorch nn.Module
      │  torch.fx.symbolic_trace
      ▼
  fx.Graph
      │  mlacc/trace.py: convert fx ops -> our own IR
      ▼
  mlacc IR (a flat list of typed Ops: matmul, add, relu, ...)
      │  mlacc/passes/fusion.py: fuse elementwise op chains
      ▼
  Fused IR (fewer, larger ops)
      │  mlacc/codegen.py: lower each Op to -ISA instructions
      ▼
  Instruction list (LOAD / STORE / COMPUTE, targeting an on-chip buffer)
      │  mlacc/simulator.py: execute against a cost model
      ▼
  Actual output values + simulated cycle count
```

You build and test this bottom-to-top-of-dependency, not in pipeline
order: IR first (nothing depends on torch), then the fusion pass (depends
only on IR), then the cost model + simulator (depends only on the ISA),
then codegen (glues IR -> ISA), and only at the very end the torch.fx
tracing front-end. This mirrors real compiler engineering: you can build
and unit-test middle/back-end stages against hand-constructed IR long
before the front-end (parsing a real source language, or here, tracing a
real PyTorch graph) exists.

## Why this shape, specifically

The entire point of this project is to make a real claim you can defend in
an interview: **fusing elementwise operations reduces off-chip memory
traffic, and that's most of why kernel fusion matters on real
accelerators** — not because fused code has fewer instructions in some
abstract sense, but because every unfused op boundary is a STORE to
memory followed by a LOAD back in, and off-chip memory bandwidth is
usually the actual bottleneck on ML accelerators, not compute throughput.
Your cost model and simulator need to make this trade-off real, not just
asserted in a README.

## Pipeline components

1. **`mlacc/ir.py`** — define `Op` and `Graph`. No dependencies. Run
   `tests/test_ir.py`.
2. **`mlacc/passes/fusion.py`** — implement the elementwise-fusion pass
   over hand-built `Graph` objects (no torch needed for this stage at
   all). Run `tests/test_fusion.py`. This is the stage with the most
   design constraints described in the "Fusion correctness" section.
3. **`mlacc/isa.py`** and **`mlacc/cost_model.py`** — define your 
   instruction set and hardware parameters.
4. **`mlacc/simulator.py`** — execute a hand-built instruction list
   against the cost model, both for correctness (real numpy arrays) and
   for a cycle-count estimate. Run `tests/test_simulator.py`.
5. **`mlacc/codegen.py`** — lower an IR `Graph` (fused or not) into a
   list of ISA instructions, managing what's resident in the on-chip
   buffer vs. what needs a LOAD/STORE. Run `tests/test_codegen.py` — this
   is the test that actually proves fusion reduces memory traffic.
6. **`mlacc/trace.py`** — use `torch.fx.symbolic_trace` on a real
   `nn.Module` and convert its graph into the project IR.
7. **`mlacc/main.py`** — CLI that runs a model through the whole
   pipeline twice (fused and unfused), confirms both produce the same
   numeric output, and reports the simulated cycle counts side by side.

## Fusion correctness

Fusion observes two constraints:

- **Multi-consumer producers.** If op A's output feeds both op B and some
  other op C, you generally can't fuse A into B alone — B would still
  need A's result available separately for C. A real compiler either
  refuses to fuse this case rather than duplicating the computation.
- **Non-elementwise ops break the chain.** `matmul` is not elementwise
  (each output element depends on many input elements, not just the
  corresponding ones) — `matmul` remains a boundary between elementwise
  chains.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running tests as you go

```bash
python tests/test_ir.py
python tests/test_fusion.py
python tests/test_simulator.py
python tests/test_codegen.py
```

The simulator and code generator are NumPy-only, while the end-to-end CLI
uses the PyTorch FX front end:

```bash
python -m mlacc.main --size 1024 --sweep
```

The command prints the unfused and fused cycle counts, verifies that both
schedules produce the same output, and sweeps sustained memory bandwidth and
buffer capacity. The current buffer model spills an output when the resident
tensors exceed capacity; tiled execution remains a separate stretch goal.

The cycle estimator also has a C++/pybind11 implementation. Build it with:

```bash
python setup.py build_ext --inplace
```

`mlacc.native.estimate_cycles` accepts the same ISA instructions and hardware
configuration as the Python simulator. The Python implementation remains the
functional reference; the native backend accelerates cycle estimation.

The native matmul ladder can be benchmarked with:

```bash
python -m mlacc.matmul_benchmark --m 256 --n 256 --k 256
```

It checks each implementation against NumPy before reporting timings for
naive, tiled, and NEON matmul.

Generate the roofline plot with:

```bash
python -m mlacc.roofline --output roofline.png \
  --benchmark-output matmul_benchmark.png
```

The command generates two presentation plots from the same benchmark:

![mlacc roofline plot](roofline.png)

The roofline shows modeled `add`, `relu`, and `matmul` points, the compute and
bandwidth ceilings, the ridge point, and measured native matmul points.

![C++ matmul benchmark](matmul_benchmark.png)

The benchmark chart shows execution time and achieved GFLOP/s for the C++
matmul ladder. Results are local measurements on Apple Silicon; NumPy uses
the platform's optimized Accelerate/BLAS implementation and is intentionally
not included in the C++-only comparison chart.

The unit tests build `Graph`/instruction objects directly. The FX front end
and end-to-end CLI require PyTorch.

## Implemented artifacts

- PyTorch FX tracing for `Linear` and `ReLU` models.
- Elementwise fusion with multi-consumer and non-elementwise safety checks.
- Load/Store/Compute code generation with buffer-aware spilling.
- Functional NumPy simulation and configurable cycle modeling.
- A C++ cycle estimator exposed through pybind11.
- Naive, tiled, and NEON matmul kernels with NumPy correctness checks.
- A roofline model and benchmark visualizations in `roofline.png` and
  `matmul_benchmark.png`.

## Current boundaries

- The simulator test suite has one pre-existing semantic mismatch: one test
  describes `relu(x - 10)` but emits `relu(x)`.
- Buffer-aware spilling is implemented; tiled execution is not.
- The modeled roofline uses explicit assumptions from `HardwareConfig`; it is
  not a measurement of the host CPU's peak performance.
