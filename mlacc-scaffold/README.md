# mlacc — a tiny ML compiler for a toy accelerator

This is a scaffold, not a working project. It's the structure, the design
of the pipeline, and real tests you make pass by implementing the actual
logic. This is deliberately a step up from a generic "toy LLVM compiler"
project: it's shaped like what Annapurna Labs' Neuron SDK actually does —
take a model graph and compile it down to run efficiently on fixed,
custom hardware — not a generic teaching exercise.

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
      │  mlacc/codegen.py: lower each Op to toy-ISA instructions
      ▼
  Instruction list (LOAD / STORE / COMPUTE, targeting a fake on-chip buffer)
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

## Suggested build order

1. **`mlacc/ir.py`** — define `Op` and `Graph`. No dependencies. Run
   `tests/test_ir.py`.
2. **`mlacc/passes/fusion.py`** — implement the elementwise-fusion pass
   over hand-built `Graph` objects (no torch needed for this stage at
   all). Run `tests/test_fusion.py`. This is the stage with the most
   interesting design decisions — see the "fusion correctness" section
   below before you start.
3. **`mlacc/isa.py`** and **`mlacc/cost_model.py`** — define your toy
   instruction set and hardware parameters.
4. **`mlacc/simulator.py`** — execute a hand-built instruction list
   against the cost model, both for correctness (real numpy arrays) and
   for a cycle-count estimate. Run `tests/test_simulator.py`.
5. **`mlacc/codegen.py`** — lower an IR `Graph` (fused or not) into a
   list of ISA instructions, managing what's resident in the on-chip
   buffer vs. what needs a LOAD/STORE. Run `tests/test_codegen.py` — this
   is the test that actually proves fusion reduces memory traffic.
6. **`mlacc/trace.py`** — only now, wire up `torch.fx.symbolic_trace` on
   a real `nn.Module` and convert its graph into your IR. This is the
   only stage that needs `pip install torch`.
7. **`mlacc/main.py`** — CLI that runs a small model through the whole
   pipeline twice (fused and unfused), confirms both produce the same
   numeric output, and reports the simulated cycle counts side by side.

## Fusion correctness — read this before writing `fusion.py`

The naive version of "fuse adjacent elementwise ops" is wrong in at least
two ways that your tests will check for:

- **Multi-consumer producers.** If op A's output feeds both op B and some
  other op C, you generally can't fuse A into B alone — B would still
  need A's result available separately for C. A real compiler either
  refuses to fuse this case or duplicates the computation; for this
  project, refusing to fuse (and explaining why in a comment) is the
  right scope.
- **Non-elementwise ops break the chain.** `matmul` is not elementwise
  (each output element depends on many input elements, not just the
  corresponding ones) — you cannot fuse a `matmul` into an elementwise
  chain the same way you fuse `add` into `relu`. Get this distinction
  right in your `Op` design (e.g. a `is_elementwise` flag or a fixed set)
  before you start the fusion pass itself.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch numpy   # torch only needed once you reach trace.py
```

## Running tests as you go

```bash
python tests/test_ir.py
python tests/test_fusion.py
python tests/test_simulator.py
python tests/test_codegen.py
```

These don't need torch at all — they build `Graph`/instruction objects by
hand. Only `mlacc/trace.py` and the final `main.py` demo need torch
installed.

## What "done" looks like

- All four test files pass.
- `python -m mlacc.main` traces a small model (a couple of Linear+ReLU
  layers — sketch one yourself in `examples/`), runs it through both the
  fused and unfused pipelines, confirms the numeric outputs match, and
  prints a simulated cycle count for each — with the fused version
  showing measurably fewer cycles because of reduced off-chip traffic.
- You can explain, without looking at your code: why matmul can't be
  fused the same way as elementwise ops, why multi-consumer producers
  block fusion, and specifically which cost-model parameter (compute
  throughput vs. memory bandwidth) is actually responsible for the
  speedup you measure. If your speedup comes from the wrong parameter,
  your cost model doesn't reflect reality yet — go fix it, don't just
  report the number.

## Stretch goals (only after the core pipeline works)

- A basic memory-tiling pass: if an intermediate tensor is too big for
  your fake on-chip buffer, split the computation into tiles that fit,
  and account for the extra LOAD/STORE traffic tiling introduces.
- Fuse a `matmul` with a following `add` (bias) and `relu` into one
  "epilogue-fused" instruction — this is exactly what real ML compilers
  do (bias+activation fused into a GEMM epilogue), and is a natural next
  step once plain elementwise fusion works.
- Model a second hardware config (e.g. lower memory bandwidth, more
  compute units) and show the *same* fusion pass gives a different
  relative speedup — proving you understand fusion's benefit is
  bandwidth-bound, not free.
