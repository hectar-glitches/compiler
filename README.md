# mlacc

A small compiler and simulator for a custom ML accelerator, built to study one
question: how much does operator fusion reduce off-chip memory traffic, and
when does that reduction actually translate into fewer cycles?

## Pipeline

```
PyTorch nn.Module
      |  torch.fx.symbolic_trace
      v
   mlacc IR (Graph of Ops)
      |  fusion pass
      v
   Fused IR (elementwise chains merged)
      |  codegen
      v
   Toy ISA (Load / Store / Compute)
      |  simulator
      v
   Cycle count + final memory state
```

Each stage is a standalone module, so any stage can be tested and reasoned
about independently of the ones around it.

## What the fusion pass does

`passes/fusion.py` walks the graph and merges consecutive elementwise ops
(`add`, `relu`, `mul`) into a single `fused_elementwise` instruction whenever
two conditions hold:

- every op in the chain is elementwise
- the tensor connecting two ops in the chain has exactly one consumer

The second condition is what makes this a correctness-preserving
transformation rather than just a memory optimization. If a tensor feeds two
different ops, folding it into one of them would silently drop the other
op's input. `matmul` is excluded from the elementwise set entirely, since it
has a different cost profile (compute-bound rather than memory-bound) and
fusing it would break the fusability analysis.

## Instruction set and cost model

The ISA has three instructions: `Load`, `Store`, and `Compute`. The
simulated hardware has one compute core, one on-chip buffer, and one
off-chip memory. This isolates the specific thing the project measures:
the off-chip memory traffic that fusion removes.

Cost model:
- `Load` / `Store`: `size_bytes / memory_bandwidth_bytes_per_cycle + fixed_memory_latency_cycles`
- `Compute`: `flops / compute_flops_per_cycle`

Fusion's effect on this model is direct: a fused chain of *N* elementwise ops
replaces *N* separate Store+Load round trips to off-chip memory with a
single Store and a single Load around the whole chain. Compute cost is
unchanged, since fusion doesn't change how much arithmetic happens, only how
often intermediate results cross the memory boundary.

## Codegen strategy

`lower_to_isa` uses the simplest possible strategy: every op's output is
stored to memory, and every op's input is loaded from memory. This keeps the
unfused case's Store+Load round trips visible and simple to reason about,
which is what makes the fused-vs-unfused comparison clean. A buffer-aware
codegen that kept values on-chip between dependent ops would collapse that
distinction and is left as a stretch goal.

## Results

_Fill in after running `main.py` on the example workload:_

| | Cycles | Off-chip Loads/Stores |
|---|---|---|
| Unfused | | |
| Fused | | |

## Relationship to prior work

The IR/fusion/ISA/codegen split mirrors the structure of TVM's Relay
(graph-level IR) and TE/TIR (schedule and low-level IR) split. The fusion
pass here implements one rule — merging linear chains of elementwise ops —
which is a special case of what TVM's `FuseOps` computes generally via
post-dominator tree analysis across multiple op categories.

This project's motivating idea — that fusion's benefit comes from cutting
memory traffic, not from doing less compute — is also the core idea behind
FlashAttention's kernel fusion and the fused kernels used in inference
engines like vLLM and SGLang.

## Repo layout

```
mlacc/
  ir.py              Op and Graph dataclasses
  passes/fusion.py   elementwise fusion pass
  isa.py             Load / Store / Compute instruction definitions
  cost_model.py       HardwareConfig and per-instruction cost functions
  simulator.py        executes instructions, tracks memory + cycles
  codegen.py          lowers a Graph to a list of instructions
  trace.py            traces a PyTorch nn.Module into a Graph
  main.py              end-to-end demo: fused vs. unfused comparison
examples/tiny_mlp.py
tests/
```

## Running

From the project root:

```
python tests/test_ir.py
python tests/test_fusion.py
python tests/test_simulator.py
python tests/test_codegen.py
python main.py
```

## Stretch goals

- A memory tiling pass, applying the same traffic-reduction idea within a
  single op rather than across a chain of ops
- Matmul + bias + relu epilogue fusion, a pattern used in real inference
  kernels
- A second `HardwareConfig` with a different compute-to-bandwidth ratio, to
  show the regime where fusion's benefit shrinks or disappears

## Background reading

- The TVM paper (the closest single reference for the IR/schedule split
  used here)
- *Programming Massively Parallel Processors* (Kirk & Hwu)
- *Computer Architecture: A Quantitative Approach*
