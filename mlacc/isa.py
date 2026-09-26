"""
A toy accelerator instruction set. Deliberately minimal: this models a
single compute core with one small on-chip buffer and one off-chip memory,
because the entire point of this project is measuring the off-chip
memory traffic that fusion avoids -- not building a realistic ISA.

Instructions to define (as a small set of dataclasses or a tagged union,
your call):

- `Load(tensor_name, size_bytes)` — bring a tensor from off-chip memory
  into the on-chip buffer.
- `Store(tensor_name, size_bytes)` — write a tensor from the on-chip
  buffer back to off-chip memory.
- `Compute(op_type, input_names, output_name, output_size_bytes, flops, sub_ops=None)`
  — perform one operation using operands already in the on-chip buffer,
  producing a result in the on-chip buffer. For a single, non-fused op,
  `op_type` is e.g. "add" or "relu" and `sub_ops` is None. For a FUSED
  instruction (multiple ops collapsed into one Compute), set
  `op_type="fused"` and `sub_ops=["add", "relu"]` (in execution order) --
  this lets the simulator apply each sub-op in sequence to the running
  intermediate result, using the same per-op-type numpy implementations
  it already has for the non-fused case, rather than needing a separate
  hardcoded implementation for every possible fusion combination.
  `flops` is however you choose to estimate the compute cost (e.g. for
  elementwise ops, roughly output_size / dtype_size per sub-op, summed;
  for matmul, 2 * M * K * N for an (M,K)x(K,N) multiply).

You will also want a way to represent "this value must be computed
fresh vs. this value is already resident in the on-chip buffer" -- that
distinction is what codegen.py needs to decide when a Load is actually
necessary.
"""

from dataclasses import dataclass


@dataclass
class Load:
    pass


@dataclass
class Store:
    pass


@dataclass
class Compute:
    pass
