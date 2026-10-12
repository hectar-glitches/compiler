"""
Instruction definitions for the modeled accelerator memory and compute
pipeline.


- `Load(tensor_name, size_bytes)` — copy a tensor from off-chip memory
  into the on-chip buffer.
- `Store(tensor_name, size_bytes)` — write a tensor from the on-chip
  buffer back to off-chip memory.
- `Compute(op_type, input_names, output_name, output_size_bytes, flops, sub_ops=None)`
  — execute one operation using operands already in the on-chip buffer,
  producing a result in the on-chip buffer.

"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Load:
    tensor_name: str
    size_bytes: int

@dataclass
class Store:
    tensor_name: str
    size_bytes: int


@dataclass
class Compute:
    op_type: str
    input_names: list[str]
    output_name: str
    output_size_bytes: int
    flops: int
    sub_ops: list[str] | None = None