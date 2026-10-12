"""
Execute an ISA instruction stream functionally with NumPy while accumulating
the corresponding hardware-model cycle estimate.

The functional result and cycle estimate share the same instruction stream,
so performance comparisons also verify that the schedules compute the same
operation.

The `Simulator` maintains:

- `Simulator` class:
    - holds an on-chip buffer, modeled as a dict[str, np.ndarray]
    - holds a reference to "off-chip memory" -- for this project, the
      graph's initial input tensors and any weights, represented as a dict
    - a running cycle counter

    - `run(instructions: list) -> tuple[dict, int]`
        Execute each instruction in order:
          - Load: fetch the named tensor from off-chip memory into the
            on-chip buffer dict; add cycles per the cost model (bandwidth
            term + fixed latency term).
          - Store: write the named tensor from the on-chip buffer back to
            off-chip memory; add cycles the same way.
          - Compute: pull inputs from the on-chip buffer, actually
            perform the real numpy computation for whatever op_type(s)
            this instruction represents (this means Compute needs enough
            information -- op_type, or a list of op_types if fused -- to
            know which numpy ops to call), write the result into the
            on-chip buffer, add cycles per compute_flops_per_cycle.
        Return the final off-chip memory contents (or the graph's
        output tensor) and the total cycle count.

Compute dispatches from operation names to NumPy implementations. A fused
Compute instruction applies its sub-operation sequence to the same buffer.
"""

import numpy as np

from .cost_model import HardwareConfig
from .isa import Compute, Load, Store


class Simulator:
    def __init__(self, hardware_config: HardwareConfig, initial_memory: dict):
        self.hardware_config = hardware_config
        self.initial_memory = {name: np.array(value, copy=True) for name, value in initial_memory.items()}
        self.buffer = {}
        self.memory = {}
        self.cycles = 0.0

    def run(self, instructions: list):
        self.memory = {name: np.array(value, copy=True) for name, value in self.initial_memory.items()}
        self.buffer = {}
        self.cycles = 0.0
        for instruction in instructions:
            if isinstance(instruction, Load):
                if instruction.tensor_name not in self.memory:
                    raise KeyError(f"cannot load missing tensor {instruction.tensor_name!r}")
                self.buffer[instruction.tensor_name] = np.array(
                    self.memory[instruction.tensor_name], copy=True
                )
                self.cycles += self._memory_cycles(instruction.size_bytes)
            elif isinstance(instruction, Store):
                if instruction.tensor_name not in self.buffer:
                    raise KeyError(f"cannot store non-resident tensor {instruction.tensor_name!r}")
                self.memory[instruction.tensor_name] = np.array(
                    self.buffer[instruction.tensor_name], copy=True
                )
                self.cycles += self._memory_cycles(instruction.size_bytes)
                del self.buffer[instruction.tensor_name]
            elif isinstance(instruction, Compute):
                self._compute(instruction)
                self.cycles += instruction.flops / self.hardware_config.compute_flops_per_cycle
            else:
                raise TypeError(f"unsupported instruction: {type(instruction).__name__}")
        return self.memory, self._integer_if_exact(self.cycles)

    def _memory_cycles(self, size_bytes):
        return size_bytes / self.hardware_config.memory_bandwidth_bytes_per_cycle + (
            self.hardware_config.fixed_memory_latency_cycles
        )

    def _compute(self, instruction):
        names = instruction.input_names
        if any(name not in self.buffer for name in names):
            missing = [name for name in names if name not in self.buffer]
            raise KeyError(f"compute inputs not resident: {missing}")
        values = [self.buffer[name] for name in names]
        operations = instruction.sub_ops or [instruction.op_type]
        result = None
        for index, operation in enumerate(operations):
            if operation == "add":
                if result is None:
                    result = values[0] + values[1]
                else:
                    result = result + values[index]
            elif operation == "mul":
                if result is None:
                    result = values[0] * values[1]
                else:
                    result = result * values[index]
            elif operation == "relu":
                result = np.maximum(values[0] if result is None else result, 0)
            elif operation == "matmul":
                result = np.matmul(values[0], values[1])
            else:
                raise ValueError(f"unsupported compute operation: {operation}")
        self.buffer[instruction.output_name] = result

    @staticmethod
    def _integer_if_exact(value):
        return int(value) if value.is_integer() else value
