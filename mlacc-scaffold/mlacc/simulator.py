"""
Simulator: executes a list of ISA instructions two ways at once --
functionally (real numpy arrays, so you can verify correctness) and as a
cycle-cost estimate (using the HardwareConfig cost model).

Why both: a cycle count with no correctness check is easy to get
"impressively" wrong. Verifying the fused and unfused pipelines produce
bit-identical numeric output is what makes the speedup claim trustworthy
-- you're not measuring two different computations, you're measuring two
schedules of the same one.

Functions/classes to implement:

- `Simulator` class:
    - holds an on-chip buffer, modeled as a dict[str, np.ndarray]
    - holds a reference to "off-chip memory" -- for this toy project, the
      graph's initial input tensors and any weights, also just a dict
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
        Return the final off-chip memory contents (or just the graph's
        output tensor) and the total cycle count.

Think about where Compute actually gets the numpy function to run. One
approach: a small dict mapping op_type strings ("add", "relu", "matmul")
to numpy-based implementations, that both codegen.py and simulator.py can
share, so "what does the op X actually compute" is defined in exactly one
place. For a fused Compute instruction (multiple op_types), apply them in
sequence to the same buffer.
"""

from .cost_model import HardwareConfig  # noqa: F401


class Simulator:
    def __init__(self, hardware_config: HardwareConfig, initial_memory: dict):
        raise NotImplementedError("implement me")

    def run(self, instructions: list):
        raise NotImplementedError("implement me")
