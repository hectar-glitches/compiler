"""
Hardware cost model: the numbers that turn an instruction list into a
cycle estimate.

`HardwareConfig` defines the parameters used by the cycle model:

    compute_flops_per_cycle: float
        How many floating-point ops the modeled core does per cycle.

    memory_bandwidth_bytes_per_cycle: float
        Sustained off-chip bandwidth.

    fixed_memory_latency_cycles: float
        A fixed per-Load/Store overhead (round-trip latency), separate
        from the bandwidth-proportional cost.

    on_chip_buffer_bytes: int
        Total on-chip buffer capacity used by buffer-aware scheduling and
        future tiled execution.

The default values model a bandwidth-constrained accelerator: moving a byte
costs substantially more than one floating-point operation.
"""

from dataclasses import dataclass


@dataclass
class HardwareConfig:
    compute_flops_per_cycle: float = 16.0
    memory_bandwidth_bytes_per_cycle: float = 1.0
    fixed_memory_latency_cycles: float = 10.0
    on_chip_buffer_bytes: int = 4096

    def __post_init__(self):
        if self.compute_flops_per_cycle <= 0:
            raise ValueError("compute_flops_per_cycle must be positive")
        if self.memory_bandwidth_bytes_per_cycle <= 0:
            raise ValueError("memory_bandwidth_bytes_per_cycle must be positive")
        if self.fixed_memory_latency_cycles < 0:
            raise ValueError("fixed_memory_latency_cycles must be non-negative")
        if self.on_chip_buffer_bytes <= 0:
            raise ValueError("on_chip_buffer_bytes must be positive")
