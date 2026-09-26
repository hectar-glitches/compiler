"""
Hardware cost model: the numbers that turn an instruction list into a
cycle estimate. This is the part that makes your speedup claim real
instead of hand-waved -- get the relative magnitudes roughly realistic
(memory bandwidth is the usual bottleneck on real accelerators, not raw
compute throughput) or your simulator will show fusion "helping" for the
wrong reason.

Suggested fields for a `HardwareConfig` dataclass:

    compute_flops_per_cycle: float
        How many floating-point ops your fake core does per cycle.

    memory_bandwidth_bytes_per_cycle: float
        Sustained off-chip bandwidth. Pick this LOW relative to compute
        throughput -- e.g. a compute-to-bandwidth ratio somewhat like a
        real accelerator's, where moving a byte costs meaningfully more
        cycles than computing on it. This ratio is exactly what makes
        fusion's memory-traffic reduction show up as a real speedup in
        your simulator rather than a rounding error.

    fixed_memory_latency_cycles: float
        A fixed per-Load/Store overhead (round-trip latency), separate
        from the bandwidth-proportional cost. Real memory systems have
        both a fixed latency and a bandwidth limit -- modeling only one
        of the two will make your results less convincing if someone
        asks about it.

    on_chip_buffer_bytes: int
        Total on-chip buffer capacity. Not used by the basic simulator,
        but needed if you attempt the memory-tiling stretch goal.

Pick specific numbers and write down WHY in a comment -- e.g. "roughly
modeled after [some real accelerator's published specs]" or "chosen so
compute:bandwidth ratio is realistic at ~10:1" -- a made-up-but-justified
number is fine; an unexamined one isn't.
"""

from dataclasses import dataclass


@dataclass
class HardwareConfig:
    pass
    # TODO: compute_flops_per_cycle, memory_bandwidth_bytes_per_cycle,
    # fixed_memory_latency_cycles, on_chip_buffer_bytes
