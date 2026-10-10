"""
End-to-end demo: trace a model, run it through both the unfused and
fused pipelines, verify identical numeric output, and report simulated
cycle counts for each.

Steps to implement:

1. Build an example model (see examples/tiny_mlp.py, which defines a couple of
   Linear+ReLU layers there) and an example input tensor.
2. `trace_model(model, example_input)` -> graph, tensor_values, tensor_shapes
3. `lower_to_isa(graph, tensor_shapes)` -> unfused_instructions
4. `fuse_elementwise(graph)` -> fused_graph
   `lower_to_isa(fused_graph, tensor_shapes)` -> fused_instructions
5. Run both instruction lists through `Simulator` with the SAME
   HardwareConfig and the SAME initial tensor_values.
6. Assert the two runs' output tensors match (within floating point
   tolerance -- `numpy.allclose`, not `==`).
7. Print both cycle counts and the speedup ratio.

This is also a good place to print instruction counts (not just cycles)
for both versions, side by side, since "fewer Load/Store instructions" is
the concrete, inspectable evidence behind the cycle-count claim.
"""

import argparse
import sys

import numpy as np
import torch

from .codegen import lower_to_isa
from .cost_model import HardwareConfig
from .passes.fusion import fuse_elementwise
from .simulator import Simulator
from .trace import trace_model
from examples.tiny_mlp import TinyMLP


def build_demo(batch_size=1024):
    torch.manual_seed(0)
    model = TinyMLP().eval()
    example_input = torch.linspace(
        -1.0, 1.0, batch_size * 8, dtype=torch.float32
    ).reshape(batch_size, 8)
    return trace_model(model, example_input)


def run_pair(graph, values, shapes, config):
    unfused = lower_to_isa(
        graph, shapes, on_chip_buffer_bytes=config.on_chip_buffer_bytes
    )
    fused = lower_to_isa(
        fuse_elementwise(graph),
        shapes,
        on_chip_buffer_bytes=config.on_chip_buffer_bytes,
    )
    unfused_memory, unfused_cycles = Simulator(config, values).run(unfused)
    fused_memory, fused_cycles = Simulator(config, values).run(fused)
    np.testing.assert_allclose(unfused_memory[graph.graph_output], fused_memory[graph.graph_output])
    return unfused, unfused_cycles, fused, fused_cycles


def print_sweep(graph, values, shapes):
    print("\nBandwidth/buffer sweep (cycles, fused speedup):")
    for bandwidth in (0.5, 1.0, 2.0, 4.0):
        for buffer_size in (64, 192, 4096):
            config = HardwareConfig(
                compute_flops_per_cycle=16.0,
                memory_bandwidth_bytes_per_cycle=bandwidth,
                fixed_memory_latency_cycles=10.0,
                on_chip_buffer_bytes=buffer_size,
            )
            _, unfused_cycles, _, fused_cycles = run_pair(graph, values, shapes, config)
            print(
                f"  bandwidth={bandwidth:>3.1f} B/cycle, buffer={buffer_size:>4} B: "
                f"{unfused_cycles:>7g} -> {fused_cycles:>7g} "
                f"({unfused_cycles / fused_cycles:.2f}x)"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the mlacc fusion simulator")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args(argv)
    if args.size <= 0:
        parser.error("--size must be positive")

    graph, values, shapes = build_demo(args.size)
    config = HardwareConfig()
    unfused, unfused_cycles, fused, fused_cycles = run_pair(
        graph, values, shapes, config
    )
    print(f"unfused: {len(unfused)} instructions, {unfused_cycles:g} cycles")
    print(f"fused:   {len(fused)} instructions, {fused_cycles:g} cycles")
    print(f"speedup: {unfused_cycles / fused_cycles:.2f}x")
    if args.sweep:
        print_sweep(graph, values, shapes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
