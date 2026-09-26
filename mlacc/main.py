"""
End-to-end demo: trace a small model, run it through both the unfused and
fused pipelines, verify identical numeric output, and report simulated
cycle counts for each.

Steps to implement:

1. Build a small model (see examples/tiny_mlp.py -- sketch a couple of
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

import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("implement me")


if __name__ == "__main__":
    sys.exit(main())
