"""
Codegen: lower a Graph (fused or unfused -- this function shouldn't care
which) into a flat list of ISA instructions (Load / Store / Compute).

This is where fusion's benefit actually gets realized as fewer
instructions, not just a smaller op count in the IR. The key logic:

For each Op in the graph, in order:
    1. For each of its inputs NOT already resident in your on-chip
       buffer model (track this as you emit instructions), emit a Load.
    2. Emit one Compute instruction for the op (a fused Op with multiple
       sub-operations still emits exactly ONE Compute instruction -- that
       collapsing is the whole point).
    3. Decide whether to emit a Store for the op's output immediately, or
       leave it in the on-chip buffer because the very next op consumes
       it directly. This decision is exactly what differs between the
       fused and unfused pipelines fed the SAME graph shape: in the
       unfused version, add's output gets Stored and then immediately
       re-Loaded for relu, because they're separate ops with no
       information that they're adjacent; in the fused version, there's
       only one Compute instruction, so that intermediate Store+Load
       pair never exists at all.

A reasonable simplification for a first pass: Store every op's output
immediately after computing it (as if the on-chip buffer were tiny), and
Load whatever's needed before each op. This alone is enough to show
fusion's benefit, since a fused op simply has fewer op boundaries and
therefore fewer Store+Load pairs. A more realistic codegen would track
buffer occupancy and only Store when the buffer needs the space or when a
later, non-adjacent op needs the value again -- worth attempting after
the simple version works and is tested.

Function to implement:

- `lower_to_isa(graph, tensor_shapes: dict[str, tuple], dtype_size: int = 4) -> list`
    Returns the flat instruction list. `tensor_shapes` gives you the
    shape of every named tensor in the graph (inputs, intermediates, and
    output) so you can compute size_bytes and flops for each instruction.
"""

from .ir import Graph  # noqa: F401


def lower_to_isa(graph, tensor_shapes, dtype_size: int = 4):
    raise NotImplementedError("implement me")
