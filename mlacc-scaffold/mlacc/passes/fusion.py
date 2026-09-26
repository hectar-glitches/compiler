"""
Elementwise operator fusion pass.

Input: a Graph with individual ops (e.g. matmul -> add -> relu, as three
separate Ops each with their own output tensor).

Output: a new Graph where consecutive, single-consumer, elementwise ops
are merged into one "fused" Op. E.g. `add -> relu` (both elementwise,
add's output consumed only by relu) becomes a single Op with
op_type="fused_elementwise" and some way of recording the sub-operations
it performs (a list, e.g. ["add", "relu"], stored whichever way makes
sense given your Op design -- you may need to extend Op with an optional
field for this, which is a reasonable and expected change to make here).

Algorithm sketch (yours to implement, this is not the only valid approach):

1. Walk the graph's ops in order, tracking how many times each tensor
   name is consumed as an input across the WHOLE graph (build this count
   first, in one pass, before deciding anything about fusion).
2. Walk again, greedily building fusion chains: start a new chain at any
   elementwise op. Extend the chain forward as long as the next op is
   also elementwise AND its only input is the current chain's output AND
   that output has exactly one consumer (this op) in the whole graph.

   Note: this IR-level `fused_ops` list (e.g. `["add", "relu"]` on the
   merged Op) is a distinct concept from the ISA-level `sub_ops` field on
   a `Compute` instruction in isa.py -- codegen.py is what translates one
   into the other. Keep them as separate fields with separate names; don't
   try to reuse one for both.
3. Stop a chain (and emit it as one fused Op, or pass through a
   single un-fused op unchanged if the chain length is 1) when you hit a
   non-elementwise op, a multi-consumer tensor, or the end of the graph.
4. matmul (and anything else not in ELEMENTWISE_OPS) always passes
   through unchanged -- never merged into a chain.

Functions to implement:

- `fuse_elementwise(graph: Graph) -> Graph` — the main entry point.

Test this against hand-built Graph objects representing exactly the two
tricky cases called out in README.md: a multi-consumer producer, and a
matmul that should never get absorbed into a chain.
"""

from ..ir import Graph  # noqa: F401 (mlacc.passes -> mlacc.ir; this is correct
                          # as written for the mlacc/passes/fusion.py layout)


def fuse_elementwise(graph):
    raise NotImplementedError("implement me")
