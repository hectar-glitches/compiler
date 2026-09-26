"""
These tests assume the Op/Graph field names used in ir.py's hints
(op_type, inputs, output, shape / ops, graph_inputs, graph_output), and
assume a fused Op ends up with op_type == "fused_elementwise" with the
sub-operations recorded somewhere on the Op (the tests below check for an
attribute called `fused_ops` -- a list like ["add", "relu"] -- adjust if
you named it differently, but be deliberate about the rename).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mlacc.ir import Graph, Op
from mlacc.passes.fusion import fuse_elementwise


def test_fuses_simple_chain():
    # matmul -> add -> relu: add and relu should fuse; matmul stays alone.
    ops = [
        Op(op_type="matmul", inputs=["x", "w"], output="h1", shape=(4,)),
        Op(op_type="add", inputs=["h1", "b"], output="h2", shape=(4,)),
        Op(op_type="relu", inputs=["h2"], output="h3", shape=(4,)),
    ]
    graph = Graph(ops=ops, graph_inputs=["x", "w", "b"], graph_output="h3")
    fused = fuse_elementwise(graph)

    op_types = [op.op_type for op in fused.ops]
    assert op_types == ["matmul", "fused_elementwise"], op_types

    fused_op = fused.ops[1]
    assert fused_op.fused_ops == ["add", "relu"]
    assert fused_op.output == "h3"


def test_does_not_fuse_matmul_into_chain():
    # add -> matmul -> relu: matmul breaks the chain; nothing should fuse
    # since add and relu are no longer adjacent to each other.
    ops = [
        Op(op_type="add", inputs=["x", "b"], output="h1", shape=(4,)),
        Op(op_type="matmul", inputs=["h1", "w"], output="h2", shape=(4,)),
        Op(op_type="relu", inputs=["h2"], output="h3", shape=(4,)),
    ]
    graph = Graph(ops=ops, graph_inputs=["x", "b", "w"], graph_output="h3")
    fused = fuse_elementwise(graph)

    op_types = [op.op_type for op in fused.ops]
    assert op_types == ["add", "matmul", "relu"], op_types


def test_multi_consumer_blocks_fusion():
    # h1 is consumed by BOTH relu (h2) and a second op (h3) -- add must
    # NOT be fused into relu, because h1's value is needed independently
    # by the second consumer.
    ops = [
        Op(op_type="add", inputs=["x", "b"], output="h1", shape=(4,)),
        Op(op_type="relu", inputs=["h1"], output="h2", shape=(4,)),
        Op(op_type="mul", inputs=["h1", "x"], output="h3", shape=(4,)),
    ]
    graph = Graph(
        ops=ops, graph_inputs=["x", "b"], graph_output="h3"
    )
    fused = fuse_elementwise(graph)

    op_types = [op.op_type for op in fused.ops]
    assert "add" in op_types, (
        "add should remain un-fused since h1 has two consumers: "
        f"got {op_types}"
    )


def test_single_op_chain_passes_through_unchanged():
    # A lone elementwise op with no fusable neighbor should still work
    # (a "chain" of length 1 is just the original op, not wrapped).
    ops = [Op(op_type="relu", inputs=["x"], output="y", shape=(4,))]
    graph = Graph(ops=ops, graph_inputs=["x"], graph_output="y")
    fused = fuse_elementwise(graph)
    assert [op.op_type for op in fused.ops] == ["relu"]


def test_two_separate_chains_both_fuse():
    # matmul -> add -> relu -> matmul -> add -> relu: two independent
    # fusable chains separated by a matmul in the middle.
    ops = [
        Op(op_type="matmul", inputs=["x", "w1"], output="a1", shape=(4,)),
        Op(op_type="add", inputs=["a1", "b1"], output="a2", shape=(4,)),
        Op(op_type="relu", inputs=["a2"], output="a3", shape=(4,)),
        Op(op_type="matmul", inputs=["a3", "w2"], output="b1_", shape=(4,)),
        Op(op_type="add", inputs=["b1_", "b2"], output="b2_", shape=(4,)),
        Op(op_type="relu", inputs=["b2_"], output="out", shape=(4,)),
    ]
    graph = Graph(
        ops=ops,
        graph_inputs=["x", "w1", "b1", "w2", "b2"],
        graph_output="out",
    )
    fused = fuse_elementwise(graph)
    op_types = [op.op_type for op in fused.ops]
    assert op_types == [
        "matmul", "fused_elementwise", "matmul", "fused_elementwise",
    ], op_types


if __name__ == "__main__":
    import inspect

    tests = [
        (name, fn) for name, fn in list(globals().items())
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
