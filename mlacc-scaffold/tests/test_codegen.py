"""
Tests codegen in isolation from the fusion pass -- we hand-build both an
"unfused-shaped" and a "fused-shaped" Graph directly, so a bug in
fusion.py can't hide a bug in codegen.py or vice versa.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mlacc.codegen import lower_to_isa
from mlacc.ir import Graph, Op
from mlacc.isa import Compute, Load, Store


def test_unfused_graph_has_store_load_roundtrip():
    ops = [
        Op(op_type="add", inputs=["x", "b"], output="h1", shape=(4,)),
        Op(op_type="relu", inputs=["h1"], output="h2", shape=(4,)),
    ]
    graph = Graph(ops=ops, graph_inputs=["x", "b"], graph_output="h2")
    shapes = {"x": (4,), "b": (4,), "h1": (4,), "h2": (4,)}

    instructions = lower_to_isa(graph, shapes)

    stores = [i for i in instructions if isinstance(i, Store)]
    loads = [i for i in instructions if isinstance(i, Load)]
    computes = [i for i in instructions if isinstance(i, Compute)]

    assert len(computes) == 2, "expected one Compute per op"
    # The unfused version should Store h1 and then Load it again for relu.
    assert any(s.tensor_name == "h1" for s in stores), (
        "expected h1 to be Stored between the two ops"
    )
    assert any(l.tensor_name == "h1" for l in loads), (
        "expected h1 to be re-Loaded before relu"
    )


def test_fused_graph_skips_intermediate_roundtrip():
    fused_op = Op(
        op_type="fused_elementwise", inputs=["x", "b"], output="h2",
        shape=(4,),
    )
    fused_op.fused_ops = ["add", "relu"]  # set post-construction is fine
                                            # if your Op doesn't declare
                                            # this field by default

    graph = Graph(ops=[fused_op], graph_inputs=["x", "b"], graph_output="h2")
    shapes = {"x": (4,), "b": (4,), "h2": (4,)}

    instructions = lower_to_isa(graph, shapes)

    computes = [i for i in instructions if isinstance(i, Compute)]
    stores = [i for i in instructions if isinstance(i, Store)]

    assert len(computes) == 1, "fused op should emit exactly one Compute"
    # Crucially: h1 never exists as a named tensor here, so there's no
    # intermediate Store/Load for it at all -- that's the whole point.
    assert not any(s.tensor_name == "h1" for s in stores)


def test_fused_produces_fewer_instructions_than_unfused_for_same_computation():
    # This is the test that actually proves fusion's benefit at the
    # instruction level, holding the computation constant.
    unfused_ops = [
        Op(op_type="add", inputs=["x", "b"], output="h1", shape=(4,)),
        Op(op_type="relu", inputs=["h1"], output="h2", shape=(4,)),
    ]
    unfused_graph = Graph(
        ops=unfused_ops, graph_inputs=["x", "b"], graph_output="h2"
    )

    fused_op = Op(
        op_type="fused_elementwise", inputs=["x", "b"], output="h2",
        shape=(4,),
    )
    fused_op.fused_ops = ["add", "relu"]
    fused_graph = Graph(
        ops=[fused_op], graph_inputs=["x", "b"], graph_output="h2"
    )

    shapes = {"x": (4,), "b": (4,), "h1": (4,), "h2": (4,)}

    unfused_instrs = lower_to_isa(unfused_graph, shapes)
    fused_instrs = lower_to_isa(fused_graph, shapes)

    assert len(fused_instrs) < len(unfused_instrs), (
        f"expected fused ({len(fused_instrs)} instrs) < "
        f"unfused ({len(unfused_instrs)} instrs)"
    )


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
