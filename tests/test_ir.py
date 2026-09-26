import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mlacc.ir import ELEMENTWISE_OPS, Graph, Op


def test_op_basic_fields():
    op = Op(op_type="add", inputs=["a", "b"], output="c", shape=(4,))
    assert op.op_type == "add"
    assert op.inputs == ["a", "b"]
    assert op.output == "c"
    assert op.shape == (4,)


def test_matmul_not_elementwise():
    assert "matmul" not in ELEMENTWISE_OPS
    assert "add" in ELEMENTWISE_OPS
    assert "relu" in ELEMENTWISE_OPS


def test_graph_holds_ops_in_order():
    ops = [
        Op(op_type="matmul", inputs=["x", "w"], output="h1", shape=(4,)),
        Op(op_type="add", inputs=["h1", "b"], output="h2", shape=(4,)),
        Op(op_type="relu", inputs=["h2"], output="out", shape=(4,)),
    ]
    graph = Graph(ops=ops, graph_inputs=["x", "w", "b"], graph_output="out")
    assert [op.op_type for op in graph.ops] == ["matmul", "add", "relu"]
    assert graph.graph_output == "out"
    assert "x" in graph.graph_inputs


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
