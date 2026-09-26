"""
Our own IR: a flat list of typed operations over named tensors, roughly
analogous to what torch.fx or an ONNX graph gives you, but simplified to
exactly what we need.

Design this yourself, but here's the shape to aim for:

`Op` needs at minimum:
    - op_type: str          e.g. "matmul", "add", "relu"
    - inputs: list[str]     names of input tensors
    - output: str           name of the output tensor this op produces
    - shape: tuple[int,...] the output tensor's shape (needed later for
                             the cost model to compute memory traffic)

`Graph` needs:
    - a list of Op, in execution order
    - a way to know which tensor names are the graph's overall inputs
      (not produced by any Op) and which is the final output

A useful thing to add on Op, since the fusion pass needs it constantly:
    - a way to classify an op as elementwise or not. A simple approach:
      a module-level set `ELEMENTWISE_OPS = {"add", "relu", "mul"}` and a
      property/method that checks membership. Keep "matmul" out of that
      set on purpose (see README's "Fusion correctness" section for why).

You'll also eventually want, for the fusion pass specifically, a cheap way
to answer "how many ops in this graph consume tensor X as an input?" --
that's how you detect the multi-consumer case that should block fusion.
Whether you compute that on demand or maintain it incrementally is your
call.
"""

from dataclasses import dataclass, field


ELEMENTWISE_OPS = {"add", "relu", "mul"}
# matmul is deliberately NOT in this set -- see README.md's
# "Fusion correctness" section for why elementwise fusion rules don't
# apply to it.


@dataclass
class Op:
    pass
    # TODO: op_type, inputs, output, shape at minimum


@dataclass
class Graph:
    pass
    # TODO: ops: list[Op], graph_inputs: list[str], graph_output: str
