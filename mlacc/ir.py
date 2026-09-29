"""
My own IR: a flat list of typed operations over named tensors, roughly
analogous to what torch.fx or an ONNX graph gives you, but simplified to
exactly what we need.

"""

from dataclasses import dataclass, field


ELEMENTWISE_OPS = {"add", "relu", "mul"}


@dataclass
class Op:
    op_type: str
    inputs: list[str]
    output: str
    shape: tuple[int, ]
    


@dataclass
class Graph:
    ops: list[Op]
    graph_inputs: list[str]
    graph_output: str 