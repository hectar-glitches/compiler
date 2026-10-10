"""
Lower an IR graph to a flat Load/Store/Compute instruction stream.

Inputs are loaded when they are not resident. Outputs are retained while
the configured buffer has capacity and are stored when capacity is exceeded
or when they are the graph output.

The lowering sequence for each operation is:
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

With no capacity supplied, every result is stored immediately, providing a
deterministic traffic baseline.

The lowering function returns the instruction list and derives byte counts
and FLOP estimates from tensor shapes.
"""

import math

from .ir import Graph
from .isa import Compute, Load, Store


def _numel(shape):
    return math.prod(shape)


def _flops(op_type, shape, input_shapes):
    elements = _numel(shape)
    if op_type == "matmul":
        if len(input_shapes) != 2 or len(input_shapes[0]) != 2 or len(input_shapes[1]) != 2:
            raise ValueError("matmul requires two rank-2 inputs")
        return 2 * input_shapes[0][0] * input_shapes[0][1] * input_shapes[1][1]
    if op_type in {"add", "mul", "relu"}:
        return elements
    raise ValueError(f"unsupported op type: {op_type}")


def lower_to_isa(
    graph,
    tensor_shapes,
    dtype_size: int = 4,
    on_chip_buffer_bytes: int | None = None,
):
    if dtype_size <= 0:
        raise ValueError("dtype_size must be positive")
    if on_chip_buffer_bytes is not None and on_chip_buffer_bytes <= 0:
        raise ValueError("on_chip_buffer_bytes must be positive")

    instructions = []
    resident = set()
    resident_bytes = 0
    for op in graph.ops:
        for name in op.inputs:
            if name not in tensor_shapes:
                raise KeyError(f"missing shape for tensor {name!r}")
            if name not in resident:
                size_bytes = _numel(tensor_shapes[name]) * dtype_size
                instructions.append(Load(name, size_bytes))
                resident.add(name)
                resident_bytes += size_bytes

        shape = tensor_shapes.get(op.output, op.shape)
        input_shapes = [tensor_shapes[name] for name in op.inputs]
        sub_ops = op.fused_ops
        op_types = sub_ops or [op.op_type]
        output_size_bytes = _numel(shape) * dtype_size
        flops = sum(_flops(kind, shape, input_shapes) for kind in op_types)
        instructions.append(
            Compute(
                op_type=op.op_type,
                input_names=list(op.inputs),
                output_name=op.output,
                output_size_bytes=output_size_bytes,
                flops=flops,
                sub_ops=list(sub_ops) if sub_ops else None,
            )
        )
        resident.add(op.output)
        resident_bytes += output_size_bytes

        must_store = on_chip_buffer_bytes is None or op.output == graph.graph_output
        if (
            on_chip_buffer_bytes is not None
            and resident_bytes > on_chip_buffer_bytes
        ):
            must_store = True
        if must_store:
            instructions.append(Store(op.output, output_size_bytes))
            resident.discard(op.output)
            resident_bytes -= output_size_bytes
    return instructions
