"""
Convert a supported PyTorch module to the project IR with `torch.fx`.

The converter supports `nn.Linear` and `nn.ReLU`. Each Linear becomes a
matmul followed by a bias add; each ReLU becomes a relu operation.

The converter uses the traced module graph and parameter values to produce
the compiler IR and simulator inputs.

`torch.fx.symbolic_trace(model)` produces an `fx.GraphModule` whose
   `.graph.nodes` you can iterate. Each node has an `op` (e.g.
   'call_module', 'call_function'), a `target`, and `args`.
The supported module mapping is:
   - a Linear layer's node -> a `matmul` Op (against its weight) followed
     by an `add` Op (its bias) in your IR
   - a ReLU layer's node -> a `relu` Op
The converter copies weight and bias values into the simulator inputs and
records concrete shapes for every tensor.

Public interface:

`trace_model(model, example_input)` returns the IR graph, NumPy values for
the input and parameters, and concrete shapes for every tensor required by
code generation.
"""

from .ir import Graph, Op


def trace_model(model, example_input):
    try:
        import torch
        import torch.nn as nn
        from torch.fx import symbolic_trace
    except ImportError as exc:
        raise ImportError(
            "trace_model requires PyTorch; install the dependency from requirements.txt"
        ) from exc

    if not isinstance(example_input, torch.Tensor):
        raise TypeError("example_input must be a torch.Tensor")

    traced = symbolic_trace(model)
    tensor_values = {
        "input": example_input.detach().cpu().numpy().copy()
    }
    tensor_shapes = {"input": tuple(example_input.shape)}
    graph_inputs = []
    ops = []
    node_names = {}

    def node_name(node):
        if node.op == "placeholder":
            return "input"
        if node.op != "output":
            return node.name
        value = node.args[0]
        return node_name(value) if hasattr(value, "op") else str(value)

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            name = node_name(node)
            node_names[node] = name
            graph_inputs.append(name)
            continue

        if node.op == "call_module":
            module = traced.get_submodule(node.target)
            input_node = node.args[0]
            input_name = node_names[input_node]
            input_shape = tensor_shapes[input_name]
            output_name = node.name

            if isinstance(module, nn.Linear):
                if module.bias is None:
                    raise ValueError(f"Linear layer {node.target!r} must have a bias")
                weight_name = f"{node.name}_weight"
                bias_name = f"{node.name}_bias"
                matmul_name = f"{node.name}_matmul"
                weight = module.weight.detach().cpu().numpy().T.copy()
                bias = module.bias.detach().cpu().numpy().copy()
                output_shape = tuple(input_shape[:-1]) + (module.out_features,)

                tensor_values[weight_name] = weight
                tensor_values[bias_name] = bias
                tensor_shapes[weight_name] = weight.shape
                tensor_shapes[bias_name] = bias.shape
                tensor_shapes[matmul_name] = output_shape
                tensor_shapes[output_name] = output_shape
                ops.extend(
                    [
                        Op("matmul", [input_name, weight_name], matmul_name, output_shape),
                        Op("add", [matmul_name, bias_name], output_name, output_shape),
                    ]
                )
            elif isinstance(module, nn.ReLU):
                tensor_shapes[output_name] = input_shape
                ops.append(Op("relu", [input_name], output_name, input_shape))
            else:
                raise ValueError(
                    f"unsupported module {type(module).__name__!r} at {node.target!r}"
                )
            node_names[node] = output_name
            continue

        if node.op == "output":
            graph_output = node_name(node.args[0])
            continue

        raise ValueError(f"unsupported FX node: {node.op} ({node.target!r})")

    return Graph(ops, graph_inputs, graph_output), tensor_values, tensor_shapes
