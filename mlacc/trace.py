"""
Front-end: trace a real PyTorch nn.Module with torch.fx and convert its
graph into our own IR (mlacc.ir.Graph).

This is the ONLY module that needs `torch` installed -- everything else
in this project (ir, passes, isa, cost_model, simulator, codegen) is pure
Python + numpy, deliberately, so you can build and test the entire
middle/back-end before touching torch at all.

Read the torch.fx docs before starting:
https://pytorch.org/docs/stable/fx.html

Sketch of what you need:

1. `torch.fx.symbolic_trace(model)` gives you an `fx.GraphModule` whose
   `.graph.nodes` you can iterate. Each node has an `op` (e.g.
   'call_module', 'call_function'), a `target`, and `args`.
2. For a model made of `nn.Linear` and `nn.ReLU` layers (this
   is the scope to aim for -- don't try to support arbitrary PyTorch
   models), you need to map:
   - a Linear layer's node -> a `matmul` Op (against its weight) followed
     by an `add` Op (its bias) in your IR
   - a ReLU layer's node -> a `relu` Op
3. You'll need the actual weight/bias tensor VALUES (not just shapes) out
   of the traced module, since the simulator needs real numbers to
   compute against. `model.state_dict()` or iterating `model.named_parameters()`
   on the original (untraced) module is the simplest way to get these.
4. Track tensor shapes as you go (input shape is whatever you decide to
   trace with a concrete example input) -- codegen needs a shape for
   every tensor.

Function to implement:

- `trace_model(model, example_input) -> tuple[Graph, dict[str, "np.ndarray"], dict[str, tuple]]`
    Returns: the IR graph, a dict of tensor_name -> numpy array for every
    tensor with a known concrete value at trace time (inputs + all
    weights/biases), and a dict of tensor_name -> shape for every tensor
    in the graph (including intermediates, which codegen needs).
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
