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
2. For a small model made only of `nn.Linear` and `nn.ReLU` layers (this
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


def trace_model(model, example_input):
    raise NotImplementedError("implement me")
