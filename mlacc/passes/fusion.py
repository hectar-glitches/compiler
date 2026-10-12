"""
Elementwise operator fusion pass.

The pass returns a new Graph. Consecutive, single-consumer elementwise
operations such as `add -> relu` become one `fused_elementwise` operation
with the original sub-operation sequence recorded.

"""

from ..ir import Graph, Op, ELEMENTWISE_OPS
from collections import defaultdict
                        


def fuse_elementwise(graph):
      consumer_count = defaultdict(int)

      for op in graph.ops:
            for input_name in op.inputs:
                  consumer_count[input_name]+=1


      def can_extend_chain(op, chain_output, consumer_count):
           if op.op_type not in ELEMENTWISE_OPS:
                return False

           if chain_output not in op.inputs:
                return False

           if consumer_count[chain_output] != 1:
                return False

           return True

      chain = []
      current_chain = []
      chain_output_tensor = None

      def flush_chain():
            if len(current_chain) == 0:
                  return

            elif len(current_chain) == 1:
                  chain.append(current_chain[0])

            else:
                  internal_outputs = {o.output for o in current_chain}
                  inputs = [name for o in current_chain for name in o.inputs if name not in internal_outputs]

                  fused_op = Op(
                        op_type = "fused_elementwise",
                        inputs = inputs,
                        output = current_chain[-1].output,
                        shape = current_chain[-1].shape,
                        fused_ops = [o.op_type for o in current_chain],
                  )

                  chain.append(fused_op)

            current_chain.clear()
            
      for op in graph.ops:
            if chain_output_tensor is not None and can_extend_chain(op, chain_output_tensor, consumer_count):
                  current_chain.append(op)
                  chain_output_tensor = op.output

            else:
                  flush_chain()
                  if op.op_type in ELEMENTWISE_OPS:
                        current_chain.append(op)
                        chain_output_tensor = op.output 

                  else:
                        chain.append(op)
                        chain_output_tensor = None


      flush_chain()

      return Graph(
            ops=chain,
            graph_inputs=graph.graph_inputs,
            graph_output=graph.graph_output
      )