import tensorflow as tf
import numpy as np
from typing import List


class ShapeFetcher:
    @staticmethod
    def _load_graph_from_pbtxt(pbtxt_file_path: str) -> tf.Graph:
        from google.protobuf import text_format

        with open(pbtxt_file_path, "rb") as f:
            graph_def = text_format.Parse(f.read(), tf.compat.v1.GraphDef())

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def=graph_def, name="")
        return graph

    def get_nodes_with_input_tensor(self, tensor: tf.Tensor) -> List[tf.Operation]:
        return list(filter(
            lambda op: (tensor in op.inputs) and (op.type not in ["Shape"]),
            self.graph.get_operations()
        ))

    def __init__(self, pbtxt_file_path: str, op_types: List[str]):
        graph = self._load_graph_from_pbtxt(pbtxt_file_path)

        ops = graph.get_operations()
        placeholders = list(filter(lambda op: op.type == "Placeholder", ops))
        assert len(placeholders) == 1
        graph_input_tensor = placeholders[0].outputs[0]
        graph_input_tensor_shape = graph_input_tensor.get_shape().as_list()
        assert graph_input_tensor_shape[1] == graph_input_tensor_shape[2]
        assert graph_input_tensor_shape[3] == 3
        self.imsize = graph_input_tensor_shape[1]
        self.graph: tf.Graph = graph

        tensors_to_fetch: List[tf.Tensor] = []
        for op in filter(lambda op: op.type in op_types, ops):
            tensors_to_fetch.extend(op.inputs)
            tensors_to_fetch.extend(op.outputs)

        shape_tensors = dict()
        for tensor in tensors_to_fetch:
            shape_tensors[tensor.name] = tf.compat.v1.shape(tensor)
        self.shape_results = dict()

        with tf.compat.v1.Session(graph=graph) as sess:
            fake_input = np.random.randn(1, self.imsize, self.imsize, 3)
            for tensor_name, shape_tensor in shape_tensors.items():
                self.shape_results[tensor_name] = sess.run(
                    shape_tensor, feed_dict={
                        graph_input_tensor: fake_input
                    }
                )
