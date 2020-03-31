from typing import List


class ReshapePatcher:
    NODE_NAME = "ReshapePatcher/{ID}/Reshape"

    PATCH = '''
node {
  name: "ReshapePatcher/{ID}/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: {SHAPE_NUM_DIMS}
          }
        }
        tensor_content: "{SHAPE}"
      }
    }
  }
}

node {
  name: "{NODE_NAME}"
  op: "Reshape"
  input: "{INPUT_TENSOR_NAME}"
  input: "ReshapePatcher/{ID}/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
'''
    @staticmethod
    def dec2octpkg4(x: int):
        if x < 0:
            return "\\377\\377\\377\\377"
        octPkgStr = ''
        for i in range(4):
            octPkgStr = octPkgStr + \
                oct((x >> (i*8)) % 256).replace('0o', '\\')
        return octPkgStr

    def __init__(self):
        self.total = 0

    def get_patch(self, input_tensor_name: str, shape: List[int]) -> (str, str):
        """
        Returns: (node_name, patch)
        """
        node_name = self.NODE_NAME.replace("{ID}", str(self.total))

        patch = self.PATCH.replace("{ID}", str(self.total))
        patch = patch.replace("{NODE_NAME}", node_name)
        patch = patch.replace("{INPUT_TENSOR_NAME}", input_tensor_name)
        patch = patch.replace("{SHAPE_NUM_DIMS}", str(len(shape)))
        patch = patch.replace("{SHAPE}", "".join(map(self.dec2octpkg4, shape)))
        self.total += 1
        return (node_name, patch)
