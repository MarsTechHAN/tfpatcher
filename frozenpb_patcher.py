import os
import re
import sys

import tensorflow as tf
from google.protobuf import text_format
from tensorflow import gfile
from tensorflow import io

REDUCE_PATCH_PKG = {}
REDUCE_PATCH_PKG['default'] = {'shape':[-1, 1280],
              'octContent':'\\377\\377\\377\\377\\000\\005\\000\\000'}

REDUCE_PATCH_PKG['nasnet'] = {'shape':[-1, 1056],
              'octContent':'\\377\\377\\377\\377\\040\\004\\000\\000'}

KEEP_DIM_PATCH =\
'''
  attr {
    key: "keep_dims"
    value {
      b: {KEEP_DIM}
    }
  }
'''

REDUCE_DIM_PATCH =\
'''
node {
  name: "reshape/Reshape/shape"
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
            size: 2
          }
        }
        tensor_content: "{SHAPE}"
      }
    }
  }
}

node {
  name: "reshape/Reshape"
  op: "Reshape"
  input: "{INPUT_TENSOR_NAME}"
  input: "reshape/Reshape/shape"
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

SWISH_PATCH =\
'''
node {
  name: "{NAME}/Sigmoid"
  op: "Sigmoid"
  input: "{INPUT}"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}

node {
  name: "{NAME}/mul"
  op: "Mul"
  input: "{INPUT}"
  input: "{NAME}/Sigmoid"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
'''

EXPLICT_PAD_ATTR_REGEX = r'attr {\n[\s]+key: "explicit_paddings"\n[\s]'\
                      r'+value {\n[\s]+list {\n[\s]+}\n[\s]+}\n[\s]+}'

U_KEY_ATTR_REGEX = r'attr {\n[\s]+key: "U"\n[\s]+value {\n[\s]+type: DT_FLOAT\n[\s]+}\n[\s]+}'

OUTPUT_SHAPE_REGEX_3 = r'([\s]+attr {\n[\s]+key: "_output_shapes"\n[\s]+value {\n[\s]+list {\n['\
                    r'\s]+)(shape[\s]+{[\s]+([\s]+(dim[\s]+{\s+size:[\s]+[0-9]+[\s]+})|([\s]+'\
                    r'unknown_rank: \w+([\s]+})+))+([\s]+}[\s]+)+)+([\s]})+'

OUTPUT_SHAPE_REGEX_1 = r'attr {\n[\s]+key: "_output_shapes"\n[\s]+value {\n[\s]+list {\n[\s]+shape '\
                    r'{\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}'\
                    r'\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n'\
                    r'[\s]+}\n[\s]+shape {\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+}\n[\s]+shape'\
                    r' {\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+}\n[\s]+shape {\n[\s]+dim {\n[\s'\
                    r']+size: [0-9]+\n[\s]+}\n[\s]+}\n[\s]+shape {\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]'\
                    r'+}\n[\s]+}\n[\s]+shape {\n[\s]+unknown_rank: true\n[\s]+}\n[\s]+}\n[\s]+}\n[\s]+}'

OUTPUT_SHAPE_REGEX_2 = r'[\s]+attr {\n[\s]+key: "_output_shapes"\n[\s]+value {\n[\s]+list {\n[\s]+shape '\
                    r'{([\s]+dim[\s]+{\n[\s]+size:[\s]+[0-9]+\n[\s]+}\n)+([\s]+}\n)+'

def pbtxt_processing(content):
    if content.find('explicit_paddings') != -1:
      print('Find unsupported attr: explicit_paddings, removing...\n')
      content = re.sub(EXPLICT_PAD_ATTR_REGEX, '', content)

    if content.find(KEEP_DIM_PATCH.replace('{KEEP_DIM}', 'false')) != -1:
      print('Find unsupported op: reduce_dim=false, patching...')

      while 1:   
        keepDimLoc = content.find(KEEP_DIM_PATCH.replace('{KEEP_DIM}', 'false'))
        if keepDimLoc == -1:
          break

        nodeNameLoc = content.rfind('name', 0, keepDimLoc)
        nodeNameDLoc = content.find('"', nodeNameLoc) 
        nodeName = content[nodeNameDLoc+1:content.find('"', nodeNameDLoc + 1)]
        print(f'Found Node name: {nodeName}, Output Shape: {REDUCE_PATCH_PKG[NET_TYPE]["shape"]}')
        print('Patching the Mean operator...')
        
        nodeEnd = content.find('node', nodeNameLoc)
        content = content.replace(f'input: "{nodeName}"', 'input: "reshape/Reshape"')

        patcher = REDUCE_DIM_PATCH.replace('{INPUT_TENSOR_NAME}', nodeName)
        patcher = patcher.replace('{SHAPE}', REDUCE_PATCH_PKG[NET_TYPE]['octContent'])

        content = content[:nodeEnd] + patcher + content[nodeEnd:]

        content = content.replace(KEEP_DIM_PATCH.replace('{KEEP_DIM}', 'false'), KEEP_DIM_PATCH.replace('{KEEP_DIM}', 'true'))
        print('Modified reduce_dim=true...\n') 

    if content.find('FusedBatchNormV3') != -1:
      print('Find unsupported op: FusedBatchNormV3, patching...\n')
      content = content.replace('FusedBatchNormV3', 'FusedBatchNorm')
      content = re.sub(U_KEY_ATTR_REGEX, '', content)
      content = re.sub(OUTPUT_SHAPE_REGEX_1, '', content)
      content = re.sub(OUTPUT_SHAPE_REGEX_2, '', content)

    if content.find('op: "swish_f32"') != -1:
      print('Find unsupported op: swish_f32, patching...')
      while 1:
        swishOpLoc = content.find('op: "swish_f32"')
        if swishOpLoc == -1:
          break
        nodeNameLoc = content.rfind('name', 0, swishOpLoc)
        nodeNameDLoc = content.find('"', nodeNameLoc)
        nodeName = content[nodeNameDLoc+1:content.find('"', nodeNameDLoc + 1)]

        nodeInputLoc = content.find('input', swishOpLoc)
        nodeInputDLoc = content.find('"', nodeInputLoc)
        nodeInputName = content[nodeInputDLoc+1:content.find('"', nodeInputDLoc + 1)]

        print(f'Found Node name: {nodeName}\nPatching the swish_f32 operator...')

        nodeStart =  content.rfind('{', 0, nodeNameLoc)
        nodeEnd = content.find('node', nodeNameLoc)

        print(f'Generating the patcher, node input: {nodeInputName}')
        patcher = SWISH_PATCH.replace('{NAME}', nodeName)
        patcher = patcher.replace('{INPUT}', nodeInputName)

        print('Inserting patch and removing the swish_f32 node...')
        content = content[:content.rfind('node', 0 ,nodeStart)] + patcher +  content[nodeEnd:]

        print('Reconnecting the graph...\n')
        content = content.replace(f'input: "{nodeName}"', f'input: "{nodeName}/mul"')

    return content

FILE_NAME = sys.argv[1]

NET_TYPE = 'default'
if len(sys.argv) > 2:
  NET_TYPE = sys.argv[2]

if not os.path.isfile(os.path.join('../pbtxt/', os.path.basename(FILE_NAME).split('.')[0] + '.pbtxt')):
  with gfile.FastGFile(FILE_NAME,'rb') as f:
      GRAPH_DEF = tf.compat.v1.GraphDef()
      GRAPH_DEF.ParseFromString(f.read())

  tf.import_graph_def(GRAPH_DEF, name='')
  io.write_graph(GRAPH_DEF, '../pbtxt', os.path.basename(FILE_NAME).split('.')[0] + '.pbtxt', as_text=True)
else:
  PBTXT_FILE = open(os.path.join('../pbtxt/', os.path.basename(FILE_NAME).split('.')[0] + '.pbtxt'), 'r') 

  GRAPH_DEF = tf.GraphDef()

  FILE_CONTENT = pbtxt_processing(PBTXT_FILE.read())

  print('Content check OK, start merging...')
  open(os.path.basename(FILE_NAME).split('.')[0] + '_debug.pbtxt', 'w+').write(FILE_CONTENT)


  text_format.Merge(FILE_CONTENT, GRAPH_DEF)
  io.write_graph(GRAPH_DEF,
                 os.path.dirname(FILE_NAME),
                 os.path.basename(FILE_NAME).split('.')[0] + '_patched.pb',
                 as_text=False)
