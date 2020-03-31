import os
import re
import sys

import tensorflow as tf
from google.protobuf import text_format
from tensorflow import gfile
from tensorflow import io

from shape_fetcher import ShapeFetcher
from reshape_patcher import ReshapePatcher
from content_locate_utils import *


TEMP_PBTXT_FOLDER = "./pbtxt"

reshape_patcher = ReshapePatcher()


SWISH_PATCH = '''
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

MEAN_PATCH = '''
node {
  name: "{NAME}"
  op: "AvgPool"
  input: "{INPUT}"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: {KERNEL_SIZE}
        i: {KERNEL_SIZE}
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
'''

EXPLICT_PAD_ATTR_REGEX = r'attr {\n[\s]+key: "explicit_paddings"\n[\s]'\
    r'+value {\n[\s]+list {\n[\s]+}\n[\s]+}\n[\s]+}'

U_KEY_ATTR_REGEX = r'attr {\n[\s]+key: "U"\n[\s]+value {\n[\s]+type: DT_FLOAT\n[\s]+}\n[\s]+}'

OUTPUT_SHAPE_REGEX_1 = r'attr {\n[\s]+key: "_output_shapes"\n[\s]+value {\n[\s]+list {\n[\s]+shape '\
    r'{\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}'\
    r'\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n'\
    r'[\s]+}\n[\s]+shape {\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+}\n[\s]+shape'\
    r' {\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+}\n[\s]+shape {\n[\s]+dim {\n[\s'\
    r']+size: [0-9]+\n[\s]+}\n[\s]+}\n[\s]+shape {\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]'\
    r'+}\n[\s]+}\n[\s]+shape {\n[\s]+unknown_rank: true\n[\s]+}\n[\s]+}\n[\s]+}\n[\s]+}'

OUTPUT_SHAPE_REGEX_2 = r'[\s]+attr {\n[\s]+key: "_output_shapes"\n[\s]+value {\n[\s]+list {\n[\s]+shape '\
    r'{([\s]+dim[\s]+{\n[\s]+size:[\s]+[0-9]+\n[\s]+}\n)+([\s]+}\n)+'

REDUCTION_IDENCE_REGEX = r'node[\s]+{\n[\s]+name:[\s]+\"[^"]+reduction_indices'\
    r'\"\n[\s]+op:[\s]+\"Const\"[\s]+[\s]+attr[\s]+{\n[\s]+k'\
    r'ey:[\s]+\"dtype\"\n[\s]+value[\s]+{\n[\s]+type:[\s]+DT'\
    r'_INT32\n[\s]+}\n[\s]+}\n[\s]+attr[\s]+{\n[\s]+key:[\s]'\
    r'+\"value\"\n[\s]+value[\s]+{\n[\s]+tensor[\s]+{\n[\s]+'\
    r'dtype:[\s]+DT_INT32\n[\s]+tensor_shape[\s]+{\n[\s]+dim'\
    r'[\s]+{\n[\s]+size:[\s]+2\n[\s]+}\n[\s]+}\n[\s]+tensor_'\
    r'content:[\s]+\"\\001\\000\\000\\000\\002\\000\\000\\00'\
    r'0\"\n+([\s]+}\n[\s]+)(}\n[\s]+}\n})'


def patch_mean(content) -> str:
    mean_count = content.count('op: "Mean"')

    print(
        f'Find semi-supported op: Mean presented {mean_count} times in pb, patching...')
    for _ in range(mean_count):
        op_detail = locate_op_with_type(content, "Mean")
        input_node = op_detail.inputs[0][-1]

        mean_input_shape = shape_fetcher.shape_results[input_node + ":0"]

        print(f'Node: {op_detail.name}, Input Shape = {mean_input_shape}')
        print('Patching the Mean operator...')

        print(f'Generating the patcher, node input: {input_node}')
        patcher = MEAN_PATCH.replace('{NAME}', op_detail.name)
        patcher = patcher.replace('{INPUT}', input_node)
        patcher = patcher.replace('{KERNEL_SIZE}', str(mean_input_shape[1]))

        print('Inserting patch and removing the Mean node...\n')
        content = content[:op_detail.l] + patcher + content[op_detail.r + 1:]

        # check whether to use reshape or not
        op = shape_fetcher.graph.get_operation_by_name(op_detail.name)
        origin_output_shape = shape_fetcher.shape_results[op.outputs[0].name]
        if len(origin_output_shape) < 4:
            # use reshape
            print("Adding reshape because the output is squeezed...")
            reshape_node_name, patch = reshape_patcher.get_patch(
                op_detail.name, origin_output_shape
            )
            content = patch + "\n" + content
            for origin_output_op in shape_fetcher.get_nodes_with_input_tensor(
                op.outputs[0]
            ):
                origin_output_op_detail = locate_op(
                    content, origin_output_op.name
                )
                content = content[:origin_output_op_detail.inputs[0][0]] +\
                    reshape_node_name +\
                    content[origin_output_op_detail.inputs[0][1] + 1:]

    print('Removing unused const by Mean.\n')
    content = re.sub(REDUCTION_IDENCE_REGEX, '', content)

    while 1:
        indecOpLoc = content.find('reduction_indices')
        if indecOpLoc == -1:
            break
        indecNameLoc = content.rfind('name', 0, indecOpLoc)
        indecStart = content.rfind('{', 0, indecNameLoc)
        indecEnd = content.find('node', indecNameLoc)
        content = content[:content.rfind(
            'node', 0, indecStart)] + content[indecEnd:]

    return content


def pbtxt_processing(content):
    content = patch_mean(content)

    if content.find('explicit_paddings') != -1:
        print('Find unsupported attr: explicit_paddings, removing...\n')
        content = re.sub(EXPLICT_PAD_ATTR_REGEX, '', content)

    if content.find('AddV2') != -1:
        print('Find unsupported op: AddV2, patching...\n')
        content = content.replace('AddV2', 'Add')

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
            nodeName = content[nodeNameDLoc +
                               1:content.find('"', nodeNameDLoc + 1)]

            nodeInputLoc = content.find('input', swishOpLoc)
            nodeInputDLoc = content.find('"', nodeInputLoc)
            nodeInputName = content[nodeInputDLoc +
                                    1:content.find('"', nodeInputDLoc + 1)]

            print(
                f'Found Node name: {nodeName}\nPatching the swish_f32 operator...')

            nodeStart = content.rfind('{', 0, nodeNameLoc)
            nodeEnd = content.find('node', nodeNameLoc)

            print(f'Generating the patcher, node input: {nodeInputName}')
            patcher = SWISH_PATCH.replace('{NAME}', nodeName)
            patcher = patcher.replace('{INPUT}', nodeInputName)

            print('Inserting patch and removing the swish_f32 node...')
            content = content[:content.rfind(
                'node', 0, nodeStart)] + patcher + content[nodeEnd:]

            print('Reconnecting the graph...\n')
            content = content.replace(
                f'input: "{nodeName}"', f'input: "{nodeName}/mul"')

    return content


if __name__ == "__main__":
    FILE_NAME = sys.argv[1]

    if not os.path.isfile(os.path.join(TEMP_PBTXT_FOLDER, os.path.basename(FILE_NAME).split('.')[0] + '.pbtxt')):
        with gfile.FastGFile(FILE_NAME, 'rb') as f:
            GRAPH_DEF = tf.compat.v1.get_default_graph().as_graph_def(add_shapes=True)
            GRAPH_DEF.ParseFromString(f.read())

        tf.import_graph_def(GRAPH_DEF, name='')
        io.write_graph(GRAPH_DEF, TEMP_PBTXT_FOLDER,
                       os.path.basename(FILE_NAME).split('.')[0] + '.pbtxt', as_text=True)
    else:
        pbtxt_file_path = os.path.join(
            TEMP_PBTXT_FOLDER, os.path.basename(FILE_NAME).split('.')[0] + '.pbtxt')
        global shape_fetcher
        shape_fetcher = ShapeFetcher(pbtxt_file_path, ["Mean"])

        PBTXT_FILE = open(pbtxt_file_path, 'r')

        GRAPH_DEF = tf.get_default_graph().as_graph_def(add_shapes=True)

        FILE_CONTENT = pbtxt_processing(PBTXT_FILE.read())

        print('Content check OK, start merging...')
        open(os.path.basename(FILE_NAME).split('.')[
             0] + '_debug.pbtxt', 'w+').write(FILE_CONTENT)

        text_format.Merge(FILE_CONTENT, GRAPH_DEF)
        io.write_graph(GRAPH_DEF,
                       os.path.dirname(FILE_NAME),
                       os.path.basename(FILE_NAME).split('.')[
                           0] + '_patched.pb',
                       as_text=False)
