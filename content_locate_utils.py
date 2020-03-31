from collections import namedtuple
import re

OpDetail = namedtuple(
    "OpDetail",
    [
        "l", "r",  # int
        "name", "op",  # str
        "inputs"  # List[Tuple[int, int, str]]
    ]
)


def locate_op(content: str, name: str) -> OpDetail:
    idx = content.find('name: "{}"'.format(name))
    assert idx >= 0

    l = content.rfind("node", 0, idx)
    top = 1
    for i in range(content.find("{", l) + 1, len(content)):
        if content[i] == "{":
            top += 1
        elif content[i] == "}":
            top -= 1
        if top == 0:
            r = i
            break

    op = content.find("op:", l)
    op = content.find("\"", op)
    op = content[op + 1: content.find("\"", op + 1)]

    inputs = []

    idx = l
    while True:
        idx = content.find("input:", idx)
        idx = content.find("\"", idx)
        if not (0 <= idx and idx <= r):
            break

        curr_input = [idx + 1, content.find("\"", idx + 1) - 1]
        curr_input.append(content[curr_input[0]: curr_input[1] + 1])
        inputs.append(tuple(curr_input))

    return OpDetail(l, r, name, op, inputs)


def locate_op_with_type(content: str, op_type: str) -> OpDetail:
    idx = content.find('op: "{}"'.format(op_type))

    name_idx = content.rfind('name:', 0, idx)
    name_idx = content.find('"', name_idx)
    node_name = content[
        name_idx + 1: content.find('"', name_idx + 1)
    ]
    return locate_op(content, node_name)
