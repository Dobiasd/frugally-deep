#!/usr/bin/env python3
"""Force the single-input fdeep model JSON into Keras' newer flat
input_layers/output_layers format so we can regression-test loading it
on any Keras version. See https://github.com/Dobiasd/frugally-deep/issues/461.
"""

import json
import sys


def flatten(node_connections):
    if (
        isinstance(node_connections, list)
        and len(node_connections) == 1
        and isinstance(node_connections[0], list)
    ):
        return node_connections[0]
    return node_connections


def main(src, dst):
    with open(src, 'r') as f:
        data = json.load(f)
    config = data['architecture']['config']
    config['input_layers'] = flatten(config['input_layers'])
    config['output_layers'] = flatten(config['output_layers'])
    with open(dst, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
