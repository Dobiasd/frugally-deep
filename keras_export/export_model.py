#!/usr/bin/env python3
"""Export a keras model to frugally-deep.
"""

import json
import sys

import numpy as np

from keras.models import load_model
from keras import backend as K

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"

def write_text_file(path, text):
    with open(path, "w") as text_file:
        print(text, file=text_file)

def arr3_to_channels_first_format(arr):
    assert len(arr.shape) == 3
    image_format = K.image_data_format()
    if image_format == 'channels_last':
        return np.swapaxes(arr, 0, 2)
    else:
        return arr

def arr_as_arr3(arr):
    depth = len(arr.shape)
    if depth == 1:
        return arr.reshape(1, 1, *arr.shape)
    if depth == 2:
        return arr.reshape(1, *arr.shape)
    if depth == 3:
        return arr3_to_channels_first_format(arr)
    if depth == 4 and arr.shape[0] == 1:
        return arr3_to_channels_first_format(arr.reshape(arr.shape[1:]))
    else:
        raise ValueError('invalid number of dimensions')

def show_tensor3(tens):
    return {
        'shape': tens.shape,
        'values': tens.flatten().tolist()
    }

def show_test_data_as_3tensor(arr):
    return show_tensor3(arr_as_arr3(arr))

def gen_test_data(model):
    data_in = list(map(lambda l: np.random.random((1, *l.input_shape[1:])),
        model.input_layers))
    data_out = model.predict(data_in)
    return {
        'inputs': list(map(show_test_data_as_3tensor, data_in)),
        'outputs': list(map(show_test_data_as_3tensor, data_out))
    }

def show_conv2d_layer(layer):
    weights = layer.get_weights()
    assert len(weights) == 1 or len(weights) == 2
    assert len(weights[0].shape) == 4
    weights_flat = np.swapaxes(
        np.swapaxes(weights[0], 0, 3), 1, 2).flatten().tolist()
    assert len(weights_flat) > 0
    assert layer.dilation_rate == (1,1)
    assert layer.padding in ['valid', 'same']
    assert len(layer.input_shape) == 4
    assert layer.input_shape[0] == None
    result = {
        'weights': weights_flat
    }
    if len(weights) == 2:
        result['bias'] = weights[1].tolist()
    return result

def show_batch_normalization_layer(layer):
    assert layer.axis == -1 or layer.axis == 3
    result = {}
    if layer.center:
        result['beta'] = K.get_value(layer.beta).tolist()
    if layer.scale:
        result['gamma'] = K.get_value(layer.gamma).tolist()
    return result

def show_dense_layer(layer):
    assert len(layer.input_shape) == 2, "Please flatten for dense layer."
    assert layer.input_shape[0] == None, "Please flatten for dense layer."
    weights = layer.get_weights()
    assert len(weights) == 1 or len(weights) == 2
    assert len(weights[0].shape) == 2
    result = {
        'weights': weights[0].flatten().tolist()
    }
    if len(weights) == 2:
        result['bias'] = weights[1].tolist()
    return result

def get_dict_keys(d):
    return [key for key in d]

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def merge_two_disjunct_dicts(x, y):
    assert set(get_dict_keys(x)).isdisjoint(get_dict_keys(y))
    return merge_two_dicts(x, y)

def is_ascii(str):
    try:
        str.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True

def get_all_weights(model):
    show_layer_functions = {
        'Conv2D': show_conv2d_layer,
        'BatchNormalization': show_batch_normalization_layer,
        'Dense': show_dense_layer
    }
    result = {}
    layers = model.layers
    for layer in layers:
        layer_type = type(layer).__name__
        if layer_type in ['Model', 'Sequential']:
            result = merge_two_disjunct_dicts(result, get_all_weights(layer))
        else:
            show_func = show_layer_functions.get(layer_type, None)
            name = layer.name
            is_ascii(name)
            if name in result:
                raise ValueError('duplicate layer name ' + name)
            if show_func:
                result[name] = show_func(layer)
    return result

def convert_sequential_to_model(model):
    if type(model).__name__ == 'Sequential':
        name = model.name
        inbound_nodes = model.inbound_nodes
        model = model.model
        model.name = name
        model.inbound_nodes = inbound_nodes
    assert len(model.input_layers) > 0
    assert len(model.layers) > 0
    for i in range(len(model.layers)):
        if type(model.layers[i]).__name__ in ['Model', 'Sequential']:
            model.layers[i] = convert_sequential_to_model(model.layers[i])
    return model

def main():
    usage = 'usage: [Keras model in HDF5 format] [output path] [test count = 1]'
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print(usage)
        sys.exit(1)
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        test_count = 3
        if len(sys.argv) == 4:
            test_count = int(sys.argv[3])
        model = load_model(in_path)
        model = convert_sequential_to_model(model)

        json_output = {}

        json_output['architecture'] = json.loads(model.to_json())

        json_output['trainable_params'] = get_all_weights(model)

        json_output['tests'] = [gen_test_data(model) for _ in range(test_count)]

        json_output['image_data_format'] = K.image_data_format()

        write_text_file(out_path, json.dumps(
                json_output, allow_nan=False, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
