import json
import sys

from keras.models import load_model
from keras import backend as K

import numpy as np

def write_text_file(path, text):
    with open(path, "w") as text_file:
        print(text, file=text_file)

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def get_names(xs):
    return list(map(lambda x: x.name, xs))

def show_connection(layer_name, node_index, tensor_index, conn_translations):
    if layer_name in conn_translations:
        layer_name = conn_translations[layer_name]
    # https://stackoverflow.com/questions/45722380/understanding-keras-model-architecture-tensor-index
    return {
        'layer': layer_name,
        'node_index': node_index,
        'tensor_index': tensor_index
        }

def show_connections(layers, node_indices, tensor_indices, conn_translations):
    assert len(layers) == len(node_indices) == len(tensor_indices)
    result = []
    for i in range(len(layers)):
        layer_name = layers[i].name
        node_index = node_indices[i]
        tensor_index = tensor_indices[i]
        result.append(show_connection(
            layer_name, node_index, tensor_index, conn_translations))
    return result

def node_inbound_layers_dict(node, conn_translations):
    return show_connections(
        node.inbound_layers, node.node_indices, node.tensor_indices,
        conn_translations)

def layer_inbound_nodes(layer, conn_translations):
    result = []
    for inbound_node in layer.inbound_nodes:
        result.extend(node_inbound_layers_dict(inbound_node,
            conn_translations))
    return result

def layer_def_dict(layer, conn_translations):
    return {
        'name': layer.name,
        'inbound_nodes': layer_inbound_nodes(layer, conn_translations)
    }

def add_layer_def_dict(layer, d, conn_translations):
    return merge_two_dicts(layer_def_dict(layer, conn_translations), d)

def show_relu_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'ReLU'
        }, conn_translations)

def show_leaky_relu_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'LeakyReLU',
        'alpha': float(layer.alpha)
        }, conn_translations)

def show_elu_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'ELU',
        'alpha': float(layer.alpha)
        }, conn_translations)

def show_conv2d_layer(layer, conn_translations):
    weights = layer.get_weights()
    assert layer.dilation_rate == (1,1)
    return add_layer_def_dict(layer, {
        'type': 'Conv2D',
        'filters': layer.filters,
        'weights': [1, np.swapaxes(weights[0], 0, 2).flatten().tolist(), 1],
        'biases': weights[1].tolist(),
        'kernel_size': layer.kernel_size,
        'use_bias': layer.use_bias,
        'padding': layer.padding,
        'strides': layer.strides
    }, conn_translations)

def show_batch_normalization_layer(layer, conn_translations):
    # momentum is only important for training
    assert layer.axis == -1
    return add_layer_def_dict(layer, {
        'type': 'BatchNormalization',
        'epsilon': layer.epsilon,
        'gamma': K.get_value(layer.gamma).tolist(),
        'beta': K.get_value(layer.beta).tolist(),
        'scale': layer.scale,
        'center': layer.center
        }, conn_translations)

def show_dropout_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'Dropout',
        'rate': layer.rate
        }, conn_translations)

def show_maxpooling2d_layer(layer, conn_translations):
    assert layer.padding == 'valid'
    assert layer.strides == layer.pool_size
    return add_layer_def_dict(layer, {
        'type': 'MaxPooling2D',
        'pool_size': layer.pool_size
        }, conn_translations)

def show_averagepooling2d_layer(layer, conn_translations):
    assert layer.padding == 'valid'
    assert layer.strides == layer.pool_size
    return add_layer_def_dict(layer, {
        'type': 'AveragePooling2D',
        'pool_size': layer.pool_size
        }, conn_translations)

def show_upsampling2D_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'UpSampling2D',
        'size': layer.size
        }, conn_translations)

def show_flatten_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'Flatten'
        }, conn_translations)

def show_input_layer(layer, conn_translations):
    assert len(layer.input_shape) >= 2
    assert len(layer.inbound_nodes) == 1
    assert layer.input_shape[0] == None
    return add_layer_def_dict(layer, {
        'type': 'InputLayer',
        'input_shape': layer.input_shape[1:]
        }, conn_translations)

def show_dense_layer(layer, conn_translations):
    # dense layers can only be shared between notes with the same input shapes
    assert len(layer.input_shape) == 2, "Please flatten for dense layer."
    assert layer.input_shape[0] == None, "Please flatten for dense layer."
    assert layer.use_bias == True
    assert layer.kernel_constraint == None
    assert layer.bias_constraint == None
    weights, bias = layer.get_weights()
    return add_layer_def_dict(layer, {
        'type': 'Dense',
        'units': layer.units,
        'weights': weights.flatten().tolist(),
        'bias': bias.tolist()
        }, conn_translations)

def show_sigmoid_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'Sigmoid'
        }, conn_translations)

def show_hard_sigmoid_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'HardSigmoid'
        }, conn_translations)

def show_selu_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'SeLU'
        }, conn_translations)

def show_softmax_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'SoftMax'
        }, conn_translations)

def show_softplus_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'SoftPlus'
        }, conn_translations)

def show_tanh_layer(layer, conn_translations):
    return add_layer_def_dict(layer, {
        'type': 'TanH'
        }, conn_translations)

def show_concatenate_layer(layer, conn_translations):
    assert layer.axis == -1
    return add_layer_def_dict(layer, {
        'type': 'Concatenate',
        }, conn_translations)

def show_activation_layer(layer, conn_translations):
    show_activation_functions = {
        'softmax': show_softmax_layer,
        'softplus': show_softplus_layer,
        'tanh': show_tanh_layer,
        'relu': show_relu_layer,
        'sigmoid': show_sigmoid_layer,
        'hard_sigmoid': show_hard_sigmoid_layer,
        'selu': show_selu_layer
    }
    return show_activation_functions[layer.get_config()['activation']](
        layer, conn_translations)

def generate_inbound_nodes(inbound_layer, n):
    result = []
    for i in range(n):
        result.append(type('node', (object,), dict({
                'inbound_layers':
                    [type('node', (object,), dict({'name': inbound_layer}))],
                'node_indices': [0],
                'tensor_indices' : [0]
            })))
    return result

def show_model(model, _):
    result = {}
    result['type'] = 'Model'
    result['name'] = model.name
    result['layers'] = []

    used_names = set()
    def get_free_name(prefix):
        if not prefix in used_names:
            used_names.add(prefix)
            return prefix
        else:
            return get_free_name(prefix + '_2')

    # Split activations from layers
    conn_translations = {}
    for layer in model.layers:
        layer_type = type(layer).__name__
        if layer_type != 'Activation'\
                and 'activation' in layer.get_config()\
                and layer.get_config()['activation'] != 'linear':
            assert layer_type != "Concatenate"
            activation = layer.get_config()['activation']
            name = get_free_name(layer.name + "_activation")
            fake_layer = type('layer', (object,), dict({
                'inbound_nodes': generate_inbound_nodes(layer.name,
                    len(layer.inbound_nodes)),
                'name': name,
                'get_config': lambda : {'activation': activation}
            }))
            json_obj = show_activation_layer(fake_layer, conn_translations)
            result['layers'].append(json_obj)
            conn_translations[layer.name] = name

    for layer in model.layers:
        layer_type = type(layer).__name__
        show_function = SHOW_FUNCTIONS[layer_type]
        result['layers'].append(show_function(layer, conn_translations))

    print(model.summary()) # todo remove

    result['input_nodes'] = show_connections(
        model.input_layers,
        model.input_layers_node_indices,
        model.input_layers_tensor_indices,
        conn_translations)

    result['output_nodes'] = show_connections(
        model.output_layers,
        model.output_layers_node_indices,
        model.output_layers_tensor_indices,
        conn_translations)

    return result

SHOW_FUNCTIONS = {
    'InputLayer': show_input_layer,
    'Conv2D': show_conv2d_layer,
    'BatchNormalization': show_batch_normalization_layer,
    'Dropout': show_dropout_layer,
    'LeakyReLU': show_leaky_relu_layer,
    'ReLU': show_relu_layer,
    'ELU': show_elu_layer,
    'MaxPooling2D': show_maxpooling2d_layer,
    'AveragePooling2D': show_averagepooling2d_layer,
    'UpSampling2D': show_upsampling2D_layer,
    'Dense': show_dense_layer,
    'Concatenate': show_concatenate_layer,
    'Flatten': show_flatten_layer,
    'SoftMax': show_softmax_layer,
    'SoftPlus': show_softplus_layer,
    'TanH': show_tanh_layer,
    'Activation': show_activation_layer,
    'Model': show_model
}

def np_array_as_list(arr):
    return arr.tolist()

def generate_test_data(model):
    data_in = list(map(lambda l: np.random.random((1, *l.input_shape[1:])),
        model.input_layers))
    data_out = model.predict(data_in)
    return {
        'input': list(map(np_array_as_list, data_in)),
        'output': list(map(np_array_as_list, data_out))
    }

def main():
    usage = 'usage: [Keras model in HDF5 format] [output path] [test count = 3]'
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
        # todo add test_count test cases with results to yaml
        write_text_file(out_path + '.yml', model.to_yaml()) # todo remove

        model_json = show_model(model, None)
        tests = [generate_test_data(model) for _ in range(test_count)]
        model_json["tests"] = tests

        write_text_file(out_path,
            json.dumps(model_json, allow_nan=False, indent=2, sort_keys=True))
        

if __name__ == "__main__":
    main()
