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

def show_connection(layer_name, node_index, tensor_index):
    # https://stackoverflow.com/questions/45722380/understanding-keras-model-architecture-tensor-index
    return {
        'layer': layer_name,
        'node_index': node_index,
        'tensor_index': tensor_index
        }

def show_connections(layers, node_indices, tensor_indices):
    assert len(layers) == len(node_indices) == len(tensor_indices)
    result = []
    for i in range(len(layers)):
        layer_name = layers[i].name
        node_index = node_indices[i]
        tensor_index = tensor_indices[i]
        result.append(show_connection(
            layer_name, node_index, tensor_index))
    return result

def node_inbound_layers_dict(node):
    return show_connections(
        node.inbound_layers, node.node_indices, node.tensor_indices)

def layer_inbound_nodes(layer):
    result = []
    for inbound_node in layer.inbound_nodes:
        result.extend(node_inbound_layers_dict(inbound_node))
    return result

def add_activation_if_present(layer, d):
    layer_type = type(layer).__name__
    if layer_type != 'Activation'\
            and 'activation' in layer.get_config():
        return merge_two_dicts(d, {
            'activation': layer.get_config()['activation']})
    else:
        return d

def layer_def_dict(layer):
    return add_activation_if_present(layer, {
        'name': layer.name,
        'inbound_nodes': layer_inbound_nodes(layer)
    })

def add_layer_def_dict(layer, d):
    return merge_two_dicts(layer_def_dict(layer), d)

def show_relu_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'ReLU'
        })

def show_leaky_relu_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'LeakyReLU',
        'alpha': float(layer.alpha)
        })

def show_elu_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'ELU',
        'alpha': float(layer.alpha)
        })

def show_conv2d_layer(layer):
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
    })

def show_batch_normalization_layer(layer):
    # momentum is only important for training
    assert layer.axis == -1
    return add_layer_def_dict(layer, {
        'type': 'BatchNormalization',
        'epsilon': layer.epsilon,
        'gamma': K.get_value(layer.gamma).tolist(),
        'beta': K.get_value(layer.beta).tolist(),
        'scale': layer.scale,
        'center': layer.center
        })

def show_dropout_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'dropout',
        'rate': layer.rate
        })

def show_maxpooling2d_layer(layer):
    assert layer.padding == 'valid'
    assert layer.strides == layer.pool_size
    return add_layer_def_dict(layer, {
        'type': 'MaxPooling2D',
        'pool_size': layer.pool_size
        })

def show_averagepooling2d_layer(layer):
    assert layer.padding == 'valid'
    assert layer.strides == layer.pool_size
    return add_layer_def_dict(layer, {
        'type': 'AveragePooling2D',
        'pool_size': layer.pool_size
        })

def show_upsampling2D_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'UpSampling2D',
        'size': layer.size
        })

def show_flatten_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'Flatten'
        })

def show_input_layer(layer):
    assert len(layer.input_shape) >= 2
    assert len(layer.inbound_nodes) == 1
    assert layer.input_shape[0] == None
    return add_layer_def_dict(layer, {
        'type': 'InputLayer',
        'input_shape': layer.input_shape[1:]
        })

def show_dense_layer(layer):
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
        })

def show_sigmoid_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'Sigmoid'
        })

def show_hard_sigmoid_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'HardSigmoid'
        })

def show_selu_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'SeLU'
        })

def show_softmax_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'Softmax'
        })

def show_softplus_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'Softplus'
        })

def show_tanh_layer(layer):
    return add_layer_def_dict(layer, {
        'type': 'Tanh'
        })

def show_concatenate_layer(layer):
    assert layer.axis == -1
    return add_layer_def_dict(layer, {
        'type': 'Concatenate',
        })

def show_activation_layer(layer):
    show_activation_functions = {
        'softmax': show_softmax_layer,
        'softplus': show_softplus_layer,
        'tanh': show_tanh_layer,
        'relu': show_relu_layer,
        'sigmoid': show_sigmoid_layer,
        'hard_sigmoid': show_hard_sigmoid_layer,
        'selu': show_selu_layer
    }
    return show_activation_functions[layer.get_config()['activation']](layer)

def show_model(model):
    result = {}
    result['type'] = 'model'
    result['name'] = model.name

    result['layers'] = []
    for layer in model.layers:
        layer_type = type(layer).__name__
#        if layer_type != 'Activation'\
#                and 'activation' in layer.get_config()\
#                and layer.get_config()['activation'] != 'linear':
#            assert False,\
#                layer.name + ': Please move Activation into a separate layer.'
        result['layers'].append(SHOW_FUNCTIONS[layer_type](layer))

    print(model.summary()) # todo remove

    result['input_layers'] = show_connections(
        model.input_layers,
        model.input_layers_node_indices,
        model.input_layers_tensor_indices)

    result['output_layers'] = show_connections(
        model.output_layers,
        model.output_layers_node_indices,
        model.output_layers_tensor_indices)

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

def test_model(model):
    print('predicting with dummy data')
    data_in = list(map(lambda l: np.random.random((1, *l.input_shape[1:])),
        model.input_layers))
    print(data_in)
    data_out = model.predict(data_in)
    print(data_out)

def main():
    if len(sys.argv) != 3:
        print('usage: [Keras model in HDF5 format] [output path]')
        sys.exit(1)
    else:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        model = load_model(in_path)
        write_text_file(out_path + '.yml', model.to_yaml()) # todo remove
        test_model(model)

        model_json = show_model(model)
        write_text_file(out_path,
            json.dumps(model_json, allow_nan=False, indent=2, sort_keys=True))
        

if __name__ == "__main__":
    main()
