#!/usr/bin/env python3
"""Convert a Keras model to frugally-deep format.
"""

import base64
import datetime
import hashlib
import json
import sys

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.models import Model, load_model

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"

STORE_FLOATS_HUMAN_READABLE = False


def transform_input_kernel(kernel):
    """Transforms weights of a single CuDNN input kernel into the regular Keras format."""
    return kernel.T.reshape(kernel.shape, order='F')


def transform_recurrent_kernel(kernel):
    """Transforms weights of a single CuDNN recurrent kernel into the regular Keras format."""
    return kernel.T


def transform_kernels(kernels, n_gates, transform_func):
    """
    Transforms CuDNN kernel matrices (either LSTM or GRU) into the regular Keras format.

    Parameters
    ----------
    kernels : numpy.ndarray
        Composite matrix of input or recurrent kernels.
    n_gates : int
        Number of recurrent unit gates, 3 for GRU, 4 for LSTM.
    transform_func: function(numpy.ndarray)
        Function to apply to each input or recurrent kernel.

    Returns
    -------
    numpy.ndarray
        Transformed composite matrix of input or recurrent kernels in C-contiguous layout.
    """
    return np.require(np.hstack([transform_func(kernel) for kernel in np.hsplit(kernels, n_gates)]), requirements='C')


def transform_bias(bias):
    """Transforms bias weights of an LSTM layer into the regular Keras format."""
    return np.sum(np.split(bias, 2, axis=0), axis=0)


def write_text_file(path, text):
    """Write a string to a file"""
    with open(path, "w") as text_file:
        print(text, file=text_file)


def int_or_none(value):
    """Leave None values as is, convert everything else to int"""
    if value is None:
        return value
    return int(value)


def keras_shape_to_fdeep_tensor_shape(raw_shape):
    """Convert a keras shape to an fdeep shape"""
    return singleton_list_to_value(raw_shape)[1:]


def get_layer_input_shape_tensor_shape(layer):
    """Convert layer input shape to an fdeep shape"""
    return keras_shape_to_fdeep_tensor_shape(layer.input_shape)


def show_tensor(tens):
    """Serialize 3-tensor to a dict"""
    return {
        'shape': tens.shape[1:],
        'values': encode_floats(tens.flatten())
    }


def get_model_input_layers(model):
    """Works for different Keras version."""
    if hasattr(model, '_input_layers'):
        return model._input_layers
    if hasattr(model, 'input_layers'):
        return model.input_layers
    raise ValueError('can not get (_)input_layers from model')


def measure_predict(model, data_in):
    """Returns output and duration in seconds"""
    start_time = datetime.datetime.now()
    data_out = model.predict(data_in)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print('Forward pass took {} s.'.format(duration.total_seconds()))
    return data_out, duration.total_seconds()


def replace_none_with(value, shape):
    """Replace every None with a fixed value."""
    return tuple(list(map(lambda x: x if x is not None else value, shape)))


def are_embedding_layer_positions_ok_for_testing(model):
    """
    Test data can only be generated if all embeddings layers
    are positioned directly behind the input nodes
    """

    def embedding_layer_names(model):
        layers = model.layers
        result = set()
        for layer in layers:
            if isinstance(layer, Embedding):
                result.add(layer.name)
        layer_type = type(layer).__name__
        if layer_type in ['Model', 'Sequential', 'Functional']:
            result.union(embedding_layer_names(layer))
        return result

    def embedding_layer_names_at_input_nodes(model):
        result = set()
        for input_layer in get_model_input_layers(model):
            if input_layer._outbound_nodes and isinstance(
                    input_layer._outbound_nodes[0].outbound_layer, Embedding):
                result.add(input_layer._outbound_nodes[0].outbound_layer.name)
        return set(result)

    return embedding_layer_names(model) == embedding_layer_names_at_input_nodes(model)


def gen_test_data(model):
    """Generate data for model verification test."""

    def set_shape_idx_0_to_1_if_none(shape):
        """Change first element in tuple to 1."""
        if shape[0] is not None:
            return shape
        shape_lst = list(shape)
        shape_lst[0] = 1
        shape = tuple(shape_lst)
        return shape

    def generate_input_data(input_layer):
        """Random data fitting the input shape of a layer."""
        if input_layer._outbound_nodes and isinstance(
                input_layer._outbound_nodes[0].outbound_layer, Embedding):
            random_fn = lambda size: np.random.randint(
                0, input_layer._outbound_nodes[0].outbound_layer.input_dim, size)
        else:
            random_fn = np.random.normal
        try:
            shape = input_layer.batch_input_shape
        except AttributeError:
            shape = input_layer.input_shape
        return random_fn(
            size=replace_none_with(32, set_shape_idx_0_to_1_if_none(singleton_list_to_value(shape)))).astype(np.float32)

    assert are_embedding_layer_positions_ok_for_testing(
        model), "Test data can only be generated if embedding layers are positioned directly after input nodes."

    data_in = list(map(generate_input_data, get_model_input_layers(model)))

    warm_up_runs = 3
    test_runs = 5
    for i in range(warm_up_runs):
        if i == 0:
            # store the results of first call for the test
            # this is because states of recurrent layers is 0.
            # cannot call model.reset_states() in some cases in keras without an error.
            # an error occurs when recurrent layer is stateful and the initial state is passed as input
            data_out_test, duration = measure_predict(model, data_in)
        else:
            measure_predict(model, data_in)
    duration_sum = 0
    print('Starting performance measurements.')
    for _ in range(test_runs):
        data_out, duration = measure_predict(model, data_in)
        duration_sum = duration_sum + duration
    duration_avg = duration_sum / test_runs
    print('Forward pass took {} s on average.'.format(duration_avg))
    return {
        'inputs': list(map(show_tensor, as_list(data_in))),
        'outputs': list(map(show_tensor, as_list(data_out_test)))
    }


def split_every(size, seq):
    """Split a sequence every seq elements."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def encode_floats(arr):
    """Serialize a sequence of floats."""
    if STORE_FLOATS_HUMAN_READABLE:
        return arr.flatten().tolist()
    return list(split_every(1024, base64.b64encode(arr).decode('ascii')))


def prepare_filter_weights_conv_2d(weights):
    """Change dimension order of 2d filter weights to the one used in fdeep"""
    assert len(weights.shape) == 4
    return np.moveaxis(weights, [0, 1, 2, 3], [1, 2, 3, 0]).flatten()


def prepare_filter_weights_slice_conv_2d(weights):
    """Change dimension order of 2d filter weights to the one used in fdeep"""
    assert len(weights.shape) == 4
    return np.moveaxis(weights, [0, 1, 2, 3], [1, 2, 0, 3]).flatten()


def prepare_filter_weights_conv_1d(weights):
    """Change dimension order of 1d filter weights to the one used in fdeep"""
    assert len(weights.shape) == 3
    return np.moveaxis(weights, [0, 1, 2], [1, 2, 0]).flatten()


def show_conv_1d_layer(layer):
    """Serialize Conv1D layer to dict"""
    weights = layer.get_weights()
    assert len(weights) == 1 or len(weights) == 2
    assert len(weights[0].shape) == 3
    weights_flat = prepare_filter_weights_conv_1d(weights[0])
    assert layer.padding in ['valid', 'same', 'causal']
    assert len(layer.input_shape) == 3
    assert layer.input_shape[0] in {None, 1}
    result = {
        'weights': encode_floats(weights_flat)
    }
    if len(weights) == 2:
        bias = weights[1]
        result['bias'] = encode_floats(bias)
    return result


def show_conv_2d_layer(layer):
    """Serialize Conv2D layer to dict"""
    weights = layer.get_weights()
    assert len(weights) == 1 or len(weights) == 2
    assert len(weights[0].shape) == 4
    weights_flat = prepare_filter_weights_conv_2d(weights[0])
    assert layer.padding in ['valid', 'same']
    assert len(layer.input_shape) == 4
    assert layer.input_shape[0] in {None, 1}
    result = {
        'weights': encode_floats(weights_flat)
    }
    if len(weights) == 2:
        bias = weights[1]
        result['bias'] = encode_floats(bias)
    return result


def show_separable_conv_2d_layer(layer):
    """Serialize SeparableConv2D layer to dict"""
    weights = layer.get_weights()
    assert layer.depth_multiplier == 1
    assert len(weights) == 2 or len(weights) == 3
    assert len(weights[0].shape) == 4
    assert len(weights[1].shape) == 4

    # probably incorrect for depth_multiplier > 1?
    slice_weights = prepare_filter_weights_slice_conv_2d(weights[0])
    stack_weights = prepare_filter_weights_conv_2d(weights[1])

    assert layer.padding in ['valid', 'same']
    assert len(layer.input_shape) == 4
    assert layer.input_shape[0] in {None, 1}
    result = {
        'slice_weights': encode_floats(slice_weights),
        'stack_weights': encode_floats(stack_weights),
    }
    if len(weights) == 3:
        bias = weights[2]
        result['bias'] = encode_floats(bias)
    return result


def show_depthwise_conv_2d_layer(layer):
    """Serialize DepthwiseConv2D layer to dict"""
    weights = layer.get_weights()
    assert layer.depth_multiplier == 1
    assert len(weights) in [1, 2]
    assert len(weights[0].shape) == 4

    # probably incorrect for depth_multiplier > 1?
    slice_weights = prepare_filter_weights_slice_conv_2d(weights[0])

    assert layer.padding in ['valid', 'same']
    assert len(layer.input_shape) == 4
    assert layer.input_shape[0] in {None, 1}
    result = {
        'slice_weights': encode_floats(slice_weights),
    }
    if len(weights) == 2:
        bias = weights[1]
        result['bias'] = encode_floats(bias)
    return result


def show_batch_normalization_layer(layer):
    """Serialize batch normalization layer to dict"""
    moving_mean = K.get_value(layer.moving_mean)
    moving_variance = K.get_value(layer.moving_variance)
    result = {}
    result['moving_mean'] = encode_floats(moving_mean)
    result['moving_variance'] = encode_floats(moving_variance)
    if layer.center:
        beta = K.get_value(layer.beta)
        result['beta'] = encode_floats(beta)
    if layer.scale:
        gamma = K.get_value(layer.gamma)
        result['gamma'] = encode_floats(gamma)
    return result


def show_dense_layer(layer):
    """Serialize dense layer to dict"""
    weights = layer.get_weights()
    assert len(weights) == 1 or len(weights) == 2
    assert len(weights[0].shape) == 2
    weights_flat = weights[0].flatten()
    result = {
        'weights': encode_floats(weights_flat)
    }
    if len(weights) == 2:
        bias = weights[1]
        result['bias'] = encode_floats(bias)
    return result


def show_prelu_layer(layer):
    """Serialize prelu layer to dict"""
    weights = layer.get_weights()
    assert len(weights) == 1
    weights_flat = weights[0].flatten()
    result = {
        'alpha': encode_floats(weights_flat)
    }
    return result


def show_relu_layer(layer):
    """Serialize relu layer to dict"""
    assert layer.negative_slope == 0
    assert layer.threshold == 0
    return {}


def show_embedding_layer(layer):
    """Serialize Embedding layer to dict"""
    weights = layer.get_weights()
    assert len(weights) == 1
    result = {
        'weights': encode_floats(weights[0])
    }
    return result


def show_lstm_layer(layer):
    """Serialize LSTM layer to dict"""
    assert not layer.go_backwards
    assert not layer.unroll
    weights = layer.get_weights()
    if isinstance(layer.input, list):
        assert len(layer.input) in [1, 3]
    assert len(weights) == 2 or len(weights) == 3
    result = {'weights': encode_floats(weights[0]),
              'recurrent_weights': encode_floats(weights[1])}

    if len(weights) == 3:
        result['bias'] = encode_floats(weights[2])

    return result


def show_gru_layer(layer):
    """Serialize GRU layer to dict"""
    assert not layer.go_backwards
    assert not layer.unroll
    assert not layer.return_state
    weights = layer.get_weights()
    assert len(weights) == 2 or len(weights) == 3
    result = {'weights': encode_floats(weights[0]),
              'recurrent_weights': encode_floats(weights[1])}

    if len(weights) == 3:
        result['bias'] = encode_floats(weights[2])

    return result


def transform_cudnn_weights(input_weights, recurrent_weights, n_gates):
    return transform_kernels(input_weights, n_gates, transform_input_kernel), \
           transform_kernels(recurrent_weights, n_gates, transform_recurrent_kernel)


def show_cudnn_lstm_layer(layer):
    """Serialize a GPU-trained LSTM layer to dict"""
    weights = layer.get_weights()
    if isinstance(layer.input, list):
        assert len(layer.input) in [1, 3]
    assert len(weights) == 3  # CuDNN LSTM always has a bias

    n_gates = 4
    input_weights, recurrent_weights = transform_cudnn_weights(weights[0], weights[1], n_gates)

    result = {'weights': encode_floats(input_weights),
              'recurrent_weights': encode_floats(recurrent_weights),
              'bias': encode_floats(transform_bias(weights[2]))}

    return result


def show_cudnn_gru_layer(layer):
    """Serialize a GPU-trained GRU layer to dict"""
    weights = layer.get_weights()
    assert len(weights) == 3  # CuDNN GRU always has a bias

    n_gates = 3
    input_weights, recurrent_weights = transform_cudnn_weights(weights[0], weights[1], n_gates)

    result = {'weights': encode_floats(input_weights),
              'recurrent_weights': encode_floats(recurrent_weights),
              'bias': encode_floats(weights[2])}

    return result


def get_transform_func(layer):
    """Returns functions that can be applied to layer weights to transform them into the standard Keras format, if applicable."""
    if layer.__class__.__name__ in ['CuDNNGRU', 'CuDNNLSTM']:
        if layer.__class__.__name__ == 'CuDNNGRU':
            n_gates = 3
        elif layer.__class__.__name__ == 'CuDNNLSTM':
            n_gates = 4

        input_transform_func = lambda kernels: transform_kernels(kernels, n_gates, transform_input_kernel)
        recurrent_transform_func = lambda kernels: transform_kernels(kernels, n_gates, transform_recurrent_kernel)
    else:
        input_transform_func = lambda kernels: kernels
        recurrent_transform_func = lambda kernels: kernels

    if layer.__class__.__name__ == 'CuDNNLSTM':
        bias_transform_func = transform_bias
    else:
        bias_transform_func = lambda bias: bias

    return input_transform_func, recurrent_transform_func, bias_transform_func


def show_bidirectional_layer(layer):
    """Serialize Bidirectional layer to dict"""
    forward_weights = layer.forward_layer.get_weights()
    assert len(forward_weights) == 2 or len(forward_weights) == 3
    forward_input_transform_func, forward_recurrent_transform_func, forward_bias_transform_func = get_transform_func(
        layer.forward_layer)

    backward_weights = layer.backward_layer.get_weights()
    assert len(backward_weights) == 2 or len(backward_weights) == 3
    backward_input_transform_func, backward_recurrent_transform_func, backward_bias_transform_func = get_transform_func(
        layer.backward_layer)

    result = {'forward_weights': encode_floats(forward_input_transform_func(forward_weights[0])),
              'forward_recurrent_weights': encode_floats(forward_recurrent_transform_func(forward_weights[1])),
              'backward_weights': encode_floats(backward_input_transform_func(backward_weights[0])),
              'backward_recurrent_weights': encode_floats(backward_recurrent_transform_func(backward_weights[1]))}

    if len(forward_weights) == 3:
        result['forward_bias'] = encode_floats(forward_bias_transform_func(forward_weights[2]))
    if len(backward_weights) == 3:
        result['backward_bias'] = encode_floats(backward_bias_transform_func(backward_weights[2]))

    return result


def show_input_layer(layer):
    """Serialize input layer to dict"""
    assert not layer.sparse
    return {}


def show_softmax_layer(layer):
    """Serialize softmax layer to dict"""
    assert layer.axis == -1


def show_reshape_layer(layer):
    """Serialize reshape layer to dict"""
    for dim_size in layer.target_shape:
        assert dim_size != -1, 'Reshape inference not supported'


def get_layer_functions_dict():
    return {
        'Conv1D': show_conv_1d_layer,
        'Conv2D': show_conv_2d_layer,
        'SeparableConv2D': show_separable_conv_2d_layer,
        'DepthwiseConv2D': show_depthwise_conv_2d_layer,
        'BatchNormalization': show_batch_normalization_layer,
        'Dense': show_dense_layer,
        'PReLU': show_prelu_layer,
        'ReLU': show_relu_layer,
        'Embedding': show_embedding_layer,
        'LSTM': show_lstm_layer,
        'GRU': show_gru_layer,
        'CuDNNLSTM': show_cudnn_lstm_layer,
        'CuDNNGRU': show_cudnn_gru_layer,
        'Bidirectional': show_bidirectional_layer,
        'TimeDistributed': show_time_distributed_layer,
        'Input': show_input_layer,
        'Softmax': show_softmax_layer
    }


def show_time_distributed_layer(layer):
    show_layer_functions = get_layer_functions_dict()
    config = layer.get_config()
    class_name = config['layer']['class_name']

    if class_name in show_layer_functions:

        if len(layer.input_shape) == 3:
            input_shape_new = (layer.input_shape[0], layer.input_shape[2])
        elif len(layer.input_shape) == 4:
            input_shape_new = (layer.input_shape[0], layer.input_shape[2], layer.input_shape[3])
        elif len(layer.input_shape) == 5:
            input_shape_new = (layer.input_shape[0], layer.input_shape[2], layer.input_shape[3], layer.input_shape[4])
        elif len(layer.input_shape) == 6:
            input_shape_new = (layer.input_shape[0], layer.input_shape[2], layer.input_shape[3], layer.input_shape[4],
                               layer.input_shape[5])
        else:
            raise Exception('Wrong input shape')

        layer_function = show_layer_functions[class_name]
        attributes = dir(layer.layer)

        class CopiedLayer:
            pass

        copied_layer = CopiedLayer()

        for attr in attributes:
            try:
                if attr not in ['input_shape', '__class__']:
                    setattr(copied_layer, attr, getattr(layer.layer, attr))
                elif attr == 'input_shape':
                    setattr(copied_layer, 'input_shape', input_shape_new)
            except Exception:
                continue

        setattr(copied_layer, "output_shape", getattr(layer, "output_shape"))

        return layer_function(copied_layer)

    else:
        return None


def get_dict_keys(d):
    """Return keys of a dictionary"""
    return [key for key in d]


def merge_two_disjunct_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy.
    No Key is allowed to be present in both dictionaries.
    """
    assert set(get_dict_keys(x)).isdisjoint(get_dict_keys(y))
    z = x.copy()
    z.update(y)
    return z


def is_ascii(some_string):
    """Check if a string only contains ascii characters"""
    try:
        some_string.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True


def get_all_weights(model, prefix):
    """Serialize all weights of the models layers"""
    show_layer_functions = get_layer_functions_dict()
    result = {}
    layers = model.layers
    assert K.image_data_format() == 'channels_last'
    for layer in layers:
        layer_type = type(layer).__name__
        name = prefix + layer.name
        assert is_ascii(name)
        if name in result:
            raise ValueError('duplicate layer name ' + name)
        if layer_type in ['Model', 'Sequential', 'Functional']:
            result = merge_two_disjunct_dicts(result, get_all_weights(layer, name + '_'))
        else:
            if hasattr(layer, 'data_format'):
                if layer_type in ['AveragePooling1D', 'MaxPooling1D', 'AveragePooling2D', 'MaxPooling2D',
                                  'GlobalAveragePooling1D', 'GlobalMaxPooling1D', 'GlobalAveragePooling2D',
                                  'GlobalMaxPooling2D']:
                    assert layer.data_format == 'channels_last' or layer.data_format == 'channels_first'
                else:
                    assert layer.data_format == 'channels_last'

            show_func = show_layer_functions.get(layer_type, None)
            shown_layer = None
            if show_func:
                shown_layer = show_func(layer)
            if shown_layer:
                result[name] = shown_layer
            if show_func and layer_type == 'TimeDistributed':
                if name not in result:
                    result[name] = {}

                result[name]['td_input_len'] = encode_floats(np.array([len(layer.input_shape) - 1], dtype=np.float32))
                result[name]['td_output_len'] = encode_floats(np.array([len(layer.output_shape) - 1], dtype=np.float32))
    return result


def get_model_name(model):
    """Return .name or ._name or 'dummy_model_name'"""
    if hasattr(model, 'name'):
        return model.name
    if hasattr(model, '_name'):
        return model._name
    return 'dummy_model_name'


def convert_sequential_to_model(model):
    """Convert a sequential model to the underlying functional format"""
    if type(model).__name__ == 'Sequential':
        name = get_model_name(model)
        if hasattr(model, '_inbound_nodes'):
            inbound_nodes = model._inbound_nodes
        elif hasattr(model, 'inbound_nodes'):
            inbound_nodes = model.inbound_nodes
        else:
            raise ValueError('can not get (_)inbound_nodes from model')
        input_layer = Input(batch_shape=model.layers[0].input_shape)
        prev_layer = input_layer
        for layer in model.layers:
            layer._inbound_nodes = []
            prev_layer = layer(prev_layer)
        funcmodel = Model([input_layer], [prev_layer], name=name)
        model = funcmodel
        if hasattr(model, '_inbound_nodes'):
            model._inbound_nodes = inbound_nodes
        elif hasattr(model, 'inbound_nodes'):
            model.inbound_nodes = inbound_nodes
    assert model.layers
    for i in range(len(model.layers)):
        layer_type = type(model.layers[i]).__name__
        if layer_type in ['Model', 'Sequential', 'Functional']:
            new_layer = convert_sequential_to_model(model.layers[i])
            layers = getattr(model, '_layers', None)
            if not layers:
                layers = getattr(model, '_self_tracked_trackables', None)
            if layers:
                layers[i] = new_layer
                assert model.layers[i] == new_layer
    return model


def offset_conv2d_eval(depth, padding, x):
    """Perform a conv2d on x with a given padding"""
    kernel = K.variable(value=np.array([[[[1]] + [[0]] * (depth - 1)]]),
                        dtype='float32')
    return K.conv2d(x, kernel, strides=(3, 3), padding=padding)


def offset_sep_conv2d_eval(depth, padding, x):
    """Perform a separable conv2d on x with a given padding"""
    depthwise_kernel = K.variable(value=np.array([[[[1]] * depth]]),
                                  dtype='float32')
    pointwise_kernel = K.variable(value=np.array([[[[1]] + [[0]] * (depth - 1)]]),
                                  dtype='float32')
    return K.separable_conv2d(x, depthwise_kernel,
                              pointwise_kernel, strides=(3, 3), padding=padding)


def conv2d_offset_max_pool_eval(_, padding, x):
    """Perform a max pooling operation on x"""
    return K.pool2d(x, (1, 1), strides=(3, 3), padding=padding, pool_mode='max')


def conv2d_offset_average_pool_eval(_, padding, x):
    """Perform an average pooling operation on x"""
    return K.pool2d(x, (1, 1), strides=(3, 3), padding=padding, pool_mode='avg')


def check_operation_offset(depth, eval_f, padding):
    """Check if backend used an offset while placing the filter
    e.g. during a convolution.
    TensorFlow is inconsistent in doing so depending
    on the type of operation, the used device (CPU/GPU) and the input depth.
    """
    in_arr = np.array([[[[i] * depth for i in range(6)]]])
    input_data = K.variable(value=in_arr, dtype='float32')
    output = eval_f(depth, padding, input_data)
    result = K.eval(output).flatten().tolist()
    assert result in [[0, 3], [1, 4]]
    return result == [1, 4]


def get_shapes(tensors):
    """Return shapes of a list of tensors"""
    return [t['shape'] for t in tensors]


def calculate_hash(model):
    layers = model.layers
    hash_m = hashlib.sha256()
    for layer in layers:
        for weights in layer.get_weights():
            assert isinstance(weights, np.ndarray)
            hash_m.update(weights.tobytes())
        hash_m.update(layer.name.encode('ascii'))
    return hash_m.hexdigest()


def as_list(value_or_values):
    """Leave lists untouched, convert non-list types to a singleton list"""
    if isinstance(value_or_values, list):
        return value_or_values
    return [value_or_values]


def singleton_list_to_value(value_or_values):
    """
    Leaves non-list values untouched.
    Raises an Exception in case the input list does not have exactly one element.
    """
    if isinstance(value_or_values, list):
        assert len(value_or_values) == 1
        return value_or_values[0]
    return value_or_values


def model_to_fdeep_json(model, no_tests=False):
    """Convert any Keras model to the frugally-deep model format."""

    # Force creation of underlying functional model.
    # see: https://github.com/fchollet/keras/issues/8136
    # Loss and optimizer type do not matter, since we do not train the model.
    model.compile(loss='mse', optimizer='sgd')

    model = convert_sequential_to_model(model)

    test_data = None if no_tests else gen_test_data(model)

    json_output = {}
    print('Converting model architecture.')
    json_output['architecture'] = json.loads(model.to_json())
    json_output['image_data_format'] = K.image_data_format()
    json_output['input_shapes'] = list(map(get_layer_input_shape_tensor_shape, get_model_input_layers(model)))
    json_output['output_shapes'] = list(map(keras_shape_to_fdeep_tensor_shape, as_list(model.output_shape)))

    if test_data:
        json_output['tests'] = [test_data]

    print('Converting model weights.')
    json_output['trainable_params'] = get_all_weights(model, '')
    print('Done converting model weights.')

    print('Calculating model hash.')
    json_output['hash'] = calculate_hash(model)
    print('Model conversion finished.')

    return json_output


def convert(in_path, out_path, no_tests=False):
    """Convert any (h5-)stored Keras model to the frugally-deep model format."""

    print('loading {}'.format(in_path))
    model = load_model(in_path)
    json_output = model_to_fdeep_json(model, no_tests)
    print('writing {}'.format(out_path))
    write_text_file(out_path, json.dumps(
        json_output, allow_nan=False, indent=2, sort_keys=True))


def main():
    """Parse command line and convert model."""

    usage = 'usage: [Keras model in HDF5 format] [output path] (--no-tests)'

    # todo: Use ArgumentParser instead.
    if len(sys.argv) not in [3, 4]:
        print(usage)
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    no_tests = False
    if len(sys.argv) == 4:
        if sys.argv[3] not in ['--no-tests']:
            print(usage)
            sys.exit(1)
        if sys.argv[3] == '--no-tests':
            no_tests = True

    convert(in_path, out_path, no_tests)


if __name__ == "__main__":
    main()
