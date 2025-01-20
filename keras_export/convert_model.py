#!/usr/bin/env python3
"""Convert a Keras model to frugally-deep format.
"""

import argparse
import base64
import datetime
import hashlib
import json

import numpy as np
from keras import backend as K
from keras.layers import Input, Embedding, CategoryEncoding
from keras.models import Model, load_model

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"

STORE_FLOATS_HUMAN_READABLE = False


def int_or_none(value):
    """Leave None values as is, convert everything else to int"""
    if value is None:
        return value
    return int(value)


def keras_shape_to_fdeep_tensor_shape(raw_shape):
    """Convert a keras shape to an fdeep shape"""
    return singleton_list_to_value(raw_shape)[1:]


def get_layer_input_shape(layer):
    """It is stored in a different property depending on the situation."""
    if hasattr(layer, "batch_shape"):
        return layer.batch_shape
    return layer.input.shape


def get_layer_input_shape_tensor_shape(layer):
    """Convert layer input shape to an fdeep shape"""
    return keras_shape_to_fdeep_tensor_shape(get_layer_input_shape(layer))


def show_tensor(tens):
    """Serialize 3-tensor to a dict"""
    return {
        'shape': tens.shape[1:],
        'values': encode_floats(tens.flatten())
    }


def get_model_input_layers(model):
    """Gets the input layers from model.layers in the correct input order."""
    if len(model.inputs) == 1:
        from keras.src.layers.core.input_layer import InputLayer
        input_layers = []
        for layer in model.layers:
            if isinstance(layer, InputLayer):
                input_layers.append(layer)
            return input_layers
    input_layer_names = [model_input.name for model_input in model.inputs]
    model_layers = {layer.name: layer for layer in model.layers}
    return [model_layers[layer_names] for layer_names in input_layer_names]


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


def get_first_outbound_op(layer):
    """Determine primary outbound operation"""
    return layer._outbound_nodes[0].operation


def are_embedding_and_category_encoding_layer_positions_ok_for_testing(model):
    """
    Test data can only be generated if all Embedding layers
    and CategoryEncoding layers are positioned directly behind the input nodes.
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
            if input_layer._outbound_nodes and (
                    isinstance(get_first_outbound_op(input_layer), Embedding) or
                    isinstance(get_first_outbound_op(input_layer), CategoryEncoding)):
                result.add(get_first_outbound_op(input_layer).name)
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
        print("input input_layer type", type(input_layer).__name__)  # todo: remove
        print("input_layer._outbound_nodes type", type(input_layer._outbound_nodes).__name__)  # todo: remove
        if input_layer._outbound_nodes and isinstance(
                get_first_outbound_op(input_layer), Embedding):
            random_fn = lambda size: np.random.randint(
                0, get_first_outbound_op(input_layer).input_dim, size)
        elif input_layer._outbound_nodes and isinstance(
                get_first_outbound_op(input_layer), CategoryEncoding):
            random_fn = lambda size: np.random.randint(
                0, get_first_outbound_op(input_layer).num_tokens, size)
        else:
            random_fn = np.random.normal
        shape = get_layer_input_shape(input_layer)
        return random_fn(
            size=replace_none_with(32, set_shape_idx_0_to_1_if_none(singleton_list_to_value(shape)))).astype(np.float32)

    assert are_embedding_and_category_encoding_layer_positions_ok_for_testing(
        model), "Test data can only be generated if embedding layers are positioned directly after input nodes."

    data_in = list(map(generate_input_data, get_model_input_layers(model)))

    warm_up_runs = 3
    test_runs = 5
    for i in range(warm_up_runs):
        if i == 0:
            data_out_test, _ = measure_predict(model, data_in)
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
    assert len(get_layer_input_shape(layer)) == 3
    assert get_layer_input_shape(layer)[0] in {None, 1}
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
    assert len(get_layer_input_shape(layer)) == 4
    assert get_layer_input_shape(layer)[0] in {None, 1}
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
    assert len(get_layer_input_shape(layer)) == 4
    assert get_layer_input_shape(layer)[0] in {None, 1}
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
    assert len(get_layer_input_shape(layer)) == 4
    assert get_layer_input_shape(layer)[0] in {None, 1}
    result = {
        'slice_weights': encode_floats(slice_weights),
    }
    if len(weights) == 2:
        bias = weights[1]
        result['bias'] = encode_floats(bias)
    return result


def show_batch_normalization_layer(layer):
    """Serialize batch normalization layer to dict"""
    moving_mean = layer.moving_mean.numpy()
    moving_variance = layer.moving_variance.numpy()
    result = {}
    result['moving_mean'] = encode_floats(moving_mean)
    result['moving_variance'] = encode_floats(moving_variance)
    if layer.center:
        beta = layer.beta.numpy()
        result['beta'] = encode_floats(beta)
    if layer.scale:
        gamma = layer.gamma.numpy()
        result['gamma'] = encode_floats(gamma)
    return result


def show_layer_normalization_layer(layer):
    """Serialize layer normalization layer to dict"""
    result = {}
    if layer.center:
        beta = layer.beta.numpy()
        result['beta'] = encode_floats(beta)
    if layer.scale:
        gamma = layer.gamma.numpy()
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


def show_dot_layer(layer):
    """Check valid configuration of Dot layer"""
    assert len(get_layer_input_shape(layer)) == 2
    assert isinstance(layer.axes, int) or (isinstance(layer.axes, list) and len(layer.axes) == 2)
    assert get_layer_input_shape(layer)[0][0] is None
    assert get_layer_input_shape(layer)[1][0] is None
    assert len(layer.output_shape) <= 5


def show_prelu_layer(layer):
    """Serialize prelu layer to dict"""
    weights = layer.get_weights()
    assert len(weights) == 1
    weights_flat = weights[0].flatten()
    result = {
        'alpha': encode_floats(weights_flat)
    }
    return result


def show_embedding_layer(layer):
    """Serialize Embedding layer to dict"""
    weights = layer.get_weights()
    assert len(weights) == 1
    result = {
        'weights': encode_floats(weights[0])
    }
    return result


def show_input_layer(layer):
    """Serialize input layer to dict"""
    assert not layer.sparse


def show_softmax_layer(layer):
    """Serialize softmax layer to dict"""
    assert layer.axis == -1


def show_normalization_layer(layer):
    """Serialize normalization layer to dict"""
    assert len(layer.axis) <= 1, "Multiple normalization axes are not supported"
    if len(layer.axis) == 1:
        assert layer.axis[0] in (-1, 1, 2, 3, 4, 5), "Invalid axis for Normalization layer."
    return {
        'mean': encode_floats(layer.mean),
        'variance': encode_floats(layer.variance)
    }


def show_upsampling2d_layer(layer):
    """Serialize UpSampling2D layer to dict"""
    assert layer.interpolation in ["nearest", "bilinear"]


def show_resizing_layer(layer):
    """Serialize Resizing layer to dict"""
    assert layer.interpolation in ["nearest", "bilinear", "area"]


def show_rescaling_layer(layer):
    """Serialize Rescaling layer to dict"""
    assert isinstance(layer.scale, float)


def show_category_encoding_layer(layer):
    """Serialize CategoryEncoding layer to dict"""
    assert layer.output_mode in ["multi_hot", "count", "one_hot"]


def show_attention_layer(layer):
    """Serialize Attention layer to dict"""
    assert layer.score_mode in ["dot", "concat"]
    data = {}
    if layer.scale is not None:
        data['scale'] = float(layer.scale.numpy())
    if layer.score_mode == "concat":
        data['concat_score_weight'] = float(layer.concat_score_weight.numpy())
    if data:
        return data


def show_additive_attention_layer(layer):
    """Serialize AdditiveAttention layer to dict"""
    data = {}
    if layer.scale is not None:
        data['scale'] = encode_floats(layer.scale.numpy())
    if data:
        return data


def show_multi_head_attention_layer(layer):
    """Serialize MultiHeadAttention layer to dict"""
    assert layer._output_shape is None
    assert layer._attention_axes == (1,), "MultiHeadAttention supported only with attention_axes=None"
    return {
        'weight_shapes': list(map(lambda w: list(w.shape), layer.weights)),
        'weights': list(map(lambda w: encode_floats(w.numpy()), layer.weights)),
    }


def get_layer_functions_dict():
    return {
        'Conv1D': show_conv_1d_layer,
        'Conv2D': show_conv_2d_layer,
        'SeparableConv2D': show_separable_conv_2d_layer,
        'DepthwiseConv2D': show_depthwise_conv_2d_layer,
        'BatchNormalization': show_batch_normalization_layer,
        'Dense': show_dense_layer,
        'Dot': show_dot_layer,
        'PReLU': show_prelu_layer,
        'Embedding': show_embedding_layer,
        'LayerNormalization': show_layer_normalization_layer,
        'TimeDistributed': show_time_distributed_layer,
        'Input': show_input_layer,
        'Softmax': show_softmax_layer,
        'Normalization': show_normalization_layer,
        'UpSampling2D': show_upsampling2d_layer,
        'Resizing': show_resizing_layer,
        'Rescaling': show_rescaling_layer,
        'CategoryEncoding': show_category_encoding_layer,
        'Attention': show_attention_layer,
        'AdditiveAttention': show_additive_attention_layer,
        'MultiHeadAttention': show_multi_head_attention_layer,
    }


def show_time_distributed_layer(layer):
    show_layer_functions = get_layer_functions_dict()
    config = layer.get_config()
    class_name = config['layer']['class_name']

    if class_name in show_layer_functions:

        if len(get_layer_input_shape(layer)) == 3:
            input_shape_new = (get_layer_input_shape(layer)[0], get_layer_input_shape(layer)[2])
        elif len(get_layer_input_shape(layer)) == 4:
            input_shape_new = (
                get_layer_input_shape(layer)[0], get_layer_input_shape(layer)[2], get_layer_input_shape(layer)[3])
        elif len(get_layer_input_shape(layer)) == 5:
            input_shape_new = (
                get_layer_input_shape(layer)[0], get_layer_input_shape(layer)[2], get_layer_input_shape(layer)[3],
                get_layer_input_shape(layer)[4])
        elif len(get_layer_input_shape(layer)) == 6:
            input_shape_new = (
                get_layer_input_shape(layer)[0], get_layer_input_shape(layer)[2], get_layer_input_shape(layer)[3],
                get_layer_input_shape(layer)[4],
                get_layer_input_shape(layer)[5])
        else:
            raise Exception('Wrong input shape')

        layer_function = show_layer_functions[class_name]
        attributes = dir(layer.layer)

        class CopiedLayer:
            pass

        copied_layer = CopiedLayer()

        for attr in attributes:
            try:
                if attr not in ['batch_shape', '__class__']:
                    setattr(copied_layer, attr, getattr(layer.layer, attr))
            except Exception:
                continue

        setattr(copied_layer, 'batch_shape', input_shape_new)
        setattr(copied_layer, "output_shape", layer.output.shape)

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


def get_layer_weights(layer, name):
    """Serialize all weights of a single normal layer"""
    result = {}
    layer_type = type(layer).__name__
    if hasattr(layer, 'data_format'):
        assert layer.data_format == 'channels_last'

    show_func = get_layer_functions_dict().get(layer_type, None)
    shown_layer = None
    if show_func:
        shown_layer = show_func(layer)
    if shown_layer:
        result[name] = shown_layer
    if show_func and layer_type == 'TimeDistributed':
        if name not in result:
            result[name] = {}

        result[name]['td_input_len'] = encode_floats(
            np.array([len(get_layer_input_shape(layer)) - 1], dtype=np.float32))
        result[name]['td_output_len'] = encode_floats(np.array([len(layer.output.shape) - 1], dtype=np.float32))
    return result


def get_all_weights(model, prefix):
    """Serialize all weights of the models layers"""
    result = {}
    layers = model.layers
    assert K.image_data_format() == 'channels_last'
    for layer in layers:
        layer_type = type(layer).__name__
        for node in layer._inbound_nodes:
            if "training" in node.arguments.kwargs:
                is_layer_with_accidental_training_flag = layer_type in ("CenterCrop", "Resizing")
                has_training = node.arguments.kwargs["training"] is True
                assert not has_training or is_layer_with_accidental_training_flag, \
                    "training=true is not supported, see https://github.com/Dobiasd/frugally-deep/issues/284"

        name = prefix + layer.name
        assert is_ascii(name)
        if name in result:
            raise ValueError('duplicate layer name ' + name)
        if layer_type in ['Model', 'Sequential', 'Functional']:
            result = merge_two_disjunct_dicts(result, get_all_weights(layer, name + '_'))
        elif layer_type in ['TimeDistributed'] and type(layer.layer).__name__ in ['Model', 'Sequential', 'Functional']:
            inner_layer = layer.layer
            result = merge_two_disjunct_dicts(result, get_layer_weights(layer, name))
            result = merge_two_disjunct_dicts(result, get_all_weights(inner_layer, name + "_"))
        else:
            result = merge_two_disjunct_dicts(result, get_layer_weights(layer, name))
    return result


def get_model_name(model):
    """Return .name or ._name"""
    if hasattr(model, 'name'):
        return model.name
    return model._name


def convert_sequential_to_model(model):
    """Convert a sequential model to the underlying functional format"""
    if type(model).__name__ in ['Sequential']:
        name = get_model_name(model)
        inbound_nodes = model._inbound_nodes
        input_layer = Input(batch_shape=get_layer_input_shape(model.layers[0]))
        prev_layer = input_layer
        for layer in model.layers:
            layer._inbound_nodes = []
            prev_layer = layer(prev_layer)
        funcmodel = Model([input_layer], [prev_layer], name=name)
        model = funcmodel
        model._inbound_nodes = inbound_nodes
    if type(model).__name__ == 'TimeDistributed':
        model.layer = convert_sequential_to_model(model.layer)
    if type(model).__name__ in ['Model', 'Functional']:
        for i in range(len(model.layers)):
            new_layer = convert_sequential_to_model(model.layers[i])
            if new_layer == model.layers[i]:
                continue
            # https://stackoverflow.com/questions/78297541/how-to-replace-a-model-layer-using-tensorflow-2-16
            model._operations[i] = new_layer
            assert model.layers[i] == new_layer
    return model


def get_shapes(tensors):
    """Return shapes of a list of tensors"""
    return [t['shape'] for t in tensors]


def calculate_hash(model):
    layers = model.layers
    hash_m = hashlib.sha256()
    for layer in layers:
        for weights in layer.get_weights():
            if isinstance(weights, np.ndarray):
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


def assert_model_type(model):
    import keras
    assert type(model) in [keras.src.models.sequential.Sequential, keras.src.models.functional.Functional]


def convert(in_path, out_path, no_tests=False):
    """Convert any (h5-)stored Keras model to the frugally-deep model format."""
    print('loading {}'.format(in_path))
    model = load_model(in_path, compile=False)
    json_output = model_to_fdeep_json(model, no_tests)
    print('writing {}'.format(out_path))

    with open(out_path, 'w') as f:
        json.dump(json_output, f, allow_nan=False, separators=(',', ':'))


def main():
    """Parse command line and convert model."""

    parser = argparse.ArgumentParser(
        prog='frugally-deep model converter',
        description='Converts models from Keras\' .keras format to frugally-deep\'s .json format.')
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--no-tests', action='store_true')
    args = parser.parse_args()

    convert(args.input_path, args.output_path, args.no_tests)


if __name__ == "__main__":
    main()
