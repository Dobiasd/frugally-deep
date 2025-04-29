#!/usr/bin/env python3
"""Generate a test model for frugally-deep.
"""

import sys
from typing import Tuple, List, Union

import numpy as np
import tensorflow as tf
import keras
from keras.layers import ActivityRegularization
from keras.layers import AdditiveAttention
from keras.layers import Attention
from keras.layers import BatchNormalization, Concatenate, LayerNormalization, UnitNormalization
from keras.layers import CategoryEncoding, Embedding
from keras.layers import Conv1D, ZeroPadding1D, Cropping1D
from keras.layers import Conv2D, ZeroPadding2D, Cropping2D, CenterCrop
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D
from keras.layers import Identity, Conv2DTranspose, Conv1DTranspose
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import LeakyReLU, ELU, PReLU, ReLU, Softmax
from keras.layers import MaxPooling1D, AveragePooling1D, UpSampling1D
from keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.layers import MaxPooling3D, AveragePooling3D
from keras.layers import MultiHeadAttention
from keras.layers import Multiply, Add, Subtract, Average, Maximum, Minimum
from keras.layers import Normalization, Rescaling, Resizing
from keras.layers import Permute, Reshape, RepeatVector
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.layers import ZeroPadding3D, Cropping3D
from keras.models import Model, load_model, Sequential

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"

NDFloat32Array = np.typing.NDArray[np.float32]
NDUInt32Array = np.typing.NDArray[np.int32]
Shape = Tuple[int, ...]
VariableShape = Tuple[Union[None, int], ...]


def replace_none_with(value: int, shape: VariableShape) -> Shape:
    """Replace every None with a fixed value."""
    return tuple(list(map(lambda x: x if x is not None else value, shape)))


def get_shape_for_random_data(data_size: int, shape: Shape) -> Shape:
    """Include size of data to generate into shape."""
    if len(shape) == 5:
        return (data_size, shape[0], shape[1], shape[2], shape[3], shape[4])
    if len(shape) == 4:
        return (data_size, shape[0], shape[1], shape[2], shape[3])
    if len(shape) == 3:
        return (data_size, shape[0], shape[1], shape[2])
    if len(shape) == 2:
        return (data_size, shape[0], shape[1])
    if len(shape) == 1:
        return (data_size, shape[0])
    raise ValueError('can not use shape for random data:', shape)


def generate_random_data(data_size: int, shape: VariableShape) -> NDFloat32Array:
    """Random data for training."""
    return np.random.random(
        size=get_shape_for_random_data(data_size, replace_none_with(42, shape))).astype(np.float32)


def generate_input_data(data_size: int, input_shapes: List[VariableShape]) -> List[NDFloat32Array]:
    """Random input data for training."""
    return [generate_random_data(data_size, input_shape)
            for input_shape in input_shapes]


def generate_integer_random_data(data_size: int, low: int, high: int, shape: Shape) -> NDUInt32Array:
    """Random data for training."""
    return np.random.randint(
        low=low, high=high, size=get_shape_for_random_data(data_size, replace_none_with(42, shape)))


def generate_integer_input_data(data_size: int, low: int, highs: List[int], input_shapes: List[Shape]) -> List[
    NDUInt32Array]:
    """Random input data for training."""
    return [generate_integer_random_data(data_size, low, high, input_shape)
            for high, input_shape in zip(highs, input_shapes)]


def as_list(value_or_values: Union[NDFloat32Array, List[NDFloat32Array]]) -> List[NDFloat32Array]:
    """Leave lists untouched, convert non-list types to a singleton list"""
    if isinstance(value_or_values, list):
        return value_or_values
    return [value_or_values]


def generate_output_data(data_size: int, outputs: List[NDFloat32Array]) -> List[NDFloat32Array]:
    """Random output data for training."""
    return [generate_random_data(data_size, output.shape[1:])
            for output in as_list(outputs)]


def get_test_model_exhaustive() -> Model:
    """Returns a exhaustive test model."""
    input_shapes: List[VariableShape] = [
        (2,),  # 10
    ]

    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []

    input_norm_layer = Normalization()
    input_norm_layer.adapt(np.array([[0.1, 0.2]]))
    outputs.append((inputs[0] - input_norm_layer.mean) / tf.sqrt(input_norm_layer.variance))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_exhaustive')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 2
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=10)
    return model


def get_test_model_embedding() -> Model:
    """Returns a minimalistic test model for the Embedding and CategoryEncoding layers."""

    input_dims = [
        1023,  # maximum integer value in input data
        255,
        15,
    ]
    input_shapes: list[tuple[int, ...]] = [
        (100,),  # must be single-element tuple (for sequence length)
        (1000,),
        (1,),
    ]
    assert len(input_dims) == len(input_shapes)
    output_dims = [8, 3]  # embedding dimension

    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []
    for k in range(2):
        embedding = Embedding(input_dim=input_dims[k], output_dim=output_dims[k])(inputs[k])
        outputs.append(embedding)

    outputs.append(CategoryEncoding(1024, output_mode='multi_hot', sparse=False)(inputs[0]))
    # No longer working since TF 2.16: https://github.com/tensorflow/tensorflow/issues/65390
    # Error: Value passed to parameter 'values' has DataType float32 not in list of allowed values: int32, int64
    # outputs.append(CategoryEncoding(1024, output_mode='count', sparse=False)(inputs[0]))
    # outputs.append(CategoryEncoding(16, output_mode='one_hot', sparse=False)(inputs[2]))
    # Error: Value passed to parameter 'values' has DataType float32 not in list of allowed values: int32, int64
    # outputs.append(CategoryEncoding(1023, output_mode='multi_hot', sparse=True)(inputs[0]))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_embedding')
    model.compile(loss='mse', optimizer='adam')

    # fit to dummy data
    training_data_size = 2
    data_in = generate_integer_input_data(training_data_size, 0, input_dims, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=1)
    return model


def get_test_model_variable() -> Model:
    """Returns a model with variably shaped input tensors."""

    input_shapes = [
        (None, None, 1),
        (None, None, 3),
        (None, 4),
    ]

    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []

    # same as axis=-1
    outputs.append(Concatenate()([inputs[0], inputs[1]]))
    outputs.append(Conv2D(8, (3, 3), padding='same', activation='elu')(inputs[0]))
    outputs.append(Conv2D(8, (3, 3), padding='same', activation='relu')(inputs[1]))
    outputs.append(GlobalMaxPooling2D()(inputs[0]))
    outputs.append(Reshape((2, -1))(inputs[2]))
    outputs.append(Reshape((-1, 2))(inputs[2]))
    outputs.append(MaxPooling2D()(inputs[1]))
    outputs.append(AveragePooling1D(2)(inputs[2]))

    outputs.append(PReLU(shared_axes=[1, 2])(inputs[0]))
    outputs.append(PReLU(shared_axes=[1, 2])(inputs[1]))
    outputs.append(PReLU(shared_axes=[1, 2, 3])(inputs[1]))
    outputs.append(PReLU(shared_axes=[1])(inputs[2]))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_variable')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 2
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=10)
    return model


def get_test_model_autoencoder() -> Model:
    """Returns a minimal autoencoder test model."""
    input_img = Input(shape=(1,), name='input_img')
    encoded = Identity()(input_img)  # Since it's about testing node connections, this suffices.
    encoder = Model(input_img, encoded, name="encoder")

    input_encoded = Input(shape=(1,), name='input_encoded')
    decoded = Identity()(input_encoded)
    decoder = Model(input_encoded, decoded, name="decoder")

    autoencoder_input = Input(shape=(1,), name='input_autoencoder')
    x = encoder(autoencoder_input)
    autoencodedanddecoded = decoder(x)
    autoencoder = Model(inputs=autoencoder_input, outputs=autoencodedanddecoded, name="autoencoder")
    autoencoder.compile(optimizer='sgd', loss='mse')
    return autoencoder


def get_test_model_sequential() -> Model:
    """Returns a typical (VGG-like) sequential test model."""
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(Permute((3, 1, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Permute((2, 3, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), activation='elu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    # fit to dummy data
    training_data_size = 2
    data_in = [np.random.random(size=(training_data_size, 32, 32, 3))]
    data_out = [np.random.random(size=(training_data_size, 10))]
    model.fit(data_in, data_out, epochs=10)
    return model


def main() -> None:
    """Generate different test models and save them to the given directory."""
    if len(sys.argv) != 3:
        print('usage: [model name] [destination file path]', flush=True)
        sys.exit(1)
    else:
        model_name = sys.argv[1]
        dest_path = sys.argv[2]

        get_model_functions = {
            'exhaustive': get_test_model_exhaustive,
            'embedding': get_test_model_embedding,
            'variable': get_test_model_variable,
            'autoencoder': get_test_model_autoencoder,
            'sequential': get_test_model_sequential,
        }

        if not model_name in get_model_functions:
            print('unknown model name: ', model_name)
            sys.exit(2)

        np.random.seed(0)

        model_func = get_model_functions[model_name]
        model = model_func()
        model.save(dest_path)

        # Make sure models can be loaded again,
        # see https://github.com/fchollet/keras/issues/7682
        model = load_model(dest_path)
        model.summary()
        # plot_model(model, to_file= str(model_name) + '.png', show_shapes=True, show_layer_names=True)


if __name__ == "__main__":
    main()
