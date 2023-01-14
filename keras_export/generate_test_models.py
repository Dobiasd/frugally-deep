#!/usr/bin/env python3
"""Generate a test model for frugally-deep.
"""

import sys

import numpy as np
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.layers import Conv1D, ZeroPadding1D, Cropping1D
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.layers import Embedding, Normalization, Rescaling
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import LeakyReLU, ELU, PReLU, ReLU
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D, UpSampling1D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Multiply, Add, Subtract, Average, Maximum, Minimum, Dot
from tensorflow.keras.layers import Permute, Reshape, RepeatVector
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.models import Model, load_model, Sequential

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"


def replace_none_with(value, shape):
    """Replace every None with a fixed value."""
    return tuple(list(map(lambda x: x if x is not None else value, shape)))


def get_shape_for_random_data(data_size, shape):
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


def generate_random_data(data_size, shape):
    """Random data for training."""
    return np.random.random(
        size=get_shape_for_random_data(data_size, replace_none_with(42, shape)))


def generate_input_data(data_size, input_shapes):
    """Random input data for training."""
    return [generate_random_data(data_size, input_shape)
            for input_shape in input_shapes]


def generate_integer_random_data(data_size, low, high, shape):
    """Random data for training."""
    return np.random.randint(
        low=low, high=high, size=get_shape_for_random_data(data_size, replace_none_with(42, shape)))


def generate_integer_input_data(data_size, low, highs, input_shapes):
    """Random input data for training."""
    return [generate_integer_random_data(data_size, low, high, input_shape)
            for high, input_shape in zip(highs, input_shapes)]


def as_list(value_or_values):
    """Leave lists untouched, convert non-list types to a singleton list"""
    if isinstance(value_or_values, list):
        return value_or_values
    return [value_or_values]


def generate_output_data(data_size, outputs):
    """Random output data for training."""
    return [generate_random_data(data_size, output.shape[1:])
            for output in as_list(outputs)]


def get_test_model_exhaustive():
    """Returns a exhaustive test model."""
    input_shapes = [
        (2, 3, 4, 5, 6),  # 0
        (2, 3, 4, 5, 6),
        (7, 8, 9, 10),
        (7, 8, 9, 10),
        (11, 12, 13),
        (11, 12, 13),
        (14, 15),
        (14, 15),
        (16,),
        (16,),
        (2,),  # 10
        (1,),
        (2,),
        (1,),
        (1, 3),
        (1, 4),
        (1, 1, 3),
        (1, 1, 4),
        (1, 1, 1, 3),
        (1, 1, 1, 4),
        (1, 1, 1, 1, 3),  # 20
        (1, 1, 1, 1, 4),
        (26, 28, 3),
        (4, 4, 3),
        (4, 4, 3),
        (4,),
        (2, 3),
        (1,),
        (1,),
        (1,),
        (2, 3),  # 30
        (9, 16, 1),
        (1, 9, 16),
        (6, 1, 1),
        (1, 1, 1, 1, 6),
        (1, 1, 1, 10),
        (1, 1, 13),
        (1, 15),
        (1, 1, 1, 1, 6),
        (1, 1, 1, 5, 1),
        (1, 1, 4, 1, 1),  # 40
        (1, 3, 1, 1, 1),
        (2, 1, 1, 1, 1),
        (1, 1, 4, 1, 6),
        (1, 3, 1, 5, 1),
        (2, 1, 4, 1, 1),
        (1,),  # 46
        (3, 1),
        (6, 5, 4, 3, 2),
    ]

    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []

    outputs.append(Conv1D(1, 3, padding='valid')(inputs[6]))
    outputs.append(Conv1D(2, 1, padding='same')(inputs[6]))
    outputs.append(Conv1D(3, 4, padding='causal', dilation_rate=2)(inputs[6]))
    outputs.append(ZeroPadding1D(2)(inputs[6]))
    outputs.append(Cropping1D((2, 3))(inputs[6]))
    outputs.append(MaxPooling1D(2)(inputs[6]))
    outputs.append(MaxPooling1D(2, strides=2, padding='same')(inputs[6]))
    outputs.append(MaxPooling1D(2, data_format="channels_first")(inputs[6]))
    outputs.append(AveragePooling1D(2)(inputs[6]))
    outputs.append(AveragePooling1D(2, strides=2, padding='same')(inputs[6]))
    outputs.append(AveragePooling1D(2, data_format="channels_first")(inputs[6]))
    outputs.append(GlobalMaxPooling1D()(inputs[6]))
    outputs.append(GlobalMaxPooling1D(data_format="channels_first")(inputs[6]))
    outputs.append(GlobalAveragePooling1D()(inputs[6]))
    outputs.append(GlobalAveragePooling1D(data_format="channels_first")(inputs[6]))

    outputs.append(Normalization(axis=None, mean=2.1, variance=2.2)(inputs[4]))
    outputs.append(Normalization(axis=-1, mean=2.1, variance=2.2)(inputs[6]))
    outputs.append(Normalization(axis=-1, mean=2.1, variance=2.2)(inputs[46]))
    outputs.append(Normalization(axis=1, mean=2.1, variance=2.2)(inputs[46]))
    outputs.append(Normalization(axis=-1, mean=2.1, variance=2.2)(inputs[47]))
    outputs.append(Normalization(axis=1, mean=2.1, variance=2.2)(inputs[47]))
    outputs.append(Normalization(axis=2, mean=2.1, variance=2.2)(inputs[47]))
    for axis in range(1, 6):
        shape = input_shapes[0][axis - 1]
        outputs.append(Normalization(axis=axis,
                                     mean=np.random.rand(shape),
                                     variance=np.random.rand(shape)
                                     )(inputs[0]))

    outputs.append(Rescaling(23.5, 42.1)(inputs[0]))

    outputs.append(Conv2D(4, (3, 3))(inputs[4]))
    outputs.append(Conv2D(4, (3, 3), use_bias=False)(inputs[4]))
    outputs.append(Conv2D(4, (2, 4), strides=(2, 3), padding='same')(inputs[4]))
    outputs.append(Conv2D(4, (2, 4), padding='same', dilation_rate=(2, 3))(inputs[4]))

    outputs.append(SeparableConv2D(3, (3, 3))(inputs[4]))
    outputs.append(DepthwiseConv2D((3, 3))(inputs[4]))
    outputs.append(DepthwiseConv2D((1, 2))(inputs[4]))

    outputs.append(MaxPooling2D((2, 2))(inputs[4]))
    # todo: check if TensorFlow >= 2.8 supports this
    # outputs.append(MaxPooling2D((2, 2), data_format="channels_first")(inputs[4]))
    outputs.append(MaxPooling2D((1, 3), strides=(2, 3), padding='same')(inputs[4]))
    outputs.append(AveragePooling2D((2, 2))(inputs[4]))
    # todo: check if TensorFlow >= 2.8 supports this
    # outputs.append(AveragePooling2D((2, 2), data_format="channels_first")(inputs[4]))
    outputs.append(AveragePooling2D((1, 3), strides=(2, 3), padding='same')(inputs[4]))

    outputs.append(GlobalAveragePooling2D()(inputs[4]))
    outputs.append(GlobalAveragePooling2D(data_format="channels_first")(inputs[4]))
    outputs.append(GlobalMaxPooling2D()(inputs[4]))
    outputs.append(GlobalMaxPooling2D(data_format="channels_first")(inputs[4]))

    outputs.append(Permute((3, 4, 1, 5, 2))(inputs[0]))
    outputs.append(Permute((1, 5, 3, 2, 4))(inputs[0]))
    outputs.append(Permute((3, 4, 1, 2))(inputs[2]))
    outputs.append(Permute((2, 1, 3))(inputs[4]))
    outputs.append(Permute((2, 1))(inputs[6]))
    outputs.append(Permute((1,))(inputs[8]))

    outputs.append(Permute((3, 1, 2))(inputs[31]))
    outputs.append(Permute((3, 1, 2))(inputs[32]))
    outputs.append(BatchNormalization()(Permute((3, 1, 2))(inputs[31])))
    outputs.append(BatchNormalization()(Permute((3, 1, 2))(inputs[32])))

    outputs.append(BatchNormalization()(inputs[0]))
    outputs.append(BatchNormalization(axis=1)(inputs[0]))
    outputs.append(BatchNormalization(axis=2)(inputs[0]))
    outputs.append(BatchNormalization(axis=3)(inputs[0]))
    outputs.append(BatchNormalization(axis=4)(inputs[0]))
    outputs.append(BatchNormalization(axis=5)(inputs[0]))
    outputs.append(BatchNormalization()(inputs[2]))
    outputs.append(BatchNormalization(axis=1)(inputs[2]))
    outputs.append(BatchNormalization(axis=2)(inputs[2]))
    outputs.append(BatchNormalization(axis=3)(inputs[2]))
    outputs.append(BatchNormalization(axis=4)(inputs[2]))
    outputs.append(BatchNormalization()(inputs[4]))
    # todo: check if TensorFlow >= 2.1 supports this
    # outputs.append(BatchNormalization(axis=1)(inputs[4])) # tensorflow.python.framework.errors_impl.InternalError:  The CPU implementation of FusedBatchNorm only supports NHWC tensor format for now.
    outputs.append(BatchNormalization(axis=2)(inputs[4]))
    outputs.append(BatchNormalization(axis=3)(inputs[4]))
    outputs.append(BatchNormalization()(inputs[6]))
    outputs.append(BatchNormalization(axis=1)(inputs[6]))
    outputs.append(BatchNormalization(axis=2)(inputs[6]))
    outputs.append(BatchNormalization()(inputs[8]))
    outputs.append(BatchNormalization(axis=1)(inputs[8]))
    outputs.append(BatchNormalization()(inputs[27]))
    outputs.append(BatchNormalization(axis=1)(inputs[27]))
    outputs.append(BatchNormalization()(inputs[14]))
    outputs.append(BatchNormalization(axis=1)(inputs[14]))
    outputs.append(BatchNormalization(axis=2)(inputs[14]))
    outputs.append(BatchNormalization()(inputs[16]))
    # todo: check if TensorFlow >= 2.1 supports this
    # outputs.append(BatchNormalization(axis=1)(inputs[16])) # tensorflow.python.framework.errors_impl.InternalError:  The CPU implementation of FusedBatchNorm only supports NHWC tensor format for now.
    outputs.append(BatchNormalization(axis=2)(inputs[16]))
    outputs.append(BatchNormalization(axis=3)(inputs[16]))
    outputs.append(BatchNormalization()(inputs[18]))
    outputs.append(BatchNormalization(axis=1)(inputs[18]))
    outputs.append(BatchNormalization(axis=2)(inputs[18]))
    outputs.append(BatchNormalization(axis=3)(inputs[18]))
    outputs.append(BatchNormalization(axis=4)(inputs[18]))
    outputs.append(BatchNormalization()(inputs[20]))
    outputs.append(BatchNormalization(axis=1)(inputs[20]))
    outputs.append(BatchNormalization(axis=2)(inputs[20]))
    outputs.append(BatchNormalization(axis=3)(inputs[20]))
    outputs.append(BatchNormalization(axis=4)(inputs[20]))
    outputs.append(BatchNormalization(axis=5)(inputs[20]))

    outputs.append(Dropout(0.5)(inputs[4]))

    outputs.append(ZeroPadding2D(2)(inputs[4]))
    outputs.append(ZeroPadding2D((2, 3))(inputs[4]))
    outputs.append(ZeroPadding2D(((1, 2), (3, 4)))(inputs[4]))
    outputs.append(Cropping2D(2)(inputs[4]))
    outputs.append(Cropping2D((2, 3))(inputs[4]))
    outputs.append(Cropping2D(((1, 2), (3, 4)))(inputs[4]))

    outputs.append(Dense(3, use_bias=True)(inputs[13]))
    outputs.append(Dense(3, use_bias=True)(inputs[14]))
    outputs.append(Dense(4, use_bias=False)(inputs[16]))
    outputs.append(Dense(4, use_bias=False, activation='tanh')(inputs[18]))
    outputs.append(Dense(4, use_bias=False)(inputs[20]))

    outputs.append(Reshape(((2 * 3 * 4 * 5 * 6),))(inputs[0]))
    outputs.append(Reshape((2, 3 * 4 * 5 * 6))(inputs[0]))
    outputs.append(Reshape((2, 3, 4 * 5 * 6))(inputs[0]))
    outputs.append(Reshape((2, 3, 4, 5 * 6))(inputs[0]))
    outputs.append(Reshape((2, 3, 4, 5, 6))(inputs[0]))

    outputs.append(Maximum()([inputs[0], inputs[1]]))
    outputs.append(Maximum()([inputs[2], inputs[3]]))
    outputs.append(Maximum()([inputs[4], inputs[5]]))
    outputs.append(Maximum()([inputs[6], inputs[7]]))
    outputs.append(Maximum()([inputs[8], inputs[9]]))

    outputs.append(Minimum()([inputs[0], inputs[1]]))
    outputs.append(Minimum()([inputs[2], inputs[3]]))
    outputs.append(Minimum()([inputs[4], inputs[5]]))
    outputs.append(Minimum()([inputs[6], inputs[7]]))
    outputs.append(Minimum()([inputs[8], inputs[9]]))

    for normalize in [True, False]:
        outputs.append(Dot(axes=(1, 1), normalize=normalize)([inputs[8], inputs[9]]))
        outputs.append(Dot(axes=(1, 1), normalize=normalize)([inputs[0], inputs[10]]))
        outputs.append(Dot(axes=1, normalize=normalize)([inputs[0], inputs[10]]))
        outputs.append(Dot(axes=(3, 1), normalize=normalize)([inputs[31], inputs[32]]))
        outputs.append(Dot(axes=(2, 3), normalize=normalize)([inputs[31], inputs[32]]))
        outputs.append(Dot(axes=(2, 3), normalize=normalize)([inputs[14], inputs[16]]))
        outputs.append(Dot(axes=(3, 2), normalize=normalize)([inputs[24], inputs[26]]))

    outputs.append(Reshape((16,))(inputs[8]))
    outputs.append(Reshape((2, 8))(inputs[8]))
    outputs.append(Reshape((2, 2, 4))(inputs[8]))
    outputs.append(Reshape((2, 2, 2, 2))(inputs[8]))
    outputs.append(Reshape((2, 2, 1, 2, 2))(inputs[8]))

    outputs.append(RepeatVector(3)(inputs[8]))

    outputs.append(UpSampling2D(size=(1, 2), interpolation='nearest')(inputs[4]))
    outputs.append(UpSampling2D(size=(5, 3), interpolation='nearest')(inputs[4]))
    outputs.append(UpSampling2D(size=(1, 2), interpolation='bilinear')(inputs[4]))
    outputs.append(UpSampling2D(size=(5, 3), interpolation='bilinear')(inputs[4]))

    outputs.append(ReLU()(inputs[0]))

    for axis in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
        outputs.append(Concatenate(axis=axis)([inputs[0], inputs[1]]))
    for axis in [-4, -3, -2, -1, 1, 2, 3, 4]:
        outputs.append(Concatenate(axis=axis)([inputs[2], inputs[3]]))
    for axis in [-3, -2, -1, 1, 2, 3]:
        outputs.append(Concatenate(axis=axis)([inputs[4], inputs[5]]))
    for axis in [-2, -1, 1, 2]:
        outputs.append(Concatenate(axis=axis)([inputs[6], inputs[7]]))
    for axis in [-1, 1]:
        outputs.append(Concatenate(axis=axis)([inputs[8], inputs[9]]))
    for axis in [-1, 2]:
        outputs.append(Concatenate(axis=axis)([inputs[14], inputs[15]]))
    for axis in [-1, 3]:
        outputs.append(Concatenate(axis=axis)([inputs[16], inputs[17]]))
    for axis in [-1, 4]:
        outputs.append(Concatenate(axis=axis)([inputs[18], inputs[19]]))
    for axis in [-1, 5]:
        outputs.append(Concatenate(axis=axis)([inputs[20], inputs[21]]))

    outputs.append(UpSampling1D(size=2)(inputs[6]))
    # outputs.append(UpSampling1D(size=2)(inputs[8])) # ValueError: Input 0 of layer up_sampling1d_1 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: [None, 16]

    outputs.append(Multiply()([inputs[10], inputs[11]]))
    outputs.append(Multiply()([inputs[11], inputs[10]]))
    outputs.append(Multiply()([inputs[11], inputs[13]]))
    outputs.append(Multiply()([inputs[10], inputs[11], inputs[12]]))
    outputs.append(Multiply()([inputs[11], inputs[12], inputs[13]]))
    outputs.append(Multiply()([inputs[14], inputs[16], inputs[18], inputs[20]]))
    outputs.append(Multiply()([inputs[14], inputs[16]]))
    outputs.append(Multiply()([inputs[16], inputs[18]]))
    outputs.append(Multiply()([inputs[18], inputs[20]]))
    outputs.append(Multiply()([inputs[30], inputs[33]]))
    outputs.append(Multiply()([inputs[34], inputs[0]]))
    outputs.append(Multiply()([inputs[35], inputs[2]]))
    outputs.append(Multiply()([inputs[36], inputs[4]]))
    outputs.append(Multiply()([inputs[37], inputs[6]]))
    outputs.append(Multiply()([inputs[0], inputs[38]]))
    outputs.append(Multiply()([inputs[0], inputs[39]]))
    outputs.append(Multiply()([inputs[0], inputs[40]]))
    outputs.append(Multiply()([inputs[0], inputs[41]]))
    outputs.append(Multiply()([inputs[0], inputs[42]]))
    outputs.append(Multiply()([inputs[43], inputs[44]]))
    outputs.append(Multiply()([inputs[44], inputs[45]]))

    shared_conv = Conv2D(1, (1, 1),
                         padding='valid', name='shared_conv', activation='relu')

    up_scale_2 = UpSampling2D((2, 2))
    x1 = shared_conv(up_scale_2(inputs[23]))  # (1, 8, 8)
    x2 = shared_conv(up_scale_2(inputs[24]))  # (1, 8, 8)
    x3 = Conv2D(1, (1, 1), padding='valid')(up_scale_2(inputs[24]))  # (1, 8, 8)
    x = Concatenate()([x1, x2, x3])  # (3, 8, 8)
    outputs.append(x)

    x = Conv2D(3, (1, 1), padding='same', use_bias=False)(x)  # (3, 8, 8)
    outputs.append(x)
    x = Dropout(0.5)(x)
    outputs.append(x)
    x = Concatenate()([
        MaxPooling2D((2, 2))(x),
        AveragePooling2D((2, 2))(x)])  # (6, 4, 4)
    outputs.append(x)

    x = Flatten()(x)  # (1, 1, 96)
    x = Dense(4, use_bias=False)(x)
    outputs.append(x)
    x = Dense(3)(x)  # (1, 1, 3)
    outputs.append(x)

    outputs.append(Add()([inputs[26], inputs[30], inputs[30]]))
    outputs.append(Subtract()([inputs[26], inputs[30]]))
    outputs.append(Multiply()([inputs[26], inputs[30], inputs[30]]))
    outputs.append(Average()([inputs[26], inputs[30], inputs[30]]))
    outputs.append(Maximum()([inputs[26], inputs[30], inputs[30]]))
    outputs.append(Concatenate()([inputs[26], inputs[30], inputs[30]]))

    intermediate_input_shape = (3,)
    intermediate_in = Input(intermediate_input_shape)
    intermediate_x = intermediate_in
    intermediate_x = Dense(8)(intermediate_x)
    intermediate_x = Dense(5, name='duplicate_layer_name')(intermediate_x)
    intermediate_model = Model(
        inputs=[intermediate_in], outputs=[intermediate_x],
        name='intermediate_model')
    intermediate_model.compile(loss='mse', optimizer='nadam')

    x = intermediate_model(x)  # (1, 1, 5)

    intermediate_model_2 = Sequential()
    intermediate_model_2.add(Dense(7, input_shape=(5,)))
    intermediate_model_2.add(Dense(5, name='duplicate_layer_name'))
    intermediate_model_2.compile(optimizer='rmsprop',
                                 loss='categorical_crossentropy')

    x = intermediate_model_2(x)  # (1, 1, 5)

    intermediate_model_3_nested = Sequential()
    intermediate_model_3_nested.add(Dense(7, input_shape=(6,)))
    intermediate_model_3_nested.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    intermediate_model_3 = Sequential()
    intermediate_model_3.add(Dense(6, input_shape=(5,)))
    intermediate_model_3.add(intermediate_model_3_nested)
    intermediate_model_3.add(Dense(8))
    intermediate_model_3.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    x = intermediate_model_3(x)  # (1, 1, 8)

    x = Dense(3)(x)  # (1, 1, 3)

    shared_activation = Activation('tanh')

    outputs = outputs + [
        Activation('tanh')(inputs[25]),
        Activation('hard_sigmoid')(inputs[25]),
        Activation('selu')(inputs[25]),
        Activation('sigmoid')(inputs[25]),
        Activation('softplus')(inputs[25]),
        Activation('softmax')(inputs[25]),
        Activation('relu')(inputs[25]),
        Activation('relu6')(inputs[25]),
        Activation('swish')(inputs[25]),
        Activation('exponential')(inputs[25]),
        Activation('gelu')(inputs[25]),
        Activation('softsign')(inputs[25]),
        LeakyReLU()(inputs[25]),
        ReLU()(inputs[25]),
        ReLU(max_value=0.4, negative_slope=1.1, threshold=0.3)(inputs[25]),
        ELU()(inputs[25]),
        PReLU()(inputs[24]),
        PReLU()(inputs[25]),
        PReLU()(inputs[26]),
        shared_activation(inputs[25]),
        Activation('linear')(inputs[26]),
        Activation('linear')(inputs[23]),
        x,
        shared_activation(x),
    ]

    model = Model(inputs=inputs, outputs=outputs, name='test_model_exhaustive')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 2
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=10)
    return model


def get_test_model_embedding():
    """Returns a minimalistic test model for the embedding layer."""

    input_dims = [
        1023,  # maximum integer value in input data
        255
    ]
    input_shapes = [
        (100,),  # must be single-element tuple (for sequence length)
        (1000,)
    ]
    assert len(input_dims) == len(input_shapes)
    output_dims = [8, 3]  # embedding dimension

    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []
    for k in range(0, len(input_shapes)):
        embedding = Embedding(input_dim=input_dims[k], output_dim=output_dims[k])(inputs[k])
        lstm = LSTM(
            units=4,
            recurrent_activation='sigmoid',
            return_sequences=False
        )(embedding)

        outputs.append(lstm)

    model = Model(inputs=inputs, outputs=outputs, name='test_model_embedding')
    model.compile(loss='mse', optimizer='adam')

    # fit to dummy data
    training_data_size = 2
    data_in = generate_integer_input_data(training_data_size, 0, input_dims, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=1)
    return model


def get_test_model_recurrent():
    """Returns a minimalistic test model for recurrent layers."""
    input_shapes = [
        (17, 4),
        (1, 10),
        (20, 40),
        (6, 7, 10, 3)
    ]

    outputs = []

    inputs = [Input(shape=s) for s in input_shapes]

    inp = PReLU()(inputs[0])

    lstm = Bidirectional(LSTM(units=4,
                              return_sequences=True,
                              bias_initializer='random_uniform',  # default is zero use random to test computation
                              activation='tanh',
                              recurrent_activation='relu'), merge_mode='concat')(inp)

    lstm2 = Bidirectional(LSTM(units=6,
                               return_sequences=True,
                               bias_initializer='random_uniform',
                               activation='elu',
                               recurrent_activation='hard_sigmoid'), merge_mode='sum')(lstm)

    lstm3 = LSTM(units=10,
                 return_sequences=False,
                 bias_initializer='random_uniform',
                 activation='selu',
                 recurrent_activation='sigmoid')(lstm2)

    outputs.append(lstm3)

    conv1 = Conv1D(2, 1, activation='sigmoid')(inputs[1])
    lstm4 = LSTM(units=15,
                 return_sequences=False,
                 bias_initializer='random_uniform',
                 activation='tanh',
                 recurrent_activation='elu')(conv1)

    dense = (Dense(23, activation='sigmoid'))(lstm4)
    outputs.append(dense)

    time_dist_1 = TimeDistributed(Conv2D(2, (3, 3), use_bias=True))(inputs[3])
    flatten_1 = TimeDistributed(Flatten())(time_dist_1)

    outputs.append(Bidirectional(LSTM(units=6,
                                      return_sequences=True,
                                      bias_initializer='random_uniform',
                                      activation='tanh',
                                      recurrent_activation='sigmoid'), merge_mode='ave')(flatten_1))

    outputs.append(TimeDistributed(MaxPooling2D(2, 2))(inputs[3]))
    outputs.append(TimeDistributed(AveragePooling2D(2, 2))(inputs[3]))
    outputs.append(TimeDistributed(BatchNormalization())(inputs[3]))

    nested_inputs = Input(shape=input_shapes[0][1:])
    nested_x = Dense(5, activation='relu')(nested_inputs)
    nested_predictions = Dense(3, activation='softmax')(nested_x)
    nested_model = Model(inputs=nested_inputs, outputs=nested_predictions)
    nested_model.compile(loss='categorical_crossentropy', optimizer='nadam')
    outputs.append(TimeDistributed(nested_model)(inputs[0]))

    nested_sequential_model = Sequential()
    nested_sequential_model.add(Flatten(input_shape=input_shapes[0][1:]))
    nested_sequential_model.compile(optimizer='rmsprop',
                                    loss='categorical_crossentropy')
    outputs.append(TimeDistributed(nested_sequential_model)(inputs[0]))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_recurrent')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 2
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=10)
    return model


def get_test_model_lstm():
    """Returns a test model for Long Short-Term Memory (LSTM) layers."""

    input_shapes = [
        (17, 4),
        (1, 10),
        (None, 4),
        (12,),
        (12,)
    ]
    inputs = [Input(shape=s) for s in input_shapes]
    outputs = []

    for inp in inputs[:2]:
        lstm_sequences = LSTM(
            units=8,
            recurrent_activation='relu',
            return_sequences=True
        )(inp)
        lstm_regular = LSTM(
            units=3,
            recurrent_activation='sigmoid',
            return_sequences=False
        )(lstm_sequences)
        outputs.append(lstm_regular)
        lstm_state, state_h, state_c = LSTM(
            units=3,
            recurrent_activation='sigmoid',
            return_state=True
        )(inp)
        outputs.append(lstm_state)
        outputs.append(state_h)
        outputs.append(state_c)

        lstm_bidi_sequences = Bidirectional(
            LSTM(
                units=4,
                recurrent_activation='hard_sigmoid',
                return_sequences=True
            )
        )(inp)
        lstm_bidi = Bidirectional(
            LSTM(
                units=6,
                recurrent_activation='linear',
                return_sequences=False
            )
        )(lstm_bidi_sequences)
        outputs.append(lstm_bidi)

        lstm_gpu_regular = LSTM(
            units=3,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True
        )(inp)

        lstm_gpu_bidi = Bidirectional(
            LSTM(
                units=3,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True
            )
        )(inp)
    outputs.append(lstm_gpu_regular)
    outputs.append(lstm_gpu_bidi)

    outputs.extend(LSTM(units=12, return_sequences=True,
                        return_state=True)(inputs[2], initial_state=[inputs[3], inputs[4]]))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_lstm')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 2
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=10)
    return model


def get_test_model_gru():
    return get_test_model_gru_stateful_optional(False)


def get_test_model_gru_stateful():
    return get_test_model_gru_stateful_optional(True)


def get_test_model_gru_stateful_optional(stateful):
    """Returns a test model for Gated Recurrent Unit (GRU) layers."""
    input_shapes = [
        (17, 4),
        (1, 10)
    ]
    stateful_batch_size = 1
    inputs = [Input(batch_shape=(stateful_batch_size,) + s) for s in input_shapes]
    outputs = []

    for inp in inputs:
        gru_sequences = GRU(
            stateful=stateful,
            units=8,
            recurrent_activation='relu',
            reset_after=True,
            return_sequences=True,
            use_bias=True
        )(inp)
        gru_regular = GRU(
            stateful=stateful,
            units=3,
            recurrent_activation='sigmoid',
            reset_after=True,
            return_sequences=False,
            use_bias=False
        )(gru_sequences)
        outputs.append(gru_regular)

        gru_bidi_sequences = Bidirectional(
            GRU(
                stateful=stateful,
                units=4,
                recurrent_activation='hard_sigmoid',
                reset_after=False,
                return_sequences=True,
                use_bias=True
            )
        )(inp)
        gru_bidi = Bidirectional(
            GRU(
                stateful=stateful,
                units=6,
                recurrent_activation='sigmoid',
                reset_after=True,
                return_sequences=False,
                use_bias=False
            )
        )(gru_bidi_sequences)
        outputs.append(gru_bidi)

        gru_gpu_regular = GRU(
            stateful=stateful,
            units=3,
            activation='tanh',
            recurrent_activation='sigmoid',
            reset_after=True,
            use_bias=True
        )(inp)

        gru_gpu_bidi = Bidirectional(
            GRU(
                stateful=stateful,
                units=3,
                activation='tanh',
                recurrent_activation='sigmoid',
                reset_after=True,
                use_bias=True
            )
        )(inp)
        outputs.append(gru_gpu_regular)
        outputs.append(gru_gpu_bidi)

    model = Model(inputs=inputs, outputs=outputs, name='test_model_gru')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = stateful_batch_size
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, batch_size=stateful_batch_size, epochs=10)
    return model


def get_test_model_variable():
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
    outputs.append(AveragePooling1D()(inputs[2]))

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


def get_test_model_sequential():
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


def get_test_model_lstm_stateful():
    stateful_batch_size = 1
    input_shapes = [
        (17, 4),
        (1, 10),
        (None, 4),
        (12,),
        (12,)
    ]

    inputs = [Input(batch_shape=(stateful_batch_size,) + s) for s in input_shapes]
    outputs = []
    for in_num, inp in enumerate(inputs[:2]):
        stateful = bool((in_num + 1) % 2)
        lstm_sequences = LSTM(
            stateful=stateful,
            units=8,
            recurrent_activation='relu',
            return_sequences=True,
            name='lstm_sequences_' + str(in_num) + '_st-' + str(stateful)
        )(inp)
        stateful = bool((in_num) % 2)
        lstm_regular = LSTM(
            stateful=stateful,
            units=3,
            recurrent_activation='sigmoid',
            return_sequences=False,
            name='lstm_regular_' + str(in_num) + '_st-' + str(stateful)
        )(lstm_sequences)
        outputs.append(lstm_regular)
        stateful = bool((in_num + 1) % 2)
        lstm_state, state_h, state_c = LSTM(
            stateful=stateful,
            units=3,
            recurrent_activation='sigmoid',
            return_state=True,
            name='lstm_state_return_' + str(in_num) + '_st-' + str(stateful)
        )(inp)
        outputs.append(lstm_state)
        outputs.append(state_h)
        outputs.append(state_c)
        stateful = bool((in_num + 1) % 2)
        lstm_bidi_sequences = Bidirectional(
            LSTM(
                stateful=stateful,
                units=4,
                recurrent_activation='hard_sigmoid',
                return_sequences=True,
                name='bi-lstm1_' + str(in_num) + '_st-' + str(stateful)
            )
        )(inp)
        stateful = bool((in_num) % 2)
        lstm_bidi = Bidirectional(
            LSTM(
                stateful=stateful,
                units=6,
                recurrent_activation='linear',
                return_sequences=False,
                name='bi-lstm2_' + str(in_num) + '_st-' + str(stateful)
            )
        )(lstm_bidi_sequences)
        outputs.append(lstm_bidi)

    initial_state_stateful = LSTM(units=12, return_sequences=True, stateful=True, return_state=True,
                                  name='initial_state_stateful')(inputs[2], initial_state=[inputs[3], inputs[4]])
    outputs.extend(initial_state_stateful)
    initial_state_not_stateful = LSTM(units=12, return_sequences=False, stateful=False, return_state=True,
                                      name='initial_state_not_stateful')(inputs[2],
                                                                         initial_state=[inputs[3], inputs[4]])
    outputs.extend(initial_state_not_stateful)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='nadam')

    # fit to dummy data
    training_data_size = stateful_batch_size
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)

    model.fit(data_in, data_out, batch_size=stateful_batch_size, epochs=10)
    return model


def main():
    """Generate different test models and save them to the given directory."""
    if len(sys.argv) != 3:
        print('usage: [model name] [destination file path]')
        sys.exit(1)
    else:
        model_name = sys.argv[1]
        dest_path = sys.argv[2]

        get_model_functions = {
            'exhaustive': get_test_model_exhaustive,
            'embedding': get_test_model_embedding,
            'recurrent': get_test_model_recurrent,
            'lstm': get_test_model_lstm,
            'gru': get_test_model_gru,
            'variable': get_test_model_variable,
            'sequential': get_test_model_sequential,
            'lstm_stateful': get_test_model_lstm_stateful,
            'gru_stateful': get_test_model_gru_stateful
        }

        if not model_name in get_model_functions:
            print('unknown model name: ', model_name)
            sys.exit(2)

        np.random.seed(0)

        model_func = get_model_functions[model_name]
        model = model_func()
        model.save(dest_path, include_optimizer=False)

        # Make sure models can be loaded again,
        # see https://github.com/fchollet/keras/issues/7682
        model = load_model(dest_path)
        model.summary()
        # plot_model(model, to_file= str(model_name) + '.png', show_shapes=True, show_layer_names=True)  #### DEBUG stateful


if __name__ == "__main__":
    main()
