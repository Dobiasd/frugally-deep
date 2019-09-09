#!/usr/bin/env python3
"""Generate a test model for frugally-deep.
"""

import sys

import keras
import numpy as np
from keras import backend as K
from keras.layers import BatchNormalization, Concatenate
from keras.layers import Bidirectional, TimeDistributed
from keras.layers import Conv1D, ZeroPadding1D, Cropping1D
from keras.layers import Conv2D, ZeroPadding2D, Cropping2D
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import LSTM, GRU
from keras.layers import LeakyReLU, ELU, PReLU
from keras.layers import MaxPooling1D, AveragePooling1D, UpSampling1D
from keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.layers import Permute, Reshape
from keras.layers import SeparableConv2D, DepthwiseConv2D
from keras.models import Model, load_model, Sequential

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"


def is_running_on_gpu():
    """Returns if the neural network model runs on a GPU."""
    try:
        return len(keras.backend.tensorflow_backend._get_available_gpus()) > 0
    except NameError:
        return False


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
    raise ValueError('can not get determine shape for random data')


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


def generate_output_data(data_size, outputs):
    """Random output data for training."""
    return [generate_random_data(data_size, output.shape[1:])
            for output in outputs]


def get_test_model_small():
    """Returns a minimalist test model."""
    input_shapes = [
        (2, 3, 4, 5, 6),
        (2, 3, 4, 5, 6),
        (7, 8, 9, 10),
        (7, 8, 9, 10),
        (11, 12, 13),
        (11, 12, 13),
        (14, 15),
        (14, 15),
        (16,),
        (16,),
        (2,),
        (1,),
        (2,),
        (1,),
        (1, 3),
        (1, 4),
        (1, 1, 3),
        (1, 1, 4),
        (1, 1, 1, 3),
        (1, 1, 1, 4),
        (1, 1, 1, 1, 3),
        (1, 1, 1, 1, 4),
    ]

    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []

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

    outputs.append(UpSampling2D(size=(1, 2), interpolation='nearest')(inputs[4]))
    outputs.append(UpSampling2D(size=(5, 3), interpolation='nearest')(inputs[4]))
    outputs.append(UpSampling2D(size=(1, 2), interpolation='bilinear')(inputs[4]))
    outputs.append(UpSampling2D(size=(5, 3), interpolation='bilinear')(inputs[4]))

    outputs.append(keras.layers.Multiply()([inputs[10], inputs[11]]))
    outputs.append(keras.layers.Multiply()([inputs[11], inputs[10]]))
    outputs.append(keras.layers.Multiply()([inputs[11], inputs[13]]))
    outputs.append(keras.layers.Multiply()([inputs[10], inputs[11], inputs[12]]))
    outputs.append(keras.layers.Multiply()([inputs[11], inputs[12], inputs[13]]))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_small')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 1
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=10)
    return model


def get_test_model_pooling():
    """Returns a minimalistic test model for pooling layers."""

    input_shapes = [  # volume must be a multiple of 12 for Reshape
        (16, 18, 3),
        (2, 3, 6),
        (32, 32, 9),
    ]
    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []

    # 1-dimensional
    for inp in inputs:
        reshape = Reshape((-1, 12))(inp)

        # (batch_size, steps, features)
        outputs.append(AveragePooling1D(data_format="channels_last")(reshape))
        outputs.append(MaxPooling1D(data_format="channels_last")(reshape))
        outputs.append(GlobalAveragePooling1D(data_format="channels_last")(reshape))
        outputs.append(GlobalMaxPooling1D(data_format="channels_last")(reshape))

        # (batch_size, features, steps)
        outputs.append(GlobalAveragePooling1D(data_format="channels_first")(reshape))
        outputs.append(GlobalMaxPooling1D(data_format="channels_first")(reshape))

    # 2-dimensional
    for inp in inputs:
        # (batch_size, rows, cols, channels)
        outputs.append(AveragePooling2D(data_format="channels_last")(inp))
        outputs.append(MaxPooling2D(data_format="channels_last")(inp))
        outputs.append(GlobalAveragePooling2D(data_format="channels_last")(inp))
        outputs.append(GlobalMaxPooling2D(data_format="channels_last")(inp))

        # (batch_size, channels, rows, cols)
        outputs.append(AveragePooling2D(data_format="channels_first")(inp))
        outputs.append(MaxPooling2D(data_format="channels_first")(inp))
        outputs.append(GlobalAveragePooling2D(data_format="channels_first")(inp))
        outputs.append(GlobalMaxPooling2D(data_format="channels_first")(inp))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_pooling')
    model.compile(loss='mse', optimizer='adam')

    # fit to dummy data
    training_data_size = 1
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=1)
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
    training_data_size = 1
    data_in = generate_integer_input_data(training_data_size, 0, input_dims, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=1)
    return model


def get_test_model_convolutional():
    """Returns a minimalistic test model for convolutional layers."""
    input_shapes = [
        (17, 4),
        (6, 10),
        (20, 40),
    ]

    inputs = [Input(shape=s) for s in input_shapes]
    outputs = []

    for inp in inputs:
        for padding in ['same', 'valid', 'causal']:
            conv = Conv1D(6, 3, padding=padding, activation='relu')(inp)
            outputs.append(conv)

    model = Model(inputs=inputs, outputs=outputs, name='test_model_convolutional')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 1
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=10)
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

    model = Model(inputs=inputs, outputs=outputs, name='test_model_recurrent')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 1
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

        # CuDNN-enabled LSTM layers
        if is_running_on_gpu():
            # run GPU-enabled mode if GPU is available
            lstm_gpu_regular = keras.layers.CuDNNLSTM(
                units=3
            )(inp)

            lstm_gpu_bidi = Bidirectional(
                keras.layers.CuDNNLSTM(
                    units=3
                )
            )(inp)
        else:
            # fall back to equivalent regular LSTM for CPU-only mode
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
    training_data_size = 1
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=10)
    return model


def get_test_model_gru():
    """Returns a test model for Gated Recurrent Unit (GRU) layers."""

    input_shapes = [
        (17, 4),
        (1, 10)
    ]
    inputs = [Input(shape=s) for s in input_shapes]
    outputs = []

    for inp in inputs:
        gru_sequences = GRU(
            units=8,
            recurrent_activation='relu',
            reset_after=True,
            return_sequences=True,
            use_bias=True
        )(inp)
        gru_regular = GRU(
            units=3,
            recurrent_activation='sigmoid',
            reset_after=True,
            return_sequences=False,
            use_bias=False
        )(gru_sequences)
        outputs.append(gru_regular)

        gru_bidi_sequences = Bidirectional(
            GRU(
                units=4,
                recurrent_activation='hard_sigmoid',
                reset_after=False,
                return_sequences=True,
                use_bias=True
            )
        )(inp)
        gru_bidi = Bidirectional(
            GRU(
                units=6,
                recurrent_activation='sigmoid',
                reset_after=True,
                return_sequences=False,
                use_bias=False
            )
        )(gru_bidi_sequences)
        outputs.append(gru_bidi)

        # CuDNN-enabled GRU layers
        if is_running_on_gpu():
            # run GPU-enabled mode if GPU is available
            gru_gpu_regular = keras.layers.CuDNNGRU(
                units=3
            )(inp)

            gru_gpu_bidi = Bidirectional(
                keras.layers.CuDNNGRU(
                    units=3
                )
            )(inp)
        else:
            # fall back to equivalent regular GRU for CPU-only mode
            gru_gpu_regular = GRU(
                units=3,
                activation='tanh',
                recurrent_activation='sigmoid',
                reset_after=True,
                use_bias=True
            )(inp)

            gru_gpu_bidi = Bidirectional(
                GRU(
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
    training_data_size = 1
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=10)
    return model


def get_test_model_variable():
    """Returns a small model for variably shaped input tensors."""

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
    outputs.append(MaxPooling2D()(inputs[1]))
    outputs.append(AveragePooling1D()(inputs[2]))

    outputs.append(PReLU(shared_axes=[1, 2])(inputs[0]))
    outputs.append(PReLU(shared_axes=[1, 2])(inputs[1]))
    outputs.append(PReLU(shared_axes=[1, 2, 3])(inputs[1]))
    outputs.append(PReLU(shared_axes=[1])(inputs[2]))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_variable')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 1
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
    model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
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
    training_data_size = 1
    data_in = [np.random.random(size=(training_data_size, 32, 32, 3))]
    data_out = [np.random.random(size=(training_data_size, 10))]
    model.fit(data_in, data_out, epochs=10)
    return model


def get_test_model_full():
    """Returns a maximally complex test model,
    using all supported layer types with different parameter combination.
    """
    input_shapes = [
        (26, 28, 3),
        (4, 4, 3),
        (4, 4, 3),
        (4,),
        (2, 3),
        (27, 29, 1),
        (17, 1),
        (17, 4),
        (2, 3),
        (2, 3, 4, 5),
        (2, 3, 4, 5, 6),
        (2, 3, 4, 5, 6),
        (7, 8, 9, 10),
        (7, 8, 9, 10),
        (11, 12, 13),
        (11, 12, 13),
        (14, 15),
        (14, 15),
        (16,),
        (16,),
    ]

    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []

    outputs.append(Flatten()(inputs[4]))
    outputs.append(Flatten()(inputs[5]))
    outputs.append(Flatten()(inputs[9]))
    outputs.append(Flatten()(inputs[10]))

    for axis in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
        outputs.append(Concatenate(axis=axis)([inputs[10], inputs[11]]))

    for axis in [-4, -3, -2, -1, 1, 2, 3, 4]:
        outputs.append(Concatenate(axis=axis)([inputs[12], inputs[13]]))

    for axis in [-3, -2, -1, 1, 2, 3]:
        outputs.append(Concatenate(axis=axis)([inputs[14], inputs[15]]))

    for axis in [-2, -1, 1, 2]:
        outputs.append(Concatenate(axis=axis)([inputs[16], inputs[17]]))

    for axis in [-1, 1]:
        outputs.append(Concatenate(axis=axis)([inputs[18], inputs[19]]))

    for inp in inputs[6:8]:
        for padding in ['valid', 'same', 'causal']:
            for s in range(1, 6):
                for out_channels in [1, 2]:
                    for d in range(1, 4):
                        outputs.append(
                            Conv1D(out_channels, s, padding=padding,
                                   dilation_rate=d)(inp))
        for padding_size in range(0, 5):
            outputs.append(ZeroPadding1D(padding_size)(inp))
        for crop_left in range(0, 2):
            for crop_right in range(0, 2):
                outputs.append(Cropping1D((crop_left, crop_right))(inp))
        for upsampling_factor in range(1, 5):
            outputs.append(UpSampling1D(upsampling_factor)(inp))
        for padding in ['valid', 'same']:
            for pool_factor in range(1, 6):
                for s in range(1, 4):
                    outputs.append(
                        MaxPooling1D(pool_factor, strides=s,
                                     padding=padding)(inp))
                    outputs.append(
                        AveragePooling1D(pool_factor, strides=s,
                                         padding=padding)(inp))
        outputs.append(GlobalMaxPooling1D()(inp))
        outputs.append(GlobalAveragePooling1D()(inp))

    for inp in [inputs[0], inputs[5]]:
        for padding in ['valid', 'same']:
            for h in range(1, 6):
                for out_channels in [1, 2]:
                    for d in range(1, 4):
                        outputs.append(
                            Conv2D(out_channels, (h, 1), padding=padding,
                                   dilation_rate=(d, 1))(inp))
                        outputs.append(
                            SeparableConv2D(out_channels, (h, 1), padding=padding,
                                            dilation_rate=(d, 1))(inp))
                    for sy in range(1, 4):
                        outputs.append(
                            Conv2D(out_channels, (h, 1), strides=(1, sy),
                                   padding=padding)(inp))
                        outputs.append(
                            SeparableConv2D(out_channels, (h, 1),
                                            strides=(sy, sy),
                                            padding=padding)(inp))
                for sy in range(1, 4):
                    outputs.append(
                        DepthwiseConv2D((h, 1),
                                        strides=(sy, sy),
                                        padding=padding)(inp))
                    outputs.append(
                        MaxPooling2D((h, 1), strides=(1, sy),
                                     padding=padding)(inp))
            for w in range(1, 6):
                for out_channels in [1, 2]:
                    for d in range(1, 4) if sy == 1 else [1]:
                        outputs.append(
                            Conv2D(out_channels, (1, w), padding=padding,
                                   dilation_rate=(1, d))(inp))
                        outputs.append(
                            SeparableConv2D(out_channels, (1, w), padding=padding,
                                            dilation_rate=(1, d))(inp))
                    for sx in range(1, 4):
                        outputs.append(
                            Conv2D(out_channels, (1, w), strides=(sx, 1),
                                   padding=padding)(inp))
                        outputs.append(
                            SeparableConv2D(out_channels, (1, w),
                                            strides=(sx, sx),
                                            padding=padding)(inp))
                for sx in range(1, 4):
                    outputs.append(
                        DepthwiseConv2D((1, w),
                                        strides=(sy, sy),
                                        padding=padding)(inp))
                    outputs.append(
                        MaxPooling2D((1, w), strides=(1, sx),
                                     padding=padding)(inp))
    outputs.append(ZeroPadding2D(2)(inputs[0]))
    outputs.append(ZeroPadding2D((2, 3))(inputs[0]))
    outputs.append(ZeroPadding2D(((1, 2), (3, 4)))(inputs[0]))
    outputs.append(Cropping2D(2)(inputs[0]))
    outputs.append(Cropping2D((2, 3))(inputs[0]))
    outputs.append(Cropping2D(((1, 2), (3, 4)))(inputs[0]))
    for y in range(1, 3):
        for x in range(1, 3):
            outputs.append(UpSampling2D(size=(y, x))(inputs[0]))
    outputs.append(GlobalAveragePooling2D()(inputs[0]))
    outputs.append(GlobalMaxPooling2D()(inputs[0]))
    outputs.append(AveragePooling2D((2, 2))(inputs[0]))
    outputs.append(MaxPooling2D((2, 2))(inputs[0]))
    outputs.append(UpSampling2D((2, 2))(inputs[0]))
    outputs.append(Dropout(0.5)(inputs[0]))

    # same as axis=-1
    outputs.append(Concatenate()([inputs[1], inputs[2]]))
    outputs.append(Concatenate(axis=3)([inputs[1], inputs[2]]))
    # axis=0 does not make sense, since dimension 0 is the batch dimension
    outputs.append(Concatenate(axis=1)([inputs[1], inputs[2]]))
    outputs.append(Concatenate(axis=2)([inputs[1], inputs[2]]))

    outputs.append(BatchNormalization()(inputs[0]))
    outputs.append(BatchNormalization(center=False)(inputs[0]))
    outputs.append(BatchNormalization(scale=False)(inputs[0]))

    outputs.append(Conv2D(2, (3, 3), use_bias=True)(inputs[0]))
    outputs.append(Conv2D(2, (3, 3), use_bias=False)(inputs[0]))
    outputs.append(SeparableConv2D(2, (3, 3), use_bias=True)(inputs[0]))
    outputs.append(SeparableConv2D(2, (3, 3), use_bias=False)(inputs[0]))
    outputs.append(DepthwiseConv2D(2, (3, 3), use_bias=True)(inputs[0]))
    outputs.append(DepthwiseConv2D(2, (3, 3), use_bias=False)(inputs[0]))

    outputs.append(Dense(2, use_bias=True)(inputs[3]))
    outputs.append(Dense(2, use_bias=False)(inputs[3]))

    shared_conv = Conv2D(1, (1, 1),
                         padding='valid', name='shared_conv', activation='relu')

    up_scale_2 = UpSampling2D((2, 2))
    x1 = shared_conv(up_scale_2(inputs[1]))  # (1, 8, 8)
    x2 = shared_conv(up_scale_2(inputs[2]))  # (1, 8, 8)
    x3 = Conv2D(1, (1, 1), padding='valid')(up_scale_2(inputs[2]))  # (1, 8, 8)
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

    outputs.append(keras.layers.Add()([inputs[4], inputs[8], inputs[8]]))
    outputs.append(keras.layers.Subtract()([inputs[4], inputs[8]]))
    outputs.append(keras.layers.Multiply()([inputs[4], inputs[8], inputs[8]]))
    outputs.append(keras.layers.Average()([inputs[4], inputs[8], inputs[8]]))
    outputs.append(keras.layers.Maximum()([inputs[4], inputs[8], inputs[8]]))
    outputs.append(Concatenate()([inputs[4], inputs[8], inputs[8]]))

    intermediate_input_shape = (3,)
    intermediate_in = Input(intermediate_input_shape)
    intermediate_x = intermediate_in
    intermediate_x = Dense(8)(intermediate_x)
    intermediate_x = Dense(5)(intermediate_x)
    intermediate_model = Model(
        inputs=[intermediate_in], outputs=[intermediate_x],
        name='intermediate_model')
    intermediate_model.compile(loss='mse', optimizer='nadam')

    x = intermediate_model(x)  # (1, 1, 5)

    intermediate_model_2 = Sequential()
    intermediate_model_2.add(Dense(7, input_shape=(5,)))
    intermediate_model_2.add(Dense(5))
    intermediate_model_2.compile(optimizer='rmsprop',
                                 loss='categorical_crossentropy')

    x = intermediate_model_2(x)  # (1, 1, 5)

    x = Dense(3)(x)  # (1, 1, 3)

    shared_activation = Activation('tanh')

    outputs = outputs + [
        Activation('tanh')(inputs[3]),
        Activation('hard_sigmoid')(inputs[3]),
        Activation('selu')(inputs[3]),
        Activation('sigmoid')(inputs[3]),
        Activation('softplus')(inputs[3]),
        Activation('softmax')(inputs[3]),
        Activation('relu')(inputs[3]),
        LeakyReLU()(inputs[3]),
        ELU()(inputs[3]),
        PReLU()(inputs[2]),
        PReLU()(inputs[3]),
        PReLU()(inputs[4]),
        shared_activation(inputs[3]),
        Activation('linear')(inputs[4]),
        Activation('linear')(inputs[1]),
        x,
        shared_activation(x),
    ]

    print('Model has {} outputs.'.format(len(outputs)))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_full')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 1
    batch_size = 1
    epochs = 10
    data_in = generate_input_data(training_data_size, input_shapes)
    initial_data_out = model.predict(data_in)
    data_out = generate_output_data(training_data_size, initial_data_out)
    model.fit(data_in, data_out, epochs=epochs, batch_size=batch_size)
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
            'small': get_test_model_small,
            'pooling': get_test_model_pooling,
            'embedding': get_test_model_embedding,
            'convolutional': get_test_model_convolutional,
            'recurrent': get_test_model_recurrent,
            'lstm': get_test_model_lstm,
            'gru': get_test_model_gru,
            'variable': get_test_model_variable,
            'sequential': get_test_model_sequential,
            'full': get_test_model_full
        }

        if not model_name in get_model_functions:
            print('unknown model name: ', model_name)
            sys.exit(2)

        assert K.backend() == "tensorflow"
        assert K.floatx() == "float32"
        assert K.image_data_format() == 'channels_last'

        np.random.seed(0)

        model_func = get_model_functions[model_name]
        model = model_func()
        model.save(dest_path, include_optimizer=False)

        # Make sure models can be loaded again,
        # see https://github.com/fchollet/keras/issues/7682
        model = load_model(dest_path)
        print(model.summary())


if __name__ == "__main__":
    main()
