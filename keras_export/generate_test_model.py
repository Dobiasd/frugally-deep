#!/usr/bin/env python
"""Generate a test model for frugally-deep.
"""

import sys

import numpy as np

import keras
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Input, UpSampling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"

np.random.seed(0)

def get_test_model_small():
    inputs = Input(shape=(10,))
    x = Dense(3)(inputs)
    x = Dense(4)(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model

def get_test_model_full():
    input1_shape =\
        (2, 3, 3) if K.image_data_format() == 'channels_last' else (3, 2, 3)
    input2_shape = input1_shape
    input3_shape = (4,)
    input4_shape = (2,3)

    input1 = Input(shape=input1_shape)
    input2 = Input(shape=input2_shape)
    input3 = Input(shape=input3_shape)
    input4 = Input(shape=input4_shape)

    shared_conv = Conv2D(1, (1, 1),
        padding='same', strides=2, name='shared_conv', activation='relu')

    up_scale_4 = UpSampling2D((4, 4))
    x1 = shared_conv(up_scale_4(input1))
    x1 = LeakyReLU()(x1)
    x2 = shared_conv(up_scale_4(input2))
    x2 = ELU()(x2)
    x3 = Conv2D(1, (1, 1), padding='same', strides=2)(up_scale_4(input2))
    x = keras.layers.concatenate([x1, x2, x3])

    x = Conv2D(2, (3, 3), padding='valid')(x)
    x = BatchNormalization(center=False)(x)
    x = Dropout(0.5)(x)

    x = keras.layers.concatenate([
        MaxPooling2D((2,2))(x),
        AveragePooling2D((2,2))(x)])

    x = Flatten()(x)
    x = Dense(4, activation='hard_sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dense(3, activation='selu')(x)

    intermediate_input_shape = (3,)
    intermediate_in = Input(intermediate_input_shape)
    intermediate_x = intermediate_in
    intermediate_x = Dense(4)(intermediate_x)
    intermediate_x = Dense(5)(intermediate_x)
    intermediate_model = Model(
        inputs=[intermediate_in], outputs=[intermediate_x],
        name='intermediate_model')
    intermediate_model.compile(loss='mse', optimizer='nadam')

    x = intermediate_model(x)

    intermediate_model_2 = Sequential()
    intermediate_model_2.add(Dense(7, activation='sigmoid', input_shape=(5,)))
    intermediate_model_2.add(Dense(5, activation='tanh'))
    intermediate_model_2.compile(optimizer='rmsprop',
        loss='categorical_crossentropy', metrics=['accuracy'])
    x = intermediate_model_2(x)

    x = Activation('sigmoid')(x)
    x = Dense(3)(x)

    shared_activation = Activation('tanh')
    output1 = Activation('softplus')(x)
    output2 = Activation('softmax')(x)
    output3 = shared_activation(x)
    output4 = shared_activation(input3)
    output5 = input4
    output6 = input1

    model = Model(
        inputs=[input1, input2, input3, input4],
        outputs=[output1, output2, output3, output4, output5, output6],
        name='full_model')

    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data

    training_data_size = 16
    batch_size = 8
    epochs = 1

    data_in = [
        np.random.random(size=(training_data_size, *input1_shape)),
        np.random.random(size=(training_data_size, *input2_shape)),
        np.random.random(size=(training_data_size, *input3_shape)),
        np.random.random(size=(training_data_size, *input4_shape))
    ]

    data_out = [
        np.random.random(size=(training_data_size, 3)),
        np.random.random(size=(training_data_size, 3)),
        np.random.random(size=(training_data_size, 3)),
        np.random.random(size=(training_data_size, *input3_shape)),
        np.random.random(size=(training_data_size, *input4_shape)),
        np.random.random(size=(training_data_size, *input1_shape))
    ]

    model.fit(list(data_in), list(data_out),
        epochs=epochs, batch_size=batch_size)

    return model

def main():
    if len(sys.argv) != 2:
        print('usage: [output path]')
        sys.exit(1)
    else:
        model = get_test_model_full()
        model.save(sys.argv[1])
        # Make sure model can be loaded again,
        # see https://github.com/fchollet/keras/issues/7682
        model = load_model(sys.argv[1])

if __name__ == "__main__":
    main()
