#!/usr/bin/env python3
"""Generate a test model for frugally-deep.
"""

import sys

import numpy as np

import keras
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Input, UpSampling2D, Flatten, SeparableConv2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"

def get_test_model_full():
    image_format = K.image_data_format()
    input0_shape = (6, 4, 3) if image_format == 'channels_last' else (3, 6, 4)
    input1_shape = (4, 4, 3) if image_format == 'channels_last' else (3, 4, 4)
    input2_shape = input1_shape
    input3_shape = (4,)
    input4_shape = (2,3)

    input0 = Input(shape=input0_shape) # (3, 4, 6)
    input1 = Input(shape=input1_shape) # (3, 4, 4) channels_first notation
    input2 = Input(shape=input2_shape) # (3, 4, 4)
    input3 = Input(shape=input3_shape) # (3, 4, 4)
    input4 = Input(shape=input4_shape) # (3, 4, 4)

    shared_conv = Conv2D(1, (1, 1),
        padding='valid', name='shared_conv', activation='relu')

    up_scale_2 = UpSampling2D((2, 2))
    x1 = shared_conv(up_scale_2(input1)) # (1, 8, 8)
    x1 = LeakyReLU()(x1)
    x2 = shared_conv(up_scale_2(input2)) # (1, 8, 8)
    x2 = ELU()(x2)
    x3 = Conv2D(1, (1, 1), padding='valid')(up_scale_2(input2)) # (1, 8, 8)
    x = keras.layers.concatenate([x1, x2, x3]) # (3, 8, 8)

    x = Conv2D(4, (1, 1), padding='valid')(x) # (4, 8, 8)
    x = SeparableConv2D(7, (5, 3), padding='same')(x) # (7, 8, 8)
    x = SeparableConv2D(6, (3, 3), padding='same')(x) # (3, 8, 8)
    x = Conv2D(4, (3, 3), padding='same')(x) # (4, 8, 8)
    x = Conv2D(5, (3, 1), padding='same')(x) # (5, 8, 8)
    x = Conv2D(2, (1, 3), padding='same')(x) # (2, 8, 8)
    x = Conv2D(3, (5, 5), padding='same')(x) # (3, 8, 8)
    x = Conv2D(3, (1, 1), padding='same', use_bias=False)(x) # (3, 8, 8)
    x = BatchNormalization(center=False)(x)
    x = Dropout(0.5)(x)

    x = keras.layers.concatenate([
        MaxPooling2D((2,2))(x),
        AveragePooling2D((2,2))(x)]) # (6, 4, 4)

    x = Flatten()(x) # (1, 1, 96)
    x = Dense(4, activation='hard_sigmoid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Dense(3, activation='selu')(x) # (1, 1, 3)

    intermediate_input_shape = (3,)
    intermediate_in = Input(intermediate_input_shape)
    intermediate_x = intermediate_in
    intermediate_x = Dense(4)(intermediate_x)
    intermediate_x = Dense(5)(intermediate_x)
    intermediate_model = Model(
        inputs=[intermediate_in], outputs=[intermediate_x],
        name='intermediate_model')
    intermediate_model.compile(loss='mse', optimizer='nadam')

    x = intermediate_model(x) # (1, 1, 5)

    intermediate_model_2 = Sequential()
    intermediate_model_2.add(Dense(7, activation='sigmoid', input_shape=(5,)))
    intermediate_model_2.add(Dense(5, activation='tanh'))
    intermediate_model_2.compile(optimizer='rmsprop',
        loss='categorical_crossentropy', metrics=['accuracy'])

    x = intermediate_model_2(x) # (1, 1, 5)

    x = Activation('sigmoid')(x)
    x = Dense(3)(x) # (1, 1, 3)

    shared_activation = Activation('tanh')
    output1 = Activation('softplus')(x)
    output2 = Activation('softmax')(x)
    output3 = shared_activation(x)
    output4 = shared_activation(input3)
    output5 = input4
    output6 = input1

    #todo strides
    conv_outputs = []
    conv_outputs.append(SeparableConv2D(2, (3, 3),
                    padding='same')(input0))
    #for padding in ['valid', 'same']:
    #    for h in range(1, 5):
    #        for sy in range(1, 5):
    #            conv_outputs.append(Conv2D(2, (1, h), strides=(1, sy),
    #                padding=padding)(input0))
    #            conv_outputs.append(SeparableConv2D(2, (1, h), strides=(sy, sy),
    #                padding=padding)(input0))
    #    for w in range(1, 5):
    #        for sx in range(1, 5):
    #            conv_outputs.append(Conv2D(2, (w, 1), strides=(sx, 1),
    #                padding=padding)(input0))
    #            conv_outputs.append(SeparableConv2D(2, (w, 1), strides=(sx, sx),
    #                padding=padding)(input0))

    model = Model(
        inputs=[input0, input1, input2, input3, input4],
        outputs=conv_outputs +
            [output1, output2, output3, output4, output5, output6],
        name='full_model')

    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data

    training_data_size = 16
    batch_size = 8
    epochs = 1

    data_in = [
        np.random.random(size=(training_data_size, *input0_shape)),
        np.random.random(size=(training_data_size, *input1_shape)),
        np.random.random(size=(training_data_size, *input2_shape)),
        np.random.random(size=(training_data_size, *input3_shape)),
        np.random.random(size=(training_data_size, *input4_shape)),
    ]

    conv_out_data = [np.random.random(
        size=(training_data_size, *x.shape[1:])) for x in conv_outputs]

    data_out = conv_out_data + [
        np.random.random(size=(training_data_size, 3)),
        np.random.random(size=(training_data_size, 3)),
        np.random.random(size=(training_data_size, 3)),
        np.random.random(size=(training_data_size, *input3_shape)),
        np.random.random(size=(training_data_size, *input4_shape)),
        np.random.random(size=(training_data_size, *input1_shape))
    ]

    model.fit(data_in, data_out, epochs=epochs, batch_size=batch_size)

    return model

def main():
    if len(sys.argv) != 2:
        print('usage: [output path]')
        sys.exit(1)
    else:
        np.random.seed(0)
        model = get_test_model_full()
        model.save(sys.argv[1])
        # Make sure model can be loaded again,
        # see https://github.com/fchollet/keras/issues/7682
        model = load_model(sys.argv[1])
        print(model.summary())

if __name__ == "__main__":
    main()
