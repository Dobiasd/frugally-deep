#!/usr/bin/env python3
"""Generate a test model for frugally-deep.
"""

import sys

import numpy as np

import keras
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Input, UpSampling2D, Flatten, SeparableConv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

__author__ = "Tobias Hermann"
__copyright__ = "Copyright 2017, Tobias Hermann"
__license__ = "MIT"
__maintainer__ = "Tobias Hermann, https://github.com/Dobiasd/frugally-deep"
__email__ = "editgym@gmail.com"

def get_test_model_small():
    image_format = K.image_data_format()
    input_shapes = [
        (6, 8, 1) if image_format == 'channels_last' else (1, 6, 8)
        ]

    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []
    outputs.append(Conv2D(2, (3, 5), strides=(3, 3), padding='same')(inputs[0]))

    model = Model(inputs=inputs, outputs=outputs, name='test_model_small')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 1
    data_in = [np.random.random(size=(training_data_size, *input_shape))
        for input_shape in input_shapes]
    data_out = [np.random.random(size=(training_data_size, *x.shape[1:]))
        for x in outputs]
    model.fit(data_in, data_out, epochs=10)
    return model

def get_test_model_full():
    image_format = K.image_data_format()
    input_shapes = [
        (6, 8, 3) if image_format == 'channels_last' else (3, 6, 8),
        (4, 4, 3) if image_format == 'channels_last' else (3, 4, 4),
        (4, 4, 3) if image_format == 'channels_last' else (3, 4, 4),
        (4,),
        (2, 3),
        (7, 9, 1) if image_format == 'channels_last' else (1, 7, 9),
    ]
    inputs = [Input(shape=s) for s in input_shapes]

    outputs = []
    for inp in [inputs[0], inputs[5]]:
        for padding in ['valid', 'same']:
            for h in range(1, 6):
                for sy in range(1, 4):
                    outputs.append(Conv2D(2, (h, 1), strides=(1, sy),
                        padding=padding)(inp))
                    outputs.append(SeparableConv2D(2, (h, 1), strides=(sy, sy),
                        padding=padding)(inp))
            for w in range(1, 6):
                for sx in range(1, 4):
                    outputs.append(Conv2D(2, (1, w), strides=(sx, 1),
                        padding=padding)(inp))
                    outputs.append(SeparableConv2D(2, (1, w), strides=(sx, sx),
                        padding=padding)(inp))
    outputs.append(SeparableConv2D(2, (3, 3), use_bias=False)(inputs[0]))
    outputs.append(ZeroPadding2D(2)(inputs[0]))
    outputs.append(ZeroPadding2D((2, 3))(inputs[0]))
    outputs.append(ZeroPadding2D(((1, 2), (3, 4)))(inputs[0]))

    shared_conv = Conv2D(1, (1, 1),
        padding='valid', name='shared_conv', activation='relu')

    up_scale_2 = UpSampling2D((2, 2))
    x1 = shared_conv(up_scale_2(inputs[1])) #(1, 8, 8)
    x1 = LeakyReLU()(x1)
    x2 = shared_conv(up_scale_2(inputs[2])) #(1, 8, 8)
    x2 = ELU()(x2)
    x3 = Conv2D(1, (1, 1), padding='valid')(up_scale_2(inputs[2])) #(1, 8, 8)
    x = keras.layers.concatenate([x1, x2, x3]) #(3, 8, 8)

    x = Conv2D(3, (1, 1), padding='same', use_bias=False)(x) #(3, 8, 8)
    x = BatchNormalization(center=False)(x)
    x = Dropout(0.5)(x)

    x = keras.layers.concatenate([
        MaxPooling2D((2,2))(x),
        AveragePooling2D((2,2))(x)]) #(6, 4, 4)

    x = Flatten()(x) #(1, 1, 96)
    x = Dense(4, activation='hard_sigmoid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Dense(3, activation='selu')(x) #(1, 1, 3)

    intermediate_input_shape = (3,)
    intermediate_in = Input(intermediate_input_shape)
    intermediate_x = intermediate_in
    intermediate_x = Dense(4)(intermediate_x)
    intermediate_x = Dense(5)(intermediate_x)
    intermediate_model = Model(
        inputs=[intermediate_in], outputs=[intermediate_x],
        name='intermediate_model')
    intermediate_model.compile(loss='mse', optimizer='nadam')

    x = intermediate_model(x) #(1, 1, 5)

    intermediate_model_2 = Sequential()
    intermediate_model_2.add(Dense(7, activation='sigmoid', input_shape=(5,)))
    intermediate_model_2.add(Dense(5, activation='tanh'))
    intermediate_model_2.compile(optimizer='rmsprop',
        loss='categorical_crossentropy')

    x = intermediate_model_2(x) #(1, 1, 5)

    x = Activation('sigmoid')(x)
    x = Dense(3)(x) #(1, 1, 3)

    shared_activation = Activation('tanh')

    outputs = outputs + [
        Activation('softplus')(x),
        Activation('softmax')(x),
        shared_activation(x),
        shared_activation(inputs[3]),
        inputs[4],
        inputs[1]
    ]

    model = Model(inputs=inputs, outputs=outputs, name='test_model_full')
    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    training_data_size = 1
    batch_size = 1
    epochs = 10
    data_in = [np.random.random(size=(training_data_size, *input_shape))
        for input_shape in input_shapes]
    data_out = [np.random.random(size=(training_data_size, *x.shape[1:]))
        for x in outputs]
    model.fit(data_in, data_out, epochs=epochs, batch_size=batch_size)
    return model

def main():
    if len(sys.argv) != 2:
        print('usage: [output path]')
        sys.exit(1)
    else:
        np.random.seed(0)
        model = get_test_model_small()
        #model = get_test_model_full()
        model.save(sys.argv[1])
        # Make sure model can be loaded again,
        # see https://github.com/fchollet/keras/issues/7682
        model = load_model(sys.argv[1])
        print(model.summary())

if __name__ == "__main__":
    main()
