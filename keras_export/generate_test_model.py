import keras
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Input, UpSampling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import numpy as np

import sys

np.random.seed(0)

def get_test_model_small():
    inputs = Input(shape=(10,))
    x = Dense(3)(inputs)
    x = Dense(4)(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model

def get_test_model_full():
    input_shape =\
        (2, 3, 3) if K.image_data_format() == 'channels_last' else (3, 2, 3)

    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

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
    x = BatchNormalization()(x)
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

    x = Activation('sigmoid')(x)
    x = Dense(3)(x)

    output1 = Activation('tanh')(x)
    output2 = Activation('softmax')(x)
    shared_activation = Activation('softplus')
    output3 = shared_activation(x)
    output4 = shared_activation(x)

    model = Model(
        inputs=[input1, input2],
        outputs=[output1, output2, output3, output4],
        name='full_model')

    model.compile(loss='mse', optimizer='nadam')

    # fit to dummy data
    data_in = np.random.random(size=(2, 16, *input_shape))
    data_out = np.random.random(size=(4, 16, 3))
    model.fit(list(data_in), list(data_out), epochs=1, batch_size=8)
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
