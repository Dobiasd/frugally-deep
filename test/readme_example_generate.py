#!/usr/bin/env python3

import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(4,))
x = Dense(5, activation='relu')(inputs)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='nadam')

model.fit(
    np.asarray([[1, 2, 3, 4], [2, 3, 4, 5]]),
    np.asarray([[1, 0, 0], [0, 0, 1]]), epochs=10)

model.save('readme_example_model.h5', include_optimizer=False)
