#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

inputs = tf.keras.layers.Input(shape=(4,))
x = tf.keras.layers.Dense(5, activation='relu')(inputs)
predictions = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='nadam')

model.fit(
    np.asarray([[1, 2, 3, 4], [2, 3, 4, 5]]),
    np.asarray([[1, 0, 0], [0, 0, 1]]), epochs=10)

model.save('readme_example_model.h5', include_optimizer=False)
