# todo

implement Conv2DTranspose

empty node-result cache when possible to save RAM?

reduce memory usage during model conversion (decode floats)

reduce memory usage during model loading

add tests for Conv2DTranspose with dilation when it actually supports it: https://github.com/fchollet/keras/issues/8159

make whole model/prediction templatable with the data type (currently float32)

test also with tf.keras