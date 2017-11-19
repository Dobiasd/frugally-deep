# todo

empty node-result cache when possible to save RAM?

reduce memory usage during model conversion (decode floats)

reduce memory usage during model loading

implement Conv2DTranspose as soon as output shape bug is fixed: https://github.com/fchollet/keras/issues/6777

add tests for Conv2DTranspose with dilation when it actually supports it: https://github.com/fchollet/keras/issues/8159

make whole model/prediction templatable with the data type (currently float32)