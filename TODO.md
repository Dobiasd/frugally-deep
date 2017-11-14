# todo

use libEigen
    https://www.reddit.com/r/cpp/comments/7c91n0/frugallydeep_a_headeronly_library_for_using_keras/dppvf74/

empty node-result cache when possible to save RAM?

use faster GEMM to make im2col worthwhile

reduce memory usage during model conversion (decode floats)

reduce memory usage during model loading

implement Conv2DTranspose as soon as output shape bug is fixed: https://github.com/fchollet/keras/issues/6777

add tests for Conv2DTranspose with dilation when it actually supports it: https://github.com/fchollet/keras/issues/8159

make whole model/prediction templatable with the data type (currently float32)