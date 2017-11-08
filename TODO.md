# todo

release:
- re-enable logo
- github project description: Header-only library to use Keras models in C++.
- add github project tags
- post on
 - https://www.reddit.com/r/deeplearning/
 - https://www.reddit.com/r/cpp/
 - https://www.reddit.com/r/programming/
 - https://www.reddit.com/r/KerasML/
 - https://www.reddit.com/r/MachineLearning/

empty node-result cache when possible to save RAM?

use faster GEMM to make im2col worthwhile

reduce memory usage during model conversion (decode floats)

reduce memory usage during model loading

implement Conv2DTranspose as soon as output shape bug is fixed: https://github.com/fchollet/keras/issues/6777

add tests for Conv2DTranspose with dilation when it actually supports it: https://github.com/fchollet/keras/issues/8159
