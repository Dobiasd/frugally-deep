// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#include "tests/cifar10.h"
#include "tests/gradient_check.h"
#include "tests/gradient_images.h"
#include "tests/lenna_filter.h"
#include "tests/mnist.h"
#include "tests/test_helpers.h"
#include "tests/xor.h"
#include "tests/variance_inflation_training.h"
#include "tests/tiny_imagenet_200.h"
#include "applications/pharmaceutical_packages.h"
#include "applications/golf_ball_quality.h"
#include "tests/upconv_gradient_images.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

int main()
{
    //gradient_check_backprop_implementation();
    golf_ball_quality_regression_test();
    return 0;
    tiny_imagenet_200_autoencoder_test();
    upconv_gradient_images_test();
    lenna_filter_test();
    xor_as_net_test();
    gradients_classification_test();
    mnist_classification_test();
    cifar_10_classification_test();
    variance_inflation_training_test();
    pharmaceutical_packages_classification_test();
}
