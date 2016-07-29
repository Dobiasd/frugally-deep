// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <iostream>

#include "test_helpers.h"
#include "loaders/cifar10.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

inline void cifar_10_classification_test()
{
    using namespace fd;

    //const auto activation_function = leaky_relu(0.001);
    //const auto pooling_function = max_pool(2);
    const auto activation_function = elu(1);
    const auto pooling_function = gentle_max_pool(2, 0.7);
    pre_layers layers = {
        conv(size2d(3, 3), 32, 1), activation_function,
        conv(size2d(3, 3), 32, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 64, 1), activation_function,
        conv(size2d(3, 3), 64, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 128, 1), activation_function,
        conv(size2d(3, 3), 128, 1), activation_function,
        pooling_function,

        //conv(size2d(3, 3), 64, 1), elu(1),
        //conv(size2d(3, 3), 64, 1), elu(1),
        //max_pool(2),

        //conv(size2d(3, 3), 128, 1), elu(1),
        //conv(size2d(1, 1), 128, 1), elu(1),
        //max_pool(2),

        flatten(),
        fc(100),
        //tanh(true),
        fc(10),
        //tanh(true),
        softmax(),
        };

    // http://cs231n.github.io/convolutional-networks/
    pre_layers layers_simple_cs231n = {
        conv(size2d(3, 3), 12, 1),
        relu(),
        max_pool(2),
        flatten(),
        fc(10),
        softmax(),
        };

    pre_layers layers_simple = {
        conv(size2d(3, 3), 12, 1),
        elu(1),
        gentle_max_pool(2, 0.7),
        flatten(),
        fc(10),
        softmax(),
        };

    pre_layers layers_linear = {
        flatten(),
        fc(10),
        softmax(),
        };



    std::cout << frame_string("cifar_10_classification_test") << std::endl;
    std::cout << "loading cifar-10 ..." << std::flush;
    auto classifcation_data = load_cifar_10_bin(
        "./stuff/cifar-10-batches-bin", false, false);
    std::cout << " done" << std::endl;

    // todo remove
    //classifcation_data.training_data_ = fplus::take(500, classifcation_data.training_data_);
    //classifcation_data.test_data_ = fplus::take(100, classifcation_data.test_data_);

    classifcation_data = normalize_classification_dataset(classifcation_data, false);

    auto tobinet = net(layers)(size3d(3, 32, 32));
    std::cout << "net.param_count() " << tobinet->param_count() << std::endl;
    tobinet->random_init_params();
    train(tobinet, classifcation_data.training_data_, 0.001f, 0.1f, 10, 64, 3600*6);
    //test(tobinet, classifcation_data.training_data_);
    test(tobinet, classifcation_data.test_data_);
    std::cout << frame_string("cifar-10-tobi-net elu(1) gentle_max_pool(2, 0.7)") << std::endl;
}
