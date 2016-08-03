// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <iostream>

#include "test_helpers.h"
#include "loaders/mnist.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

inline void mnist_classification_test()
{
    using namespace fd;

    //const auto activation_function = leaky_relu(0.001);
    //const auto pooling_function = max_pool(2);
    const auto activation_function = elu(1);
    const auto pooling_function = gentle_max_pool(2, 0.8);
    pre_layers layers_with_pool = {
        conv(size2d(3, 3), 16, 1), activation_function,
        conv(size2d(3, 3), 16, 1), activation_function,
        pooling_function,
        conv(size2d(3, 3), 24, 1), activation_function,
        conv(size2d(3, 3), 24, 1), activation_function,
        pooling_function,
        conv(size2d(3, 3), 32, 1), activation_function,
        conv(size2d(3, 3), 32, 1), activation_function,
        conv(size2d(3, 3), 16, 1), activation_function,
        flatten(),
        fc(300),
        tanh(true),
        fc(10),
        tanh(true),
        softmax(),
        };

    pre_layers layers = {
        conv(size2d(3, 3), 16, 1), activation_function,
        conv(size2d(4, 4), 16, 2), activation_function,
        conv(size2d(3, 3), 24, 1), activation_function,
        conv(size2d(4, 4), 24, 2), activation_function,
        conv(size2d(3, 3), 32, 1), activation_function,
        conv(size2d(3, 3), 32, 1), activation_function,
        conv(size2d(3, 3), 16, 1), activation_function,
        flatten(),
        fc(300),
        tanh(true),
        fc(10),
        tanh(true),
        softmax(),
        };

    std::cout << frame_string("mnist_classification_test") << std::endl;
    std::cout << "loading mnist ..." << std::flush;
    auto classifcation_data = read_mnist("./stuff/mnist");
    std::cout << " done" << std::endl;

    // todo remove
    //classifcation_data.training_data_ = fplus::take(600, classifcation_data.training_data_);
    //classifcation_data.test_data_ = fplus::take(100, classifcation_data.test_data_);

    classifcation_data = normalize_classification_dataset(classifcation_data, false);

    auto tobinet = net(layers)(size3d(1, 28, 28));
    std::cout << "net.param_count() " << tobinet->param_count() << std::endl;
    tobinet->random_init_params();
    train(tobinet, classifcation_data.training_data_,
        0.001f, 0.1f, 100, 64, 5*60);
    //test(tobinet, classifcation_data.training_data_);
    test(tobinet, classifcation_data.test_data_);
    std::cout << frame_string("mnist-tobi-net elu(1) gentle_max_pool(2, 0.8)") << std::endl;
}
