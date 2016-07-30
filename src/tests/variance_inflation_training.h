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

inline void variance_inflation_training_test()
{
    using namespace fd;

    //const auto activation_function = leaky_relu(0.001);
    //const auto pooling_function = max_pool(2);
    const auto activation_function = elu(1);
    const auto pooling_function = gentle_max_pool(2, 0.7);
    pre_layers layers = {
        conv(size2d(3, 3), 8, 1), activation_function,
        conv(size2d(3, 3), 8, 1), activation_function,
        pooling_function,
        conv(size2d(3, 3), 12, 1), activation_function,
        conv(size2d(3, 3), 12, 1), activation_function,
        pooling_function,
        conv(size2d(3, 3), 16, 1), activation_function,
        conv(size2d(3, 3), 16, 1), activation_function,
        flatten(),
        fc(200),
        tanh(true),
        fc(10),
        tanh(true),
        softmax(),
        };


    std::cout << frame_string("variance_inflation_training_test") << std::endl;
    std::cout << "loading mnist ..." << std::flush;
    auto classifcation_data = read_mnist("./stuff/mnist");
    std::cout << " done" << std::endl;

    classifcation_data.training_data_ = fplus::take(600, classifcation_data.training_data_);
    classifcation_data.test_data_ = fplus::take(100, classifcation_data.test_data_);

    classifcation_data = normalize_classification_dataset(classifcation_data, false);


    auto layer0 = layers[0](size3d(1, 28, 28));
    layer0->random_init_params();

    typedef std::vector<matrix3d> matrix3ds;
    typedef std::vector<matrix3ds> matrix3dss;
    matrix3dss mss(10, matrix3ds());

    const auto& get_max_y_pos = [](const matrix3d& m) -> std::size_t
    {
        return matrix3d_max_pos(m).y_;
    };

    const auto& train_data = classifcation_data.training_data_;
    for (const auto& dat : train_data)
    {
        const auto output = layer0->forward_pass(dat.input_);
        mss[get_max_y_pos(dat.output_)].push_back(output);
    }

    const auto ms = fplus::transform(mean_matrix3d, mss);

    const auto mean_output = mean_matrix3d(fplus::concat(mss));

    //std::cout << show_matrix3d(mean_output) << std::endl;

    auto class_mean_outputs_corrected = fplus::transform(
        [&](const matrix3d& m) -> matrix3d
    {
        return m - mean_output;
    }, ms);

    // replace normal output data with wanted first layer output
    auto data_backprop = train_data;
    for (auto& data : data_backprop)
    {
        const auto dest =
            class_mean_outputs_corrected[get_max_y_pos(data.output_)];
        data.output_ = dest;
    }
    const auto gradient = calc_net_gradient_backprop(layer0, data_backprop);

    float_vec momentum(layer0->param_count(), 0);
    fd::float_t learning_rate = 0.1;
    const auto old_and_new_error = optimize_net_gradient(
        layer0, data_backprop, learning_rate, momentum, gradient);

    const fd::float_t old_error = old_and_new_error.first;
    const fd::float_t new_error = old_and_new_error.second;

    show_progress(0,0,0,learning_rate,old_error,new_error,
        fplus::mean_stddev<fd::float_t>(layer0->get_params()),
        fplus::mean_stddev<fd::float_t>(momentum));

    auto tobinet = net(layers)(size3d(1, 28, 28));
    std::cout << "net.param_count() " << tobinet->param_count() << std::endl;
    tobinet->random_init_params();
    std::cout << frame_string("variance_inflation_training_test") << std::endl;
}
