// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <iostream>

#include "test_helpers.h"
#include "opencv_helpers.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

inline fd::float_t extract_value_from_file_path(const std::string& file_path)
{
    const auto path_split =
        fplus::split_by_token(std::string("/"), false, file_path);
    const auto filename_split =
        fplus::split_by_token(std::string("__"), false, path_split.back());
    return fplus::read_value_unsafe<fd::float_t>(filename_split.front());
}

inline fd::classification_dataset load_golf_ball_quality_dataset(const std::string& base_dir)
{
    const auto training_images_paths = list_JPEGs(base_dir + "/training");
    const auto test_images_paths = list_JPEGs(base_dir + "/test");

    const auto output_for_article = [](const fd::float_t v) -> fd::matrix3d
    {
        return fd::matrix3d(fd::size3d(1, 1, 1), {v});
    };

    const auto path_to_input_with_output = [&](const std::string& path) -> fd::input_with_output
    {
        return {
            load_col_image_as_matrix3d(28, 448, 1024, 1024, path),
            output_for_article(extract_value_from_file_path(path))};
    };

    const auto training_data =
        fplus::transform(path_to_input_with_output, training_images_paths);

    const auto test_data =
        fplus::transform(path_to_input_with_output, test_images_paths);

    fd::classification_dataset classifcation_data =
    {
        training_data, test_data
    };

    return classifcation_data;
}

inline void golf_ball_quality_regression_test()
{
    std::cout << frame_string("golf_ball_quality_regression_test") << std::endl;
    auto classifcation_data = load_golf_ball_quality_dataset("stuff/golf_ball_quality");
    assert(!classifcation_data.training_data_.empty());
    assert(!classifcation_data.test_data_.empty());

    classifcation_data = normalize_classification_dataset(classifcation_data, false);

    using namespace fd;

    const auto activation_function = elu(1);
    const auto pooling_function = gentle_max_pool(2, 0.7);
    pre_layers layers = {
        /*
        conv(size2d(4, 4), 8, 2), activation_function,
        conv(size2d(4, 4), 8, 2), activation_function,
        conv(size2d(4, 4), 8, 2), activation_function,
        conv(size2d(4, 4), 8, 2), activation_function,
        conv(size2d(4, 4), 8, 2), activation_function,
        conv(size2d(4, 4), 8, 2), activation_function,
        conv(size2d(4, 4), 8, 2), activation_function,
        */

        conv(size2d(3, 3), 8, 1), pooling_function,
        conv(size2d(3, 3), 8, 1), pooling_function,
        conv(size2d(3, 3), 8, 1), pooling_function,
        pooling_function,
        pooling_function,
        pooling_function,
        pooling_function,
        pooling_function,
        pooling_function,

        flatten(),
        tanh(),
        fc(40),
        fc(classifcation_data.training_data_.front().output_.size().height_),
        };

    auto gradnet = net(layers)(classifcation_data.training_data_.front().input_.size());
    std::cout << "net.param_count() " << gradnet->param_count() << std::endl;

    gradnet->random_init_params();

    train(gradnet, classifcation_data.training_data_, 0.05f, 0.1f, 100);
    test_regression(gradnet, classifcation_data.test_data_);
    //test_regression(gradnet, classifcation_data.training_data_);
}
