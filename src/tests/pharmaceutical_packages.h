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

inline std::string extract_article_from_file_path(const std::string& file_path)
{
    const auto path_split =
        fplus::split_by_token(std::string("/"), false, file_path);
    const auto filename_split =
        fplus::split_by_token(std::string("__"), false, path_split.back());
    return filename_split.front();
}

inline fd::classification_dataset load_pharmaceutical_packages_dataset(const std::string& base_dir)
{
    const auto training_images_paths = list_JPEGs(base_dir + "/training");
    const auto test_images_paths = list_JPEGs(base_dir + "/test");

    const auto article_ids =
        fplus::nub(
            fplus::transform(extract_article_from_file_path,
                fplus::append(
                    training_images_paths, test_images_paths)));

    const auto article_ids_to_class_ids =
        fplus::pairs_to_map<std::map<std::string, std::size_t>>(
            fplus::transform(
                fplus::swap_pair_elems<std::size_t, std::string>,
                fplus::enumerate(article_ids)));

    const auto output_for_article = [&](const std::string& article)
    {
        const fd::size3d output_layer_size(1, article_ids_to_class_ids.size(), 1);
        fd::matrix3d result(output_layer_size);
        result.set(0,
            fplus::get_from_map_unsafe(article_ids_to_class_ids, article), 0, 1);
        return result;
    };

    const auto path_to_input_with_output = [&](const std::string& path) -> fd::input_with_output
    {
        return {
            load_col_image_as_matrix3d(256, 256, path),
            output_for_article(extract_article_from_file_path(path))};
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

inline void pharmaceutical_packages_classification_test()
{
    std::cout << frame_string("pharmaceutical_packages_classification_test") << std::endl;
    auto classifcation_data = load_pharmaceutical_packages_dataset("stuff/pharmaceutical_packages");
    assert(!classifcation_data.training_data_.empty());
    assert(!classifcation_data.test_data_.empty());

    classifcation_data = normalize_classification_dataset(classifcation_data, false);

    using namespace fd;

    const auto activation_function = elu(1);
    const auto pooling_function = gentle_max_pool(2, 0.7);
    pre_layers layers = {
        conv(size2d(3, 3), 16, 1), activation_function,
        conv(size2d(3, 3), 16, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 24, 1), activation_function,
        conv(size2d(3, 3), 24, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 32, 1), activation_function,
        conv(size2d(3, 3), 32, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 48, 1), activation_function,
        conv(size2d(3, 3), 48, 1), activation_function,
        pooling_function, // 16*16

        conv(size2d(3, 3), 64, 1), activation_function,
        conv(size2d(3, 3), 64, 1), activation_function,
        pooling_function, // 8*8

        conv(size2d(3, 3), 96, 1), activation_function,
        conv(size2d(3, 3), 96, 1), activation_function,

        flatten(),
        fc(200),
        tanh(true),
        fc(classifcation_data.training_data_.front().output_.size().height_),
        tanh(true),
        softmax(),
        };

    auto gradnet = net(layers)(size3d(3, 256, 256));
    std::cout << "net.param_count() " << gradnet->param_count() << std::endl;

    gradnet->random_init_params();

    train(gradnet, classifcation_data.training_data_, 0.1f, 0.1f, 100);
    test(gradnet, classifcation_data.test_data_);
}
