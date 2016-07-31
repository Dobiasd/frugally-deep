// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <iostream>

#include "test_helpers.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <string>
#include <vector>

std::vector<std::string> list_JPEGs(const std::string& dir_path)
{
    using namespace boost::filesystem;
    return fplus::transform_and_keep_justs([](const directory_entry& entry) -> fplus::maybe<std::string>
    {
        if (!boost::filesystem::is_regular_file(entry))
            return {};
        const auto file_name_str = entry.path().filename().string();
        if (!fplus::is_infix_of(std::string("JPEG"), file_name_str))
            return {};
        return entry.path().string();;
    }, std::vector<directory_entry>(directory_iterator(path(dir_path)), {}));
}

inline void tiny_imagenet_200_autoencoder_test()
{
    using namespace fd;

    //const auto activation_function = leaky_relu(0.001);
    //const auto pooling_function = max_pool(2);
    const auto activation_function = elu(1);
    const auto pooling_function = max_pool(2);
    const auto unpooling_function = unpool(2);
    pre_layers layers = {
/*        conv(size2d(3, 3), 32, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 48, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 64, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 96, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 128, 1), activation_function,
        pooling_function,

        conv(size2d(3, 3), 160, 1), activation_function,
        pooling_function, // down to 1*1

        conv(size2d(3, 3), 192, 1), activation_function,
        conv(size2d(3, 3), 192, 1), activation_function,

        unpooling_function,
        conv(size2d(3, 3), 160, 1), activation_function,

        unpooling_function,
        conv(size2d(3, 3), 128, 1), activation_function,

        unpooling_function,
        conv(size2d(3, 3), 96, 1), activation_function,

        unpooling_function,
        conv(size2d(3, 3), 64, 1), activation_function,

        unpooling_function,
        conv(size2d(3, 3), 48, 1), activation_function,

        unpooling_function,
        conv(size2d(3, 3), 32, 1), activation_function,
        conv(size2d(3, 3), 3, 1), activation_function,
        */
        conv(size2d(3, 3), 1, 1)//, activation_function,
        };


    std::cout << frame_string("tiny_imagenet_200_autoencoder_test") << std::endl;
    std::cout << "loading tiny_imagenet_200_ ..." << std::flush;

    const std::string bears_path = "./stuff/tiny-imagenet-200/train/n02132136/images/";
    const auto bear_file_paths = fplus::sort(list_JPEGs(bears_path));
    classification_dataset dataset;
    for (const auto& path : bear_file_paths)
    {
        //auto img = load_matrix3d_image_bgr(path);
        auto img = load_matrix3d_image_gray(path);
        if (img.size().height_ != 64 || img.size().height_ != 64)
        {
            std::cerr << "invalid image dimensions: " << path << std::endl;
            return;
        }
        dataset.training_data_.push_back({img, img});
    }
    std::cout << " done" << std::endl;

    //dataset = normalize_classification_dataset(dataset, false);

    auto tobinet = net(layers)(size3d(1, 64, 64));
    std::cout << "net.param_count() " << tobinet->param_count() << std::endl;
    tobinet->random_init_params();

    tobinet->set_params({
        0,0,0,0,1,0,0,0,0,0,
/*        0,0,0,0,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,

        0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,

        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,1,0,0,0,0,0,
        */
    });
    tobinet->set_params(randomly_change_params(tobinet->get_params(), 0.00001));

    input_with_output_vec training_shuffle_dataset = dataset.training_data_;
    train(tobinet, training_shuffle_dataset, 0.001f, 0.1f, 100, 50, 60);

    for (std::size_t i = 0; i < dataset.training_data_.size(); ++i)
    {
        const auto img = dataset.training_data_[i].input_;
        const auto out = tobinet->forward_pass(img);

        boost::filesystem::path p(bear_file_paths[i]);
        save_matrix3d_image(out, "./stuff/output_bears/" + p.stem().string() + ".png");
    }
}
