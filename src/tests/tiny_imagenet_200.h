// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <iostream>

#include "test_helpers.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.hpp>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

#include <string>
#include <vector>


inline void tiny_imagenet_200_autoencoder_test()
{
    using namespace fd;

    //const auto activation_function = leaky_relu(0.001);
    //const auto pooling_function = max_pool(2);
    const auto activation_function = elu(1);
    //const auto pooling_function = max_pool(2);
    //const auto unpooling_function = unpool(2);
    pre_layers layers = {
        conv(size2d(4, 4), 32, 2), activation_function,
        conv(size2d(4, 4), 48, 2), activation_function,
        conv(size2d(4, 4), 64, 2), activation_function,
        conv(size2d(4, 4), 96, 2), activation_function,
        conv(size2d(4, 4), 128, 2), activation_function,
        conv_transp(size2d(4, 4), 96, 2), activation_function,
        conv_transp(size2d(4, 4), 64, 2), activation_function,
        conv_transp(size2d(4, 4), 48, 2), activation_function,
        conv_transp(size2d(4, 4), 32, 2), activation_function,
        conv_transp(size2d(4, 4), 3, 2), activation_function,
        };


    std::cout << frame_string("tiny_imagenet_200_autoencoder_test") << std::endl;
    std::cout << "loading tiny_imagenet_200_ ..." << std::flush;

    const std::string bears_path = "./stuff/tiny-imagenet-200/train/n02132136/images/";
    auto bear_file_paths = fplus::sort(list_JPEGs(bears_path));
    //bear_file_paths = fplus::take(10, bear_file_paths);
    classification_dataset dataset;
    for (const auto& path : bear_file_paths)
    {
        auto img = load_matrix3d_image_bgr(path);
        //auto img = load_matrix3d_image_gray(path);
        if (img.size().height_ != 64 || img.size().height_ != 64)
        {
            std::cerr << "invalid image dimensions: " << path << std::endl;
            return;
        }
        dataset.training_data_.push_back({img, img});
    }
    std::cout << " done" << std::endl;

    //dataset = normalize_classification_dataset(dataset, false);

    /*
    // todo remove
    dataset.training_data_[0].input_ = matrix3d(size3d(1,1,2), {3,4} );
    dataset.training_data_[0].output_ = dataset.training_data_[0].input_;
    dataset.training_data_ = fplus::get_range(0, 1, dataset.training_data_);
    */

    auto tobinet = net(layers)(dataset.training_data_[0].input_.size());
    std::cout << "net.param_count() " << tobinet->param_count() << std::endl;
    tobinet->random_init_params();

    //tobinet->set_params({
      //  0.9,0.1
    //    0,0,0,0,1,0,0,0,0,0,
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
    //});
    //tobinet->set_params(randomly_change_params(tobinet->get_params(), 0.1));

    input_with_output_vec training_shuffle_dataset = dataset.training_data_;
    train(tobinet, training_shuffle_dataset, 0.001f, 0.1f, 1000, 50, 60*60, false);

    for (std::size_t i = 0; i < dataset.training_data_.size(); ++i)
    {
        const auto img = dataset.training_data_[i].input_;
        const auto out = tobinet->forward_pass(img);

        boost::filesystem::path p(bear_file_paths[i]);
        save_matrix3d_image(out, "./stuff/output_bears/" + p.stem().string() + ".png");
    }
}
