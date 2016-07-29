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


inline fd::classification_dataset load_gradient_dataset(const std::string& base_dir)
{
    fd::classification_dataset classifcation_data =
    {
        {
            {load_gray_image_as_matrix3d(base_dir + "/training/x/001.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/x/002.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/x/003.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/y/001.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/y/002.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/y/003.png"), {fd::size3d(1,2,1), {0,1}}}
        },
        {
            {load_gray_image_as_matrix3d(base_dir + "/test/x/001.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/x/002.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/x/003.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/y/001.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/y/002.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/y/003.png"), {fd::size3d(1,2,1), {0,1}}}
        }
    };

    return classifcation_data;
}

inline void gradients_classification_test()
{
    std::cout << frame_string("gradients_classification_test") << std::endl;
    auto classifcation_data = load_gradient_dataset("test_images/datasets/classification/gradients");
    assert(!classifcation_data.training_data_.empty());
    assert(!classifcation_data.test_data_.empty());

    classifcation_data = normalize_classification_dataset(classifcation_data, false);

    using namespace fd;

    pre_layers layers = {
        conv(size2d(3, 3), 2, 1),
        elu(1),
        max_pool(32),
        flatten(),
        fc(2),
        //sigmoid(),
        softmax()
        };

    auto gradnet = net(layers)(size3d(1, 32, 32));
    std::cout << "net.param_count() " << gradnet->param_count() << std::endl;

    float_vec good_params =
    {
         3,  0,  -3,
        10,  0, -10,
         3,  0,  -3,
        0,
         3,  10,  3,
         0,   0,  0,
        -3, -10, -3,
        0,
        1,0,0,1,0,0
    };

    //gradnet->set_params(good_params);

    gradnet->random_init_params();
    //gradnet->set_params(fd::randomly_change_params(gradnet->get_params(), 0.3));

    train(gradnet, classifcation_data.training_data_, 0.01f, 0.1f, 100);
    test(gradnet, classifcation_data.test_data_);
}
