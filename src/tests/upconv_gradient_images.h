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


inline void upconv_gradient_images_test()
{
    using namespace fd;

    std::cout << frame_string("upconv_gradient_images_test") << std::endl;

    std::vector<matrix3d> images =
    {
        {size3d(1,4,4), {1,1,1,1, 0,0,0,0, 1,1,1,1, 0,0,0,0}},
        {size3d(1,4,4), {1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0}}
    };

    input_with_output_vec image_pairs = fplus::zip_with(
        [](const matrix3d& img1, const matrix3d& img2) -> input_with_output
        {
            return {img1, img2};
        }, images, images);

    classification_dataset dataset =
    {
        image_pairs,
        image_pairs
    };

    const auto activation_function = elu(1);
    pre_layers layers = {
        conv(size2d(2, 2), 4, 2), activation_function,
        conv(size2d(2, 2), 8, 2), activation_function,
        conv_transp(size2d(2, 2), 4, 2), activation_function,
        conv_transp(size2d(2, 2), 1, 2), activation_function,
        };

    auto gradnet = net(layers)(dataset.training_data_[0].input_.size());
    std::cout << "net.param_count() " << gradnet->param_count() << std::endl;

    gradnet->random_init_params();

    train(gradnet, dataset.training_data_, 0.01f, 0.01f, 100, 0, 60, false);
    test(gradnet, dataset.test_data_);
}
