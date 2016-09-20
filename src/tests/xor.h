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

inline void xor_as_net_test()
{
    std::cout << frame_string("xor_as_net_test") << std::endl;

    using namespace fd;

    input_with_output_vec xor_table =
    {
       {{size3d(1,2,1), {0, 0}}, {size3d(1,1,1), {0}}},
       {{size3d(1,2,1), {0, 1}}, {size3d(1,1,1), {1}}},
       {{size3d(1,2,1), {1, 0}}, {size3d(1,1,1), {1}}},
       {{size3d(1,2,1), {1, 1}}, {size3d(1,1,1), {0}}},
    };

    classification_dataset classifcation_data =
    {
        xor_table,
        xor_table
    };

    classifcation_data = normalize_classification_dataset(classifcation_data, false);

    pre_layers layers = {
        fc(4),
        tanh(),
        fc(4),
        tanh(),
        fc(1),
        tanh(),
        };

    pre_layers layers_min = {
        fc(2),
        tanh(),
        fc(1),
        tanh(),
        };
    float_vec layers_min_good_params =
    {
         1, 1, -1, -1,
         0.5f, -1.5f,
         1, 1,
         1.5f
    };

    auto xor_net = net(layers_min)(size3d(1, 2, 1));
    std::cout << "net.param_count() " << xor_net->param_count() << std::endl;

    //xor_net->set_params(layers_min_good_params);
    xor_net->random_init_params();
    train(xor_net, classifcation_data.training_data_, 0.1f, 0.01f, 1000);
    test(xor_net, classifcation_data.test_data_);
}
