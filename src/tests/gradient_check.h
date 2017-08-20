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

inline fd::float_t relative_error(fd::float_t x, fd::float_t y)
{
    const auto divisor = fplus::max(std::abs(x), std::abs(y));
    if (divisor < 0.0001f)
        return 0;
    else
        return fplus::abs_diff(x, y) / divisor;
}

inline bool gradients_equal(fd::float_t max_diff, const fd::float_vec& xs, const fd::float_vec& ys)
{
    assert(xs.size() == ys.size());
    return fplus::all_by(
        fplus::is_less_or_equal_than<fd::float_t>(max_diff),
        fplus::zip_with(relative_error, xs, ys));
}

// http://cs231n.github.io/neural-networks-3/#gradcheck
inline void gradient_check_backprop_implementation()
{
    std::cout << frame_string("gradient_check_backprop_implementation") << std::endl;

    using namespace fd;

    const auto show_one_value = fplus::fwd::show_float_fill_left(' ', 7 + 5, 7);
    const auto show_gradient = [show_one_value](const float_vec& xs) -> std::string
    {
        return fplus::show_cont(fplus::transform(show_one_value, xs));
    };

    const auto generate_random_values = [](std::size_t count) -> float_vec
    {
        std::random_device rd; // uses seed from system automatically
        std::mt19937 gen(rd());
        std::normal_distribution<fd::float_t> d(0, 1);
        float_vec values;
        values.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            values.push_back(static_cast<fd::float_t>(d(gen)));
        }
        return values;
    };

    const auto generate_random_data = [&](
        const size3d& in_size,
        const size3d& out_size,
        std::size_t count) -> input_with_output_vec
    {
        input_with_output_vec data;
        data.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            const auto in_vals = generate_random_values(in_size.volume());
            const auto out_vals = generate_random_values(out_size.volume());
            data.push_back({{in_size, in_vals}, {out_size, out_vals}});
        }
        return data;
    };

    const auto test_net_backprop = [&](
        const std::string& name,
        layer_ptr& net,
        std::size_t data_size,
        std::size_t repetitions)
    {
        std::cout << "Testing backprop with " << name << std::endl;
        for (std::size_t i = 0; i < repetitions; ++i)
        {
            const auto data = generate_random_data(
                net->input_size(), net->output_size(), data_size);
            net->random_init_params();
            auto gradient = calc_net_gradient_numeric(net, data);
            auto gradient_backprop = calc_net_gradient_backprop(net, data);
            if (!gradients_equal(0.00001f, gradient_backprop, gradient))
            {
                std::cout << "params            " << show_gradient(net->get_params()) << std::endl;
                std::cout << "gradient          " << show_gradient(gradient) << std::endl;
                std::cout << "gradient_backprop " << show_gradient(gradient_backprop) << std::endl;
                std::cout << "abs diff          " << show_gradient(fplus::zip_with(fplus::abs_diff<fd::float_t>, gradient, gradient_backprop)) << std::endl;
                std::cout << "relative_error    " << show_gradient(fplus::zip_with(relative_error, gradient, gradient_backprop)) << std::endl;
                assert(false);
            }
        }
    };




    auto net_activation_functions = net(
    {
        flatten(),
        fc(2),
        softplus(),
        fc(2),
        identity(),
        fc(2),
        relu(),
        fc(2),
        leaky_relu(0.03f),
        fc(2),
        elu(1),
        fc(2),
        erf(),
        fc(2),
        step(),
        fc(2),
        fast_sigmoid(),
        fc(4),
        sigmoid(),
        fc(3),
        tanh(),
        fc(3),
        hard_sigmoid(),
        fc(3),
        selu()
        //softmax()
    })(size3d(1, 1, 2));
    test_net_backprop("net_activation_functions", net_activation_functions, 10, 10);





    auto conv_net_stride_1 = net(
    {
        conv(size2d(3, 3), 2, 1),
        conv(size2d(3, 3), 2, 1),
    })(size3d(1, 4, 4));
    test_net_backprop("conv_net_stride_1", conv_net_stride_1, 5, 10);


    auto conv_net_stride_2_f2 = net(
    {
        conv(size2d(2, 2), 2, 2),
        conv(size2d(2, 2), 5, 2),
        conv(size2d(2, 2), 3, 2),
    })(size3d(1, 8, 8));
    test_net_backprop("conv_net_stride_2_f2", conv_net_stride_2_f2, 5, 10);


    auto conv_net_stride_2_f4 = net(
    {
        conv(size2d(4, 4), 3, 2),
        conv(size2d(4, 4), 2, 2),
    })(size3d(1, 8, 8));
    test_net_backprop("conv_net_stride_2_f4", conv_net_stride_2_f4, 5, 10);


    auto conv_net_stride_4_f4 = net(
    {
        conv(size2d(4, 4), 1, 4),
        conv(size2d(4, 4), 1, 4),
    })(size3d(1, 16, 16));
    test_net_backprop("conv_net_stride_4_f4", conv_net_stride_4_f4, 5, 10);



    auto conv_transp_net_stride_2_f2 = net(
    {
        conv_transp(size2d(2, 2), 2, 2),
        conv_transp(size2d(2, 2), 3, 2),
    })(size3d(1, 2, 2));
    test_net_backprop("conv_transp_net_stride_2_f2", conv_transp_net_stride_2_f2, 5, 10);


    auto conv_transp_net_stride_2_f4 = net(
    {
        conv_transp(size2d(4, 4), 1, 2),
        conv_transp(size2d(4, 4), 1, 2),
    })(size3d(1, 2, 2));
    test_net_backprop("conv_transp_net_stride_2_f4", conv_transp_net_stride_2_f4, 5, 10);


    auto conv_transp_net_stride_4_f4 = net(
    {
        conv_transp(size2d(4, 4), 1, 4),
        conv_transp(size2d(4, 4), 1, 4),
    })(size3d(1, 2, 2));
    test_net_backprop("conv_transp_net_stride_4_f4", conv_transp_net_stride_4_f4, 5, 10);


    auto net_003 = net(
    {
        conv(size2d(3, 3), 1, 1),
        conv(size2d(3, 3), 1, 1),
        conv(size2d(3, 3), 1, 1),
        conv(size2d(3, 3), 1, 1),
    })(size3d(1, 5, 5));
    test_net_backprop("conv", net_003, 5, 10);




    auto net_unpool = net(
    {
        conv(size2d(3, 3), 1, 1),
        unpool(2),
        conv(size2d(3, 3), 1, 1),
    })(size3d(1, 4, 4));
    test_net_backprop("net_unpool", net_unpool, 5, 10);




    auto net_max_pool = net(
    {
        conv(size2d(3, 3), 1, 1),
        max_pool(2),
        conv(size2d(3, 3), 1, 1),
    })(size3d(1, 8, 8));
    test_net_backprop("net_max_pool", net_max_pool, 5, 10);




    auto net_avg_pool = net(
    {
        conv(size2d(3, 3), 1, 1),
        avg_pool(2),
        conv(size2d(3, 3), 1, 1),
    })(size3d(1, 8, 8));
    test_net_backprop("net_avg_pool", net_avg_pool, 5, 10);








    auto net_gentle_max_pool = net(
    {
        conv(size2d(3, 3), 1, 1),
        gentle_max_pool(2, 1),
        conv(size2d(3, 3), 1, 1),
    })(size3d(1, 8, 8));
    test_net_backprop("net_gentle_max_pool", net_gentle_max_pool, 5, 10);




    auto net_softmax = net(
    {
        fc(7),
        softmax(),
        fc(7),
    })(size3d(1, 7, 1));
    test_net_backprop("net_softmax", net_softmax, 3, 30);



    auto net_elu = net(
    {
        fc(2),
        elu(1),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_elu", net_elu, 1, 10);

    auto net_tanh_def = net(
    {
        fc(2),
        tanh(false),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_tanh_def", net_tanh_def, 1, 10);

    auto net_tanh_alpha = net(
    {
        fc(2),
        tanh(false, 0.3f),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_tanh_alpha", net_tanh_alpha, 1, 10);

    auto net_tanh_lecun = net(
    {
        fc(2),
        tanh(true),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_tanh_lecun", net_tanh_lecun, 1, 10);

    auto net_tanh_lecun_alpha = net(
    {
        fc(2),
        tanh(true, 0.2f),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_tanh_lecun_alpha", net_tanh_lecun_alpha, 1, 10);

    auto net_batch_normalization = net(
    {
        batch_normalization(0.0001),
    })(size3d(3, 4, 6));
    test_net_backprop("net_batch_normalization", net_batch_normalization, 1, 10);

    auto net_batch_normalization_flat = net(
    {
        fc(2),
        batch_normalization(0.0001),
        fc(2),
    })(size3d(1, 2, 1));
    test_net_backprop("net_batch_normalization_flat", net_batch_normalization_flat, 1, 10);

    auto net_006 = net(
    {
        conv(size2d(3, 3), 4, 1), elu(1),
        conv(size2d(1, 1), 2, 1), elu(1),
        max_pool(2),
        flatten(),
        fc(4),
        fc(2),
        //softmax()
    })(size3d(1, 4, 4));
    test_net_backprop("net_006", net_006, 5, 10);




    std::cout << frame_string("Backprop implementation seems to be correct.") << std::endl;
}
