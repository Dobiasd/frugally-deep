// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/typedefs.h"

#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.h>

#include <cassert>
#include <cstddef>
#include <random>
#include <vector>

namespace fd
{

struct input_with_output
{
    matrix3d input_;
    matrix3d output_;
};
typedef std::vector<input_with_output> input_with_output_vec;

struct classification_dataset
{
    input_with_output_vec training_data_;
    input_with_output_vec test_data_;
};

float_vec randomly_change_params(const float_vec& old_params)
{
    // todo: develop optimization strategy

    // todo seed
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    float_vec new_params = old_params;
    for (std::size_t i = 0; i < new_params.size(); ++i)
    {
        new_params[i] += d(gen);
    }
    return new_params;
}

void optimize_net(layer_ptr& net,
    const std::function<float_t(
        const layer_ptr& net,
        const input_with_output_vec& dataset)>& calc_error,
    const input_with_output_vec& dataset)
{
    auto old_error = calc_error(net, dataset);
    float_vec old_params = net->get_params();
    float_vec new_params = randomly_change_params(old_params);
    net->set_params(new_params);
    auto new_error = calc_error(net, dataset);
    std::cout << "todo remove, new_error, old_error " << new_error << ", " << old_error << std::endl;
    if (new_error > old_error)
    {
        std::cout << "todo remove net->set_params(old_params);" << std::endl;
        // todo remove
        net->set_params(old_params);
        assert(net->get_params() == old_params);
        net->set_params(new_params);
        assert(net->get_params() == new_params);
        assert(new_params != old_params);

        net->set_params(old_params);
    }
}

float_t calc_mean_error(
    const layer_ptr& net,
    const input_with_output_vec& dataset)
{
    std::vector<matrix3d> error_matrices;
    for (const auto& data : dataset)
    {
        auto result = net->forward_pass(data.input_);
        auto error = abs_diff_matrix3ds(result, data.output_);
        error_matrices.push_back(error);
        //std::cout << "todo remove show_matrix3d(result) " << show_matrix3d(result) << std::endl;
    }
    auto error_matrix_sum =
        fplus::fold_left_1(add_matrix3ds, error_matrices) /
        static_cast<float_t>(error_matrices.size());
    std::cout << "todo remove error_matrices.size() " << error_matrices.size() << std::endl;
    std::cout << "todo remove error_matrix_sum.size().width() " << error_matrix_sum.size().width() << std::endl;
    return matrix3d_sum_all_values(error_matrix_sum) /
        static_cast<float_t>(error_matrix_sum.size().width());
}

void train(layer_ptr& net,
    const input_with_output_vec& dataset,
    std::size_t max_iters,
    float_t mean_error_goal)
{
    auto show_progress = [](std::size_t iter, float_t error)
    {
        std::cout << "iteration " << iter << ", error " << error << std::endl;
    };
    for (std::size_t iter = 0; iter < max_iters; ++iter)
    {
        auto error = calc_mean_error(net, dataset);
        if (iter % 10 == 0)
        {
            show_progress(iter, error);
        }
        if (error < mean_error_goal)
        {
            show_progress(iter, error);
            return;
        }
        optimize_net(net, calc_mean_error, dataset);
    }
    show_progress(max_iters, calc_mean_error(net, dataset));
}

void test(layer_ptr& net, const input_with_output_vec& dataset)
{
    std::size_t correct_count = 0;
    for (const auto& data_idx_and_data : fplus::enumerate(dataset))
    {
        const auto& data_idx = data_idx_and_data.first;
        const auto& data = data_idx_and_data.second;
        assert(data.output_.size() == net->output_size());
        auto out_vol = net->forward_pass(data.input_);
        auto classification_result = matrix3d_max_pos(out_vol).x();
        auto wanted_result = matrix3d_max_pos(data.output_).x();
        if (classification_result == wanted_result)
        {
            ++correct_count;
        }
        int percent = fplus::round<int>(
            100 * static_cast<double>(correct_count) /
            static_cast<double>(dataset.size()));
        std::cout << "test accuracy: " << percent << " % ("
            << correct_count << "/" << data_idx << ")" << std::endl;
    }
}

} // namespace fd
