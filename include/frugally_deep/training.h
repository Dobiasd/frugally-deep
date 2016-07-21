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

// http://stackoverflow.com/a/19471595/1866775
class timer
{
public:
    timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

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

// todo https://en.wikipedia.org/wiki/Feature_scaling
// x' = (x - mean(x)) / stddev(x)
classification_dataset normalize_classification_dataset(
    const classification_dataset& dataset,
    bool normalize_output_too)
{
    float_t min_value = std::numeric_limits<float_t>::max();
    float_t max_value = std::numeric_limits<float_t>::min();

    const auto update_min_and_max_value = [&]
        (const std::pair<float_t, float_t>& min_max_value)
    {
        min_value = std::min(min_value, min_max_value.first);
        max_value = std::max(max_value, min_max_value.second);
    };

    for (const auto& in_with_out : dataset.training_data_)
    {
        update_min_and_max_value(matrix3d_min_max_value(in_with_out.input_));
        if (normalize_output_too)
        {
            update_min_and_max_value(matrix3d_min_max_value(in_with_out.output_));
        }
    }
    for (const auto& in_with_out : dataset.test_data_)
    {
        update_min_and_max_value(matrix3d_min_max_value(in_with_out.input_));
        if (normalize_output_too)
        {
            update_min_and_max_value(matrix3d_min_max_value(in_with_out.output_));
        }
    }

    const float_t value_factor = 2 / (max_value - min_value);
    const auto adjust_value = [min_value, value_factor]
        (float_t val) -> float_t
    {
        return ((val - min_value) * value_factor) - 1;
    };

    const auto adjust_mat = fplus::bind_1st_of_2(
        transform_matrix3d<decltype(adjust_value)>, adjust_value);
    const auto adjust_input_and_output = [adjust_mat, normalize_output_too]
        (const input_with_output& in_with_out) -> input_with_output
    {
        return {
            adjust_mat(in_with_out.input_),
            normalize_output_too ? adjust_mat(in_with_out.output_) : in_with_out.output_
        };
    };
    return {
        fplus::transform(adjust_input_and_output, dataset.training_data_),
        fplus::transform(adjust_input_and_output, dataset.test_data_)
    };
}

float_vec randomly_change_params(const float_vec& old_params, float_t stddev)
{
    // todo seed
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float_t> d(0, stddev);
    float_vec new_params = old_params;
    for (std::size_t i = 0; i < new_params.size(); ++i)
    {
        new_params[i] += static_cast<float_t>(d(gen));
    }
    return new_params;
}

void optimize_net_random(layer_ptr& net,
    const std::function<float_t(
        const layer_ptr& net,
        const input_with_output_vec& dataset)>& calc_error,
    const input_with_output_vec& dataset)
{
    auto old_error = calc_error(net, dataset);
    float_vec old_params = net->get_params();
    float_vec new_params = randomly_change_params(old_params, 0.1f);
    net->set_params(new_params);
    auto new_error = calc_error(net, dataset);
    if (new_error > old_error)
    {
        net->set_params(old_params);
    }
}

float_t test_params(
    const float_vec& params,
    layer_ptr& net,
    const std::function<float_t(
        const layer_ptr& net,
        const input_with_output_vec& dataset)>& calc_error,
    const input_with_output_vec& dataset)
{
    float_vec old_params = net->get_params();
    net->set_params(params);
    float_t error = calc_error(net, dataset);
    net->set_params(old_params);
    return error;
}

void optimize_net_gradient(
    layer_ptr& net,
    const std::function<float_t(
        const layer_ptr& net,
        const input_with_output_vec& dataset)>& calc_error,
    const input_with_output_vec& dataset,
    float_t& speed_factor)
{
    const float_t gradient_delta = 0.001f;

    auto calculate_gradient_dim =
        [&net, gradient_delta, &calc_error, &dataset]
        (const float_vec& curr_params, std::size_t i) -> float_t
    {
        float_vec params_plus = curr_params;
        float_vec params_minus = curr_params;

        params_plus[i] += gradient_delta;
        params_minus[i] -= gradient_delta;

        float_t plus_error = test_params(params_plus, net, calc_error, dataset);
        float_t minus_error = test_params(params_minus, net, calc_error, dataset);

        auto gradient = (plus_error - minus_error) / (2 * gradient_delta);
        return gradient;
    };

    float_vec old_params = net->get_params();
    float_vec gradient(old_params.size(), 0);

    for (std::size_t i = 0; i < old_params.size(); ++i)
    {
        float_t dim_gradient = calculate_gradient_dim(old_params, i);
        gradient[i] = dim_gradient;
    }

    float_vec new_params = old_params;
    //std::cout << "gradient " << fplus::show_cont(gradient) << std::endl;
    for (std::size_t i = 0; i < old_params.size(); ++i)
    {
        new_params[i] -= gradient[i] * speed_factor;
    }

    float_t old_error = calc_error(net, dataset);
    float_t new_error = test_params(new_params, net, calc_error, dataset);

    if (new_error >= old_error)
    {
        speed_factor *= 0.99f;
    }

    net->set_params(new_params);
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
    //std::cout << "todo remove ____" << std::endl;
    auto error_matrix_sum =
        fplus::fold_left_1(add_matrix3ds, error_matrices) /
        static_cast<float_t>(error_matrices.size());
    //std::cout << "todo remove error_matrices.size() " << error_matrices.size() << std::endl;
    //std::cout << "todo remove eFrror_matrix_sum.size().width_ " << error_matrix_sum.size().width_ << std::endl;
    return matrix3d_sum_all_values(error_matrix_sum) /
        static_cast<float_t>(error_matrix_sum.size().width_);
}

void train(layer_ptr& net,
    const input_with_output_vec& dataset,
    std::size_t max_iters,
    float_t mean_error_goal,
    float_t learning_rate)
{
    auto show_progress = [](std::size_t iter, float_t error, float_t current_learning_rate)
    {
        std::cout << "iteration " << fplus::to_string_fill_left(' ', 10, iter)
        << ", learning rate " << fplus::fill_left(' ', 15, fplus::show_float<float_t>(1, 10)(current_learning_rate))
        << ", error " << fplus::fill_left(' ', 15, fplus::show_float<float_t>(1, 10)(error))
        << std::endl;
    };
    auto show_params = [](const layer_ptr& current_net)
    {
        std::cout << "params " << fplus::show_cont(current_net->get_params()) << std::endl;
    };
    timer stopwatch;
    for (std::size_t iter = 0; iter < max_iters; ++iter)
    {
        auto error = calc_mean_error(net, dataset);
        if (iter == 0 || stopwatch.elapsed() > 0.5)
        {
            stopwatch.reset();
            show_progress(iter, error, learning_rate);
            show_params(net);
        }
        if (error < mean_error_goal || learning_rate < 0.0000001f)
        {
            show_progress(iter, error, learning_rate);
            show_params(net);
            return;
        }
        optimize_net_gradient(net, calc_mean_error, dataset, learning_rate);
    }
    show_progress(max_iters, calc_mean_error(net, dataset), learning_rate);
    show_params(net);
}

void test(layer_ptr& net, const input_with_output_vec& dataset)
{
    std::cout << "running test" << std::endl;
    std::size_t correct_count = 0;
    for (const auto& data : dataset)
    {
        assert(data.output_.size() == net->output_size());
        auto out_vol = net->forward_pass(data.input_);
        auto classification_result = matrix3d_max_pos(out_vol).x_;
        auto wanted_result = matrix3d_max_pos(data.output_).x_;
        if (classification_result == wanted_result)
        {
            ++correct_count;
        }
    }
    int percent = fplus::round<int>(
        100 * static_cast<double>(correct_count) /
        static_cast<double>(dataset.size()));
    std::cout << "test accuracy: " << percent << " % ("
        << correct_count << "/" << dataset.size() << ")" << std::endl;
}

} // namespace fd
