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

inline float_vec flatten_classification_dataset(
    const classification_dataset& dataset,
    bool include_output,
    bool include_test_data)
{
    const auto flatten_input_and_output = [&]
        (const input_with_output& in_with_out) -> float_vec
    {
        return fplus::append(
            in_with_out.input_.as_vector(),
            include_output
                ? in_with_out.output_.as_vector()
                : float_vec()
            );
    };

    return fplus::append(
        fplus::transform_and_concat(flatten_input_and_output, dataset.training_data_),
        include_test_data
            ? fplus::transform_and_concat(flatten_input_and_output, dataset.test_data_)
            : float_vec()
    );
}

template <typename F>
classification_dataset transform_classification_dataset(
    const F& f,
    const classification_dataset& dataset,
    bool normalize_output_too)
{
    const auto transform_input_and_output = [&]
        (const input_with_output& in_with_out) -> input_with_output
    {
        return {
            transform_matrix3d(f, in_with_out.input_),
            normalize_output_too
                ? transform_matrix3d(f, in_with_out.output_)
                : in_with_out.output_
            };
    };

    return
    {
        fplus::transform(transform_input_and_output, dataset.training_data_),
        fplus::transform(transform_input_and_output, dataset.test_data_)
    };
}

// zero mean/unit variance
// https://en.wikipedia.org/wiki/Feature_scaling
inline classification_dataset normalize_classification_dataset(
    const classification_dataset& dataset,
    bool normalize_output_too)
{
    // Calculate mean and stddev only on the training set.
    // And apply it to training set and test set.
    const float_vec all_values =
        flatten_classification_dataset(dataset, normalize_output_too, false);
    const auto mean_and_stddev = fplus::mean_stddev<float_t>(all_values);
    const float_t mean = mean_and_stddev.first;
    const float_t stddev = fplus::max(mean_and_stddev.second, 0.00000001f);
    const auto f = [mean, stddev](float_t x)
    {
        return (x - mean) / stddev;
    };
    return transform_classification_dataset(f, dataset, normalize_output_too);
}

inline float_vec randomly_change_params(const float_vec& old_params, float_t stddev)
{
    std::random_device rd; // uses seed from system automatically
    std::mt19937 gen(rd());
    std::normal_distribution<float_t> d(0, stddev);
    float_vec new_params = old_params;
    for (std::size_t i = 0; i < new_params.size(); ++i)
    {
        new_params[i] += static_cast<float_t>(d(gen));
    }
    return new_params;
}

inline std::vector<matrix3d> calc_all_output_errors(
    const layer_ptr& net,
    const input_with_output_vec& dataset)
{
    std::vector<matrix3d> error_matrices;
    for (const auto& data : dataset)
    {
        auto result = net->forward_pass(data.input_);
        auto error = sub_matrix3d(result, data.output_);
        error_matrices.push_back(error);
    }
    return error_matrices;
}

inline matrix3d mean_matrix3d(const std::vector<matrix3d>& ms)
{
    return fplus::fold_left_1(add_matrix3ds, ms) /
        static_cast<float_t>(ms.size());
}

inline float_t square_error_and_sum_div_2(const matrix3d& error)
{
    const auto squared_error = transform_matrix3d(fplus::square<float_t>, error);
    return matrix3d_sum_all_values(squared_error) / 2;
}

inline float_t test_params(
    const float_vec& params,
    layer_ptr& net,
    const input_with_output& data)
{
    float_vec old_params = net->get_params();
    net->set_params(params);
    auto net_result = net->forward_pass(data.input_);
    auto error = sub_matrix3d(net_result, data.output_);
    auto result = square_error_and_sum_div_2(error);
    net->set_params(old_params);
    return result;
}

inline float_t test_params_dataset(const float_vec& params,
    layer_ptr& net,
    const input_with_output_vec& dataset)
{
    float_vec all_square_error_and_sum_div_2_s =
        fplus::transform([&](const input_with_output& data)
        {
            return test_params(params, net, data);
        },
        dataset);
    return fplus::mean<float_t>(all_square_error_and_sum_div_2_s);
}

inline float_vec calc_net_gradient_numeric(
    layer_ptr& net,
    const input_with_output_vec& dataset)
{
    const float_t gradient_delta = 0.00001f;

    const auto calculate_gradient_dim =
        [&net, gradient_delta](
            const float_vec& curr_params,
            const input_with_output& data,
            std::size_t i) -> float_t
    {
        float_vec params_plus = curr_params;
        float_vec params_minus = curr_params;

        params_plus[i] += gradient_delta;
        params_minus[i] -= gradient_delta;

        float_t plus_error = test_params(params_plus, net, data);
        float_t minus_error = test_params(params_minus, net, data);

        return (plus_error - minus_error) / (2 * gradient_delta);
    };

    const auto calc_gradient =
        [&net, &calculate_gradient_dim]
        (const input_with_output& data) -> float_vec
    {
        float_vec old_params = net->get_params();
        float_vec gradient(old_params.size(), 0);
        for (std::size_t i = 0; i < old_params.size(); ++i)
        {
            float_t dim_gradient = calculate_gradient_dim(old_params, data, i);
            gradient[i] = dim_gradient;
        }
        return gradient;
    };

    std::vector<matrix3d> gradients;
    gradients.reserve(dataset.size());
    for (const auto& data : dataset)
    {
        float_vec gradient = calc_gradient(data);
        gradients.push_back(matrix3d(size3d(1, gradient.size(), 1), gradient));
    }
    return mean_matrix3d(gradients).as_vector();
}

inline float_vec calc_net_gradient_backprop(
    layer_ptr& net,
    const input_with_output_vec& dataset)
{
    std::vector<matrix3d> gradients;
    gradients.reserve(dataset.size());
    for (const auto& data : dataset)
    {
        auto result = net->forward_pass(data.input_);
        auto error = sub_matrix3d(result, data.output_);
        float_vec gradient;
        gradient.reserve(net->param_count());
        net->backward_pass(error, gradient);
        gradient = fplus::reverse(gradient);
        gradients.push_back(matrix3d(size3d(1, gradient.size(), 1), gradient));
    }
    return mean_matrix3d(gradients).as_vector();
}

inline std::pair<float_t, float_t> optimize_net_random(layer_ptr& net,
    const input_with_output_vec& dataset)
{
    float_vec old_params = net->get_params();
    float_vec new_params = randomly_change_params(old_params, 0.1f);

    float_t old_error = test_params_dataset(net->get_params(), net, dataset);
    float_t new_error = test_params_dataset(new_params, net, dataset);

    if (new_error > old_error)
    {
        net->set_params(old_params);
    }
    return {old_error, new_error};
}

inline std::pair<float_t, float_t> optimize_net_gradient(
    layer_ptr& net,
    const input_with_output_vec& dataset,
    float_t& speed_factor)
{
    //auto gradient = calc_net_gradient_numeric(net, dataset);
    auto gradient = calc_net_gradient_backprop(net, dataset);
    //std::cout << "gradient " << fplus::show_cont(gradient) << std::endl;

    float_vec old_params = net->get_params();
    float_vec new_params = old_params;
    for (std::size_t i = 0; i < old_params.size(); ++i)
    {
        float_t change = speed_factor * -gradient[i];
        new_params[i] += change;
    }

    float_t old_error = test_params_dataset(old_params, net, dataset);
    float_t new_error = test_params_dataset(new_params, net, dataset);

    if (new_error >= old_error)
    {
        speed_factor *= 0.99f;
    }
    net->set_params(new_params);

    return {old_error, new_error};
}

inline void train(layer_ptr& net,
    const input_with_output_vec& dataset,
    std::size_t max_iters,
    float_t mean_error_goal,
    float_t learning_rate,
    std::size_t batch_size = 0)
{
    if (batch_size == 0 || batch_size > dataset.size())
        batch_size = dataset.size();
    std::cout << "starting training, dataset.size " << dataset.size()
        << " batch_size " << batch_size << std::endl;
    auto show_progress = [](
        std::size_t iter,
        float_t new_learning_rate,
        float_t old_error,
        float_t new_error,
        const std::pair<float_t,float_t>& weights_mean_and_stddev)
    {
        std::cout << "iteration " << fplus::to_string_fill_left(' ', 10, iter)
        << ", batch old error " << fplus::fill_left(' ', 15, fplus::show_float<float_t>(1, 10)(old_error))
        << ", batch new error " << fplus::fill_left(' ', 15, fplus::show_float<float_t>(1, 10)(new_error))
        << ", error diff " << fplus::fill_left(' ', 15, fplus::show_float<float_t>(1, 10)(new_error - old_error))
        << ", new learning rate " << fplus::fill_left(' ', 15, fplus::show_float<float_t>(1, 10)(new_learning_rate))
        << ", weights_mean " << fplus::fill_left(' ', 15, fplus::show_float<float_t>(1, 10)(weights_mean_and_stddev.first))
        << ", weights_stddev " << fplus::fill_left(' ', 15, fplus::show_float<float_t>(1, 10)(weights_mean_and_stddev.second))
        << std::endl;
    };
    /*
    auto show_params = [](const layer_ptr& current_net)
    {
        auto show = fplus::show_float_fill_left<float_t>(' ', 8, 4);
        std::cout << "params " << fplus::show_cont(fplus::transform(show, current_net->get_params())) << std::endl;
    };
    */
    timer stopwatch;
    for (std::size_t iter = 0; iter < max_iters; ++iter)
    {
        const auto batch =
            batch_size == 0
                ? dataset
                : fplus::sample(batch_size, dataset);

        const auto old_and_new_error =
            optimize_net_gradient(net, batch, learning_rate);

        const float_t old_error = old_and_new_error.first;
        const float_t new_error = old_and_new_error.second;

        if (new_error < mean_error_goal ||
            iter == 0 ||
            stopwatch.elapsed() > 0.5)
        {
            show_progress(iter, learning_rate, old_error, new_error,
                fplus::mean_stddev<float_t>(net->get_params()));
            stopwatch.reset();
            if (new_error < mean_error_goal)
                return;
        }
        if (learning_rate < 0.0000001f)
        {
            return;
        }
    }
    //const float_t error = test_params_dataset(net->get_params(), net, dataset);
    //show_progress(max_iters, learning_rate, 0, error);
    //show_params(net);
}

inline void test(layer_ptr& net, const input_with_output_vec& dataset)
{
    std::cout << "running test" << std::endl;
    std::size_t correct_count = 0;
    for (const auto& data : dataset)
    {
        assert(data.output_.size() == net->output_size());
        const auto out_vol = net->forward_pass(data.input_);
        const auto classification_result = matrix3d_max_pos(out_vol);
        const auto wanted_result = matrix3d_max_pos(data.output_);
        //std::cout << "c " << classification_result.y_ << " - d " << wanted_result.y_ << std::endl;
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
