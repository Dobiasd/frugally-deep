// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

namespace fd
{

// Since "batch size" is always 1 it simply scales and shifts the input tensor.
class batch_normalization_layer : public layer
{
public:
    explicit batch_normalization_layer(const size3d& size_in, float_t epsilon)
        : layer(size_in, size_in),
        epsilon_(epsilon),
        beta_(size_in.depth_),
        gamma_(size_in.depth_),
        last_input_(size_in)
    {
    }
    std::size_t param_count() const override
    {
        return beta_.size() + gamma_.size();
    }
    float_vec get_params() const override
    {
        return fplus::append(beta_, gamma_);
    }
    void set_params(const float_vec_const_it ps_begin,
        const float_vec_const_it ps_end) override
    {
        assert(static_cast<std::size_t>(std::distance(ps_begin, ps_end)) ==
            param_count());
        auto it_beta_begin = ps_begin;
        auto it_beta_end = it_beta_begin;
        std::advance(it_beta_end, beta_.size());

        auto it_gamma_begin = it_beta_end;
        auto it_gamma_end = it_gamma_begin;
        std::advance(it_gamma_end, gamma_.size());

        std::copy(it_beta_begin, it_beta_end, beta_.begin());
        std::copy(it_gamma_begin, it_gamma_end, gamma_.begin());
    }
    void random_init_params() override
    {
        const auto params =
            generate_normal_distribution_values(0, 1, param_count());
        set_params(std::begin(params), std::end(params));
    }
protected:
    float_t epsilon_;
    float_vec beta_;
    float_vec gamma_;
    mutable matrix3d last_input_;
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        last_input_ = input;
        // todo: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
        auto slices = matrix3d_to_depth_slices(input);
        slices = fplus::zip_with(multiply_matrix2d_elems, slices, gamma_); // todo + epsilon
        slices = fplus::zip_with(add_to_matrix2d_elems, slices, beta_);
        return matrix3d_from_depth_slices(slices);
    }

    matrix3d backward_pass_impl(const matrix3d& e,
        float_vec& params_deltas_acc_reversed) const override
    {
        const float_vec d_beta = fplus::transform(matrix2d_sum_all_values,
            matrix3d_to_depth_slices(e));
        const float_vec d_gamma = fplus::transform(matrix2d_sum_all_values,
            matrix3d_to_depth_slices(
                multiply_matrix3ds_elementwise(last_input_, e)));
        const float_vec param_deltas_vec = fplus::append(d_beta, d_gamma);
        params_deltas_acc_reversed.insert(std::end(params_deltas_acc_reversed),
            param_deltas_vec.rbegin(), param_deltas_vec.rend());

        auto slices = matrix3d_to_depth_slices(e);
        slices = fplus::zip_with(multiply_matrix2d_elems, slices, gamma_); // todo + epsilon
        return matrix3d_from_depth_slices(slices);
    }
};

} // namespace fd
