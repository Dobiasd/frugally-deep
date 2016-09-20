// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/convolution.h"
#include "frugally_deep/convolution_transpose.h"
#include "frugally_deep/filter.h"
#include "frugally_deep/size2d.h"
#include "frugally_deep/size3d.h"
#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

// todo: variable padding, variable strides
class convolutional_layer : public layer
{
public:
    static std::vector<filter> generate_filters(
        std::size_t depth, const size2d& filter_size, std::size_t k)
    {
        return std::vector<filter>(k, filter(matrix3d(
            size3d(depth, filter_size.height_, filter_size.width_)), 0));
    }
    explicit convolutional_layer(
            const size3d& size_in, const size2d& filter_size,
            std::size_t k, std::size_t stride)
        : layer(size_in,
            size3d(k, size_in.height_ / stride, size_in.width_ / stride)),
        filters_(generate_filters(size_in.depth_, filter_size, k)),
        padding_y_((size_out_.height_ * stride - size_in.height_ + filter_size.height_ - stride) / 2),
        padding_x_((size_out_.width_ * stride - size_in.width_ + filter_size.width_ - stride) / 2),
        stride_(stride)
    {
        assert(k != 0);
        assert((size_out_.height_ * stride - size_in.height_ + filter_size.height_ - stride) % 2 == 0);
        assert((size_out_.width_ * stride - size_in.width_ + filter_size.width_ - stride) % 2 == 0);
    }
    std::size_t param_count() const override
    {
        auto counts = fplus::transform(
            [](const filter& f) { return f.param_count(); },
            filters_);
        return fplus::sum(counts);
    }
    float_vec get_params() const override
    {
        return fplus::concat(
            fplus::transform(
                [](const filter& f) { return f.get_params(); },
                filters_));
    }
    void set_params(const float_vec_const_it ps_begin,
        const float_vec_const_it ps_end) override
    {
        assert(static_cast<std::size_t>(std::distance(ps_begin, ps_end)) == param_count());
        const auto filter_param_count =
            static_cast<float_vec::difference_type>(
                filters_.front().param_count());
        auto it = ps_begin;
        for (auto& filt : filters_)
        {
            filt.set_params(it, it + filter_param_count);
            it += filter_param_count;
        }
    }
    void random_init_params() override
    {
        const auto params = fplus::transform_and_concat(
            [](const filter& filt)
            {
                float_t mean = 0;
                float_t stddev = 1 / static_cast<float_t>(
                    std::sqrt(filt.size().volume()));
                auto filt_params = generate_normal_distribution_values(
                    mean, stddev, filt.param_count());
                filt_params.back() = 0; // bias;
                return filt_params;
            }, filters_);
        set_params(std::begin(params), std::end(params));
    }
protected:
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        return convolve(stride_, padding_x_, padding_y_, filters_, input);
    }
    matrix3d backward_pass_impl(const matrix3d& input,
        float_vec& params_deltas_acc_reversed) const override
    {
        // forward pass: x `conv` w = y
        // backward pass: e_x = flip(w) `conv` f_y
        // gradient computation: x `conv` e_y = delta_j / delta_w
        const auto remove_filter_bias = [](const filter& f)
        {
            return filter(f.get_matrix3d(), 0);
        };
        const auto flipped_filters =
            fplus::transform(
                remove_filter_bias,
                flip_filters_spatially(filters_));

        const auto output = convolve_transpose(
            stride_, padding_y_, padding_x_, flipped_filters, input);

        const auto input_slices = matrix3d_to_depth_slices(input);

        // todo: sparse schon hier aussen machen
        const auto last_input_inverted = invert_x_y_positions(last_input_);
        const std::vector<matrix3d> filter_deltas =
            fplus::transform([&](const matrix2d& input_slice) -> matrix3d
                {
                    // todo: assertions
                    const auto sparse_input_slice = sparse_matrix2d(
                        stride_, input_slice);
                    return convolve(1, padding_y_, padding_x_,
                        sparse_input_slice, last_input_ );
                }, input_slices);

        assert(filter_deltas.front().size() == filters_.front().size());

        const std::vector<float_t> bias_deltas =
            fplus::transform([&](const matrix2d& input_slice) -> float_t
                {
                    return matrix2d_sum_all_values(input_slice);
                }, input_slices);

        assert(bias_deltas.size() == filter_deltas.size());
        const filter_vec delta_filters =
            fplus::zip_with(
                [](const matrix3d& f_m, float_t bias_delta) -> filter
                {
                    return filter(f_m, bias_delta);
                },
                filter_deltas, bias_deltas);

        const float_vec param_deltas_vec =
            fplus::concat(
                fplus::transform([](const filter& f) -> float_vec
                {
                    return f.get_params();
                }, delta_filters));

        // see slide 10 of http://de.slideshare.net/kuwajima/cnnbp

        params_deltas_acc_reversed.insert(std::end(params_deltas_acc_reversed),
            param_deltas_vec.rbegin(), param_deltas_vec.rend());
        return output;
    }
    filter_vec filters_;
    std::size_t padding_y_;
    std::size_t padding_x_;
    std::size_t stride_;
};

} // namespace fd
