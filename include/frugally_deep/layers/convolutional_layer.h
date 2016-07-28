// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/convolution.h"
#include "frugally_deep/filter.h"
#include "frugally_deep/size2d.h"
#include "frugally_deep/size3d.h"
#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.h>

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
        : layer(size_in, size3d(k, size_in.height_, size_in.width_)),
        filters_(generate_filters(size_in.depth_, filter_size, k))
    {
        assert(stride == 1); // todo: allow different strides
        assert(k != 0);
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
                float_t stddev = 1 / std::sqrt(filt.size().volume());
                auto filt_params = generate_normal_distribution_values(
                    mean, stddev, filt.param_count());
                filt_params.back() = 0; // bias;
                return filt_params;
            }, filters_);
        set_params(std::begin(params), std::end(params));
    }
protected:
    static matrix3d pad_matrix3d(
        std::size_t offset_y,
        std::size_t offset_x,
        const matrix3d& in_vol)
    {
        matrix3d out_vol(size3d(
            in_vol.size().depth_,
            in_vol.size().height_ + 2 * offset_y,
            in_vol.size().width_ + 2 * offset_x));
        for (std::size_t z = 0; z < in_vol.size().depth_; ++z)
        {
            for (std::size_t y = 0; y < in_vol.size().height_; ++y)
            {
                for (std::size_t x = 0; x < in_vol.size().width_; ++x)
                {
                    out_vol.set(z, y + offset_y, x + offset_x,
                        in_vol.get(z, y, x));
                }
            }
        }
        return out_vol;
    }
    static matrix3d pad_matrix3d_for_filters(
        const filter_vec& filters,
        const matrix3d& in_vol)
    {
        assert(!filters.empty());
        const size3d& filt_size = filters[0].size();
        assert((filt_size.height_ + 2) % 2 == 1);
        assert((filt_size.width_ + 2) % 2 == 1);
        std::size_t offset_y = (filt_size.height_ - 1) / 2;
        std::size_t offset_x = (filt_size.width_ - 1) / 2;
        return pad_matrix3d(offset_y, offset_x, in_vol);
    }
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        return convolve(filters_, pad_matrix3d_for_filters(filters_, input));
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
        const auto output = convolve(
            flipped_filters,
            pad_matrix3d_for_filters(flipped_filters, input));

        const auto input_slices = matrix3d_to_depth_slices(input);
        const auto last_input_padded = pad_matrix3d_for_filters(filters_, last_input_);

        const std::vector<matrix3d> filter_deltas =
            fplus::transform([&](const matrix2d& input_slice) -> matrix3d
                {
                    return convolve(input_slice, last_input_padded);
                }, input_slices);

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
};

} // namespace fd
