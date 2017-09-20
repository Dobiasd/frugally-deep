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
    enum class padding { valid, same };
    static std::vector<filter> generate_filters(
        const size3d& filter_size, std::size_t k)
    {
        return std::vector<filter>(k, filter(matrix3d(
            size3d(filter_size)), 0));
    }
    explicit convolutional_layer(
            const std::string& name, const size3d& filter_size,
            std::size_t k, const size2d& strides, padding p,
            const float_vec& weights, const float_vec& bias)
        : layer(name),
        filters_(generate_filters(filter_size, k))
        //padding_y_((size_out_.height_ * stride - size_in.height_ + filter_size.height_ - stride) / 2),
        //padding_x_((size_out_.width_ * stride - size_in.width_ + filter_size.width_ - stride) / 2),
        //stride_(stride)
    {
        assert(k != 0);
        assert(filter_size.width_ > 0); // todo remove
        assert(strides.width_ > 0); // todo remove
        assert(p == p); // todo remove
        fill_filters(weights, bias);
        //assert((size_out_.height_ * stride - size_in.height_ + filter_size.height_ - stride) % 2 == 0);
        //assert((size_out_.width_ * stride - size_in.width_ + filter_size.width_ - stride) % 2 == 0);
    }
protected:
    matrix3ds apply_impl(const matrix3ds& inputs) const override
    {
        assert(inputs.size() == 1);
        const auto& input = inputs[0];
        return {convolve(stride_, padding_x_, padding_y_, filters_, input)};
    }
    void fill_filters(const float_vec& weights, const float_vec& bias)
    {
        assert(!filters_.empty());
        const std::size_t param_count = fplus::sum(fplus::transform(
                [](const filter& f) -> std::size_t { return f.size().volume(); },
            filters_));

        assert(static_cast<std::size_t>(weights.size()) == param_count);
        const auto filter_param_cnt = filters_.front().size().volume();

        const auto filter_weights =
            fplus::split_every(filter_param_cnt, weights);
        assert(filter_weights.size() == filters_.size());
        assert(bias.size() == filters_.size());
        auto it_filter_val = std::begin(filter_weights);
        auto it_filter_bias = std::begin(bias);
        for (auto& filt : filters_)
        {
            filt.set_params(*it_filter_val, *it_filter_bias);
            ++it_filter_val;
            ++it_filter_bias;
        }
    }
    filter_vec filters_;
    std::size_t padding_y_;
    std::size_t padding_x_;
    std::size_t stride_;
};

} // namespace fd
