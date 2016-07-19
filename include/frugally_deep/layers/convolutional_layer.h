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
        : size_in_(size_in), filters_(generate_filters(size_in.depth_, filter_size, k))
    {
        assert(stride == 1); // todo: allow different strides
    }
    matrix3d forward_pass(const matrix3d& input) const override
    {
        return convolve(filters_, input);
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
    void set_params(const float_vec& params) override
    {
        assert(params.size() == param_count());
        auto params_per_filter =
            fplus::split_every(filters_.front().param_count(), params);
        for (std::size_t i = 0; i < filters_.size(); ++i)
        {
            filters_[i].set_params(params_per_filter[i]);
        }
    }
    const size3d& input_size() const override
    {
        return size_in_;
    }
    size3d output_size() const override
    {
        return size3d(filters_.size(), size_in_.height_, size_in_.width_);
    }
private:
    size3d size_in_;
    std::vector<filter> filters_;
};

} // namespace fd
