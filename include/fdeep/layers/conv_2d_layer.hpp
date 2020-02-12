// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/convolution.hpp"
#include "fdeep/filter.hpp"
#include "fdeep/shape2.hpp"
#include "fdeep/shape5.hpp"
#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class conv_2d_layer : public layer
{
public:
    explicit conv_2d_layer(
            const std::string& name, const shape5& filter_shape,
            std::size_t k, const shape2& strides, padding p,
            const shape2& dilation_rate,
            const float_vec& weights, const float_vec& bias,
            std::size_t output_dimensions)
        : layer(name),
        filters_(generate_im2col_filter_matrix(
            generate_filters(dilation_rate, filter_shape, k, weights, bias))),
        strides_(strides),
        padding_(p),
        output_dimensions_(output_dimensions)
    {
        assertion(k > 0, "needs at least one filter");
        assertion(filter_shape.volume() > 0, "filter must have volume");
        assertion(strides.area() > 0, "invalid strides");
    }
protected:
    tensor5s apply_impl(const tensor5s& inputs) const override
    {
        assertion(inputs.size() == 1, "only one input tensor allowed");
        const auto result = convolve(strides_, padding_, filters_, inputs.front());
        if (output_dimensions_ == 1)
        {
            // To support correct output rank for 1d version of layer.
            assertion(result.shape().rank_ == 3, "Invalid rank of conv output");
            return {tensor5_with_changed_rank(result, 2)};
        }
        return {result};
    }
    im2col_filter_matrix filters_;
    shape2 strides_;
    padding padding_;
    std::size_t output_dimensions_;
};

} } // namespace fdeep, namespace internal
