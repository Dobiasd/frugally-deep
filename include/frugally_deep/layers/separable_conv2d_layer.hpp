// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/convolution.hpp"
#include "frugally_deep/convolution_transpose.hpp"
#include "frugally_deep/filter.hpp"
#include "frugally_deep/shape2.hpp"
#include "frugally_deep/shape3.hpp"
#include "frugally_deep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cstddef>
#include <vector>

namespace fdeep { namespace internal
{

// todo: variable padding, variable strides
// Convolve depth slices separately first.
// Then convolve normally with kernel_size = (1, 1)
class separable_conv2d_layer : public layer
{
public:
    explicit separable_conv2d_layer(
            const std::string& name, std::size_t input_depth,
            const shape3& filter_size,
            std::size_t k, const shape2& strides, padding p,
            const float_vec& slice_weights, const float_vec& stack_weights,
            const float_vec& bias_0,
            const float_vec& bias)
        : layer(name),
        filters_slice_(generate_filters(
            filter_size, input_depth, slice_weights, bias_0)),
        filters_stack_(generate_filters(
            shape3(input_depth, 1, 1), k, stack_weights, bias)),
        padding_(p),
        strides_(strides)
    {
        assertion(k > 0, "needs at least one filter");
        assertion(filter_size.volume() > 0, "filter must have volume");
        assertion(strides.area() > 0, "invalid strides");
    }
protected:
    tensor3s apply_impl(const tensor3s& inputs) const override
    {
        assertion(inputs.size() == 1, "only one input tensor allowed");

        const auto input_slices = tensor3_to_depth_slices(inputs.front());

        assertion(input_slices.size() == filters_slice_.size(),
            "invalid input depth");

        const auto convolve_slice =
            [&](const tensor2& slice, const filter& f) -> tensor2
        {
            assertion(f.shape().depth_ == 1, "invalid filter depth");
            const auto result = convolve(strides_, padding_,
                filter_vec(1, f), tensor2_to_tensor3(slice));
            assertion(result.shape().depth_ == 1, "invalid conv output");
            return tensor3_to_depth_slices(result).front();
        };

        const auto temp = tensor3_from_depth_slices(fplus::zip_with(
            convolve_slice, input_slices, filters_slice_));

        return {convolve(shape2(1, 1), padding::valid, filters_stack_, temp)};
    }
    filter_vec filters_slice_;
    filter_vec filters_stack_;
    padding padding_;
    shape2 strides_;
};

} } // namespace fdeep, namespace internal
