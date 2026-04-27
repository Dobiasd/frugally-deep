// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/convolution3d.hpp"
#include "fdeep/filter.hpp"
#include "fdeep/layers/layer.hpp"
#include "fdeep/shape3.hpp"
#include "fdeep/tensor_shape.hpp"

#include <fplus/fplus.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace fdeep {
namespace internal {

    class conv_3d_transpose_layer : public layer {
    public:
        explicit conv_3d_transpose_layer(
            const std::string& name, const tensor_shape& filter_shape,
            std::size_t k, const shape3& strides, padding p,
            const shape3& dilation_rate,
            const float_vec& weights, const float_vec& bias)
            : layer(name)
            , filters_(generate_im2col_filter_matrix_3d(
                  generate_filters_3d(dilation_rate, filter_shape, k, weights, bias, true)))
            , dilation_rate_(dilation_rate)
            , strides_(strides)
            , padding_(p)
        {
            assertion(k > 0, "needs at least one filter");
            assertion(filter_shape.volume() > 0, "filter must have volume");
            assertion(strides.volume() > 0, "invalid strides");
        }

    protected:
        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);
            return { convolve_transposed_3d(strides_, padding_, filters_, input) };
        }
        convolution3d_filter_matrices filters_;
        shape3 dilation_rate_;
        shape3 strides_;
        padding padding_;
    };

}
}
