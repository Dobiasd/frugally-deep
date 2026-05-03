// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>
#include <vector>

namespace fdeep {
namespace internal {

    class rms_normalization_layer : public layer {
    public:
        explicit rms_normalization_layer(const std::string& name,
            std::vector<int> axes,
            const float_vec& scale,
            float_type epsilon)
            : layer(name)
            , axes_(axes)
            , scale_(fplus::make_shared_ref<float_vec>(scale))
            , epsilon_(epsilon)
        {
        }

    protected:
        const std::vector<int> axes_;
        const shared_float_vec scale_;
        const float_type epsilon_;

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);

            // mean(x^2) over the specified axes
            const tensor squared = mult_tensors(input, input);
            const tensor summed = reduce(add_tensors, squared, axes_);
            const auto factor = static_cast<float_type>(squared.shape().volume()) / static_cast<float_type>(summed.shape().volume());
            const tensor mean_sq = transform_tensor(fplus::divide_by(factor), summed);

            // 1 / sqrt(mean_sq + eps)
            const float_type eps = epsilon_;
            const tensor inv_rms = transform_tensor(
                [eps](float_type v) -> float_type {
                    return static_cast<float_type>(1) / std::sqrt(v + eps);
                },
                mean_sq);
            const tensor normalized = mult_tensors(input, broadcast(inv_rms, input.shape()));

            // multiply by learnable scale, broadcast to match input shape
            std::vector<std::size_t> dims(5, 1);
            tensor_shape input_shape = input.shape();
            input_shape.maximize_rank();
            const auto input_shape_dimensions = input_shape.dimensions();
            for (const auto axis : axes_) {
                const std::size_t pos = rank_aligned_axis_to_absolute_axis(input.shape().rank(), axis) - 1;
                dims[pos] = input_shape_dimensions[pos];
            }
            const tensor_shape params_shape = create_tensor_shape_from_dims(dims);
            const tensor scale_t = scale_->empty()
                ? tensor(input.shape(), 1)
                : broadcast(tensor(params_shape, scale_), input.shape());

            return { mult_tensors(normalized, scale_t) };
        }
    };

}
}
