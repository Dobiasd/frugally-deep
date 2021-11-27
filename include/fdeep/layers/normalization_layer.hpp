// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class normalization_layer : public layer
{
public:
    explicit normalization_layer(
            const std::string& name,
            const std::vector<int>& axes,
            const float_vec& mean, const float_vec& variance)
        : layer(name),
        axes_(axes),
        mean_(mean),
        variance_(variance)
    {
        assertion(axes.size() <= 2, "Unsupported number of axes for Normalization layer");
    }
protected:
    tensors apply_impl(const tensors& inputs) const override final
    {
        const auto& input = single_tensor_from_tensors(inputs);

        const int rank = static_cast<int>(input.shape().rank());

        const auto axes = fplus::transform([&](int a) {
            return a < 0 ? a + rank : rank - a;
            }, axes_);

        const auto transform_slice = [&](const std::size_t idx, const tensor& slice) -> tensor
        {
            const auto sqrt_of_variance = std::sqrt(variance_[idx]);
            return transform_tensor([&](float_type x){ return (x - mean_[idx]) / sqrt_of_variance; }, slice);
        };

        const auto transform_slice_with_idx = [&](const tensors& slices) -> tensors
        {
            return fplus::transform_with_idx(transform_slice, slices);
        };

        if (axes_.empty())
            return {transform_slice(0, input)};
        else if (axes[0] == 0)
            return {concatenate_tensors_depth(transform_slice_with_idx(tensor_to_depth_slices(input)))};
        else if (axes[0] == 1)
            return {concatenate_tensors_width(transform_slice_with_idx(tensor_to_tensors_width_slices(input)))};
        else if (axes[0] == 2)
            return {concatenate_tensors_height(transform_slice_with_idx(tensor_to_tensors_height_slices(input)))};
        else if (axes[0] == 3)
            return {concatenate_tensors_dim4(transform_slice_with_idx(tensor_to_tensors_dim4_slices(input)))};
        else if (axes[0] == 4)
            return {concatenate_tensors_dim5(transform_slice_with_idx(tensor_to_tensors_dim5_slices(input)))};
        else
            raise_error("Invalid axis (" + std::to_string(axes[0]) + ") for Normalization layer");
        return {};
    }
    const std::vector<int> axes_;
    float_vec mean_;
    float_vec variance_;
};

} } // namespace fdeep, namespace internal
