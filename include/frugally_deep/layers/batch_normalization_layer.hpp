// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.hpp"

namespace fdeep { namespace internal
{

// https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
// Since "batch size" is always 1 it simply scales and shifts the input tensor.
class batch_normalization_layer : public layer
{
public:
    explicit batch_normalization_layer(const std::string& name, float_t epsilon,
        const float_vec& beta, const float_vec& gamma)
        : layer(name),
        epsilon_(epsilon),
        beta_(beta),
        gamma_(gamma)
    {
    }
protected:
    float_t epsilon_;
    float_vec beta_;
    float_vec gamma_;
    tensor3s apply_impl(const tensor3s& inputs) const override
    {
        assertion(inputs.size() == 1, "invalid number of tensors");
        const auto& input = inputs.front();

        const auto transform_slice = [this]
            (const tensor2& slice, const std::size_t idx) -> tensor2
        {
            const auto mu = tensor2_mean_value(slice);
            const auto xmu = sub_from_tensor2_elems(slice, mu);
            const auto sq = transform_tensor2(fplus::square<float_t>, xmu);
            const auto var = tensor2_mean_value(sq);
            const auto sqrtvar = std::sqrt(var + epsilon_);
            auto ivar = static_cast<float_t>(1) / sqrtvar;
            if (!gamma_.empty())
                ivar *= gamma_[idx];
            auto out = multiply_tensor2_elems(xmu, ivar);
            if (!beta_.empty())
                out = add_to_tensor2_elems(out, beta_[idx]);
            return out;
        };

        const auto slices = tensor3_to_depth_slices(input);
        if (!gamma_.empty())
        {
            assertion(slices.size() == gamma_.size(), "invalid gamma");
        }
        if (!beta_.empty())
        {
            assertion(slices.size() == beta_.size(), "invalid beta");
        }

        std::vector<tensor2> result;
        result.reserve(slices.size());
        for (std::size_t idx = 0; idx < slices.size(); ++idx)
        {
            result.push_back(transform_slice(slices[idx], idx));
        }
        return {tensor3_from_depth_slices(result)};
    }
};

} } // namespace fdeep, namespace internal
