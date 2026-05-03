// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace fdeep {
namespace internal {

    class group_normalization_layer : public layer {
    public:
        explicit group_normalization_layer(const std::string& name,
            std::size_t groups,
            int axis,
            float_type epsilon,
            const float_vec& beta,
            const float_vec& gamma)
            : layer(name)
            , groups_(groups)
            , axis_(axis)
            , epsilon_(epsilon)
            , beta_(beta)
            , gamma_(gamma)
        {
        }

    protected:
        const std::size_t groups_;
        const int axis_;
        const float_type epsilon_;
        const float_vec beta_;
        const float_vec gamma_;

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);
            const auto in_shape = input.shape();

            const std::size_t absolute_axis = rank_aligned_axis_to_absolute_axis(in_shape.rank(), axis_);
            assertion(absolute_axis == 5,
                "GroupNormalization is currently only supported on the last (channel) axis.");

            const std::size_t channels = in_shape.depth_;
            assertion(groups_ > 0, "GroupNormalization groups must be > 0.");
            assertion(channels % groups_ == 0,
                "Number of channels must be divisible by number of groups.");
            const std::size_t channels_per_group = channels / groups_;

            const std::size_t spatial_volume = in_shape.size_dim_5_
                * in_shape.size_dim_4_ * in_shape.height_ * in_shape.width_;
            const std::size_t group_volume = spatial_volume * channels_per_group;
            assertion(group_volume > 0, "GroupNormalization input has zero volume.");

            const auto& src = *input.as_vector();
            float_vec means(groups_, 0);
            float_vec vars(groups_, 0);

            // mean per group
            for (std::size_t i = 0; i < src.size(); ++i) {
                const std::size_t c = i % channels;
                const std::size_t g = c / channels_per_group;
                means[g] += src[i];
            }
            for (std::size_t g = 0; g < groups_; ++g)
                means[g] /= static_cast<float_type>(group_volume);

            // variance per group
            for (std::size_t i = 0; i < src.size(); ++i) {
                const std::size_t c = i % channels;
                const std::size_t g = c / channels_per_group;
                const float_type d = src[i] - means[g];
                vars[g] += d * d;
            }
            for (std::size_t g = 0; g < groups_; ++g)
                vars[g] /= static_cast<float_type>(group_volume);

            float_vec inv_std(groups_, 0);
            for (std::size_t g = 0; g < groups_; ++g)
                inv_std[g] = static_cast<float_type>(1) / std::sqrt(vars[g] + epsilon_);

            const bool has_gamma = !gamma_.empty();
            const bool has_beta = !beta_.empty();

            float_vec out(src.size());
            for (std::size_t i = 0; i < src.size(); ++i) {
                const std::size_t c = i % channels;
                const std::size_t g = c / channels_per_group;
                float_type v = (src[i] - means[g]) * inv_std[g];
                if (has_gamma)
                    v *= gamma_[c];
                if (has_beta)
                    v += beta_[c];
                out[i] = v;
            }

            return { tensor(in_shape, std::move(out)) };
        }
    };

}
}
