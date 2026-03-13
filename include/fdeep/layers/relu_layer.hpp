// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace fdeep {
namespace internal {

    class relu_layer : public activation_layer {
    public:
        explicit relu_layer(const std::string& name,
            const float_type max_value,
            const float_type negative_slope,
            const float_type threshold)
            : activation_layer(name)
            , max_value_(max_value)
            , negative_slope_(negative_slope)
            , threshold_(threshold)
        {
        }

    protected:
        tensor transform_input(const tensor& in_vol) const override
        {
            // Fast path for the common standard ReLU (max(0, x)):
            // std::transform with a simple lambda allows GCC/Clang to emit
            // SIMD vmaxps, unlike fplus::transform_convert which uses push_back.
            if (negative_slope_ == static_cast<float_type>(0) &&
                threshold_ == static_cast<float_type>(0) &&
                max_value_ == std::numeric_limits<float_type>::max()) {
                const auto& src = *in_vol.as_vector();
                float_vec result(src.size());
                std::transform(src.begin(), src.end(), result.begin(),
                    [](float_type x) -> float_type {
                        return std::max(static_cast<float_type>(0), x);
                    });
                return tensor(in_vol.shape(), std::move(result));
            }
            auto activation_function = [&](float_type x) -> float_type {
                if (x >= max_value_)
                    return max_value_;
                if (threshold_ <= x && x < max_value_)
                    return x;
                return negative_slope_ * (x - threshold_);
            };
            return transform_tensor(activation_function, in_vol);
        }
        float_type max_value_;
        float_type negative_slope_;
        float_type threshold_;
    };

}
}
