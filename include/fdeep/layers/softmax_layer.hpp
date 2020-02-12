// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/activation_layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

class softmax_layer : public activation_layer
{
public:
    explicit softmax_layer(const std::string& name)
        : activation_layer(name)
    {
    }
protected:
    tensor transform_input(const tensor& input) const override
    {
        // Get unnormalized values of exponent function.
        const auto ex = [](float_type x) -> float_type
        {
            return std::exp(x);
        };
        const float_type m = input.get(tensor_max_pos(input));
        const auto inp_shifted = subtract_tensor(input, tensor(input.shape(), m));
        auto output = transform_tensor(ex, inp_shifted);

        // Softmax function is applied along channel dimension.
        for (size_t y = 0; y < input.shape().height_; ++y)
        {
            for (size_t x = 0; x < input.shape().width_; ++x)
            {
                // Get the sum of unnormalized values for one pixel.
                // We are not using Kahan summation, since the number
                // of object classes is usually quite small.
                float_type sum_shifted = 0.0f;
                for (size_t z_class = 0; z_class < input.shape().depth_; ++z_class)
                {
                    sum_shifted += output.get_ignore_rank(tensor_pos(y, x, z_class));
                }
                // Divide the unnormalized values of each pixel by the stacks sum.
                const auto log_sum_shifted = std::log(sum_shifted);
                for (size_t z_class = 0; z_class < input.shape().depth_; ++z_class)
                {
                    const auto result = std::exp(inp_shifted.get_ignore_rank(tensor_pos(y, x, z_class)) - log_sum_shifted);
                    output.set_ignore_rank(tensor_pos(y, x, z_class), std::isinf(result) ? static_cast<float_type>(0) : result);
                }
            }
        }
        return output;
    }
};

} } // namespace fdeep, namespace internal
