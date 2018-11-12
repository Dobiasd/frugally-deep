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
    tensor5 transform_input(const tensor5& input) const override
    {
        // Get unnormalized values of exponent function.
        const auto ex = [this](float_type x) -> float_type
        {
            return std::exp(x);
        };
        const float_type m = input.get(tensor5_max_pos(input));
        const auto inp_shifted = subtract_tensor5(input, tensor5(input.shape(), m));
        auto output = transform_tensor5(ex, inp_shifted);

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
                    sum_shifted += output.get(0, 0, y, x, z_class);
                }
                // Divide the unnormalized values of each pixel by the stacks sum.
                const auto log_sum_shifted = std::log(sum_shifted);
                for (size_t z_class = 0; z_class < input.shape().depth_; ++z_class)
                {
                    output.set(0, 0, y, x, z_class,
                        std::exp(inp_shifted.get(0, 0, y, x, z_class) - log_sum_shifted));
                }
            }
        }
        return output;
    }
};

} } // namespace fdeep, namespace internal
