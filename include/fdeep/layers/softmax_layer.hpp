// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

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
    tensor3 transform_input(const tensor3& input) const override
    {
        // Softmax function is applied along channel dimension.

        const auto ex = [this](float_type x) -> float_type
        {
            return std::exp(x);
        };

        // Get unnormalised values of exponent function.
        auto output = transform_tensor3(ex, input);;

        for (size_t iRow = 0; iRow < input.shape().height_; ++iRow)
        {
            for (size_t jColumn = 0; jColumn < input.shape().width_; ++jColumn)
            {
                // Get the sum of unnormalised values for one pixel.
                // We are not using Kahan summation algorithm due to usually
                // small number of object classes.
                float_type sum = 0.0f;
                for (size_t kClass = 0; kClass < input.shape().depth_; ++kClass)
                {
                    sum += output.get(kClass, iRow, jColumn);
                }
                // Divide the unnormalised values for one pixel by sum.
                for (size_t kClass = 0; kClass < input.shape().depth_; ++kClass)
                {
                    output.set(kClass, iRow, jColumn, output.get(kClass, iRow, jColumn)/sum);
                }

            }
        }

        return output;

    }
};

} } // namespace fdeep, namespace internal