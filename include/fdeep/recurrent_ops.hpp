// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <functional>
#include <string>

namespace fdeep {
namespace internal {

    using Eigen::Dynamic;

    template <int Count>
    using RowVector = Eigen::Matrix<float_type, 1, Count>;

    inline float_type tanh_activation(float_type x)
    {
        return std::tanh(x);
    }

    inline float_type sigmoid_activation(float_type x)
    {
        return 1 / (1 + std::exp(-x));
    }

    inline float_type swish_activation(float_type x)
    {
        return x / (1 + std::exp(-x));
    }

    inline float_type hard_sigmoid_activation(float_type x)
    {
        // https://github.com/keras-team/keras/blob/f7bc67e6c105c116a2ba7f5412137acf78174b1a/keras/ops/nn.py#L316C6-L316C74
        if (x < -3) {
            return 0;
        }
        if (x > 3) {
            return 1;
        }
        return (x / static_cast<float_type>(6)) + static_cast<float_type>(0.5);
    }

    inline float_type selu_activation(float_type x)
    {
        const float_type alpha = static_cast<float_type>(1.6732632423543772848170429916717);
        const float_type scale = static_cast<float_type>(1.0507009873554804934193349852946);
        return scale * (x >= 0 ? x : alpha * (std::exp(x) - 1));
    }

    inline float_type exponential_activation(float_type x)
    {
        return static_cast<float_type>(std::exp(x));
    }

    inline float_type gelu_activation(float_type x)
    {
        return static_cast<float_type>(0.5) * x * (static_cast<float_type>(1) + static_cast<float_type>(std::erf(x / std::sqrt(static_cast<float_type>(2)))));
    }

    inline float_type softsign_activation(float_type x)
    {
        return x / (std::abs(x) + static_cast<float_type>(1));
    }

    inline float_type elu_activation(float_type x)
    {
        return x >= 0 ? x : std::exp(x) - 1;
    }

}
}
