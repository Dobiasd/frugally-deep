// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

#include "frugally_deep/matrix2d.h"

#include <fplus/fplus.hpp>

namespace fd
{

// Takes a single stack volume (size3d(n, 1, 1)) as input.
class fully_connected_layer : public layer
{
public:
    static matrix2d generate_params(const float_vec& weights,
        const float_vec& bias)
    {
        const std::size_t n_in = weights.size() / bias.size();

        return matrix2d(size2d(n_in + 1, bias.size()),
            fplus::append(weights, bias));
            /*
        matrix2d result(size2d(bias.size(), n_in + 1));
        std::size_t weight_idx = 0;
        for (std::size_t y = 0; y < result.size().height_; ++y)
        {
            for (std::size_t x = 0; x < n_in; ++x)
            {
                result.set(y, x, weights[weight_idx++]);
            }
            result.set(y, n_in, bias[y]);
        }
        return result;
        */
    }
    fully_connected_layer(const std::string& name, std::size_t units,
            const float_vec& weights,
            const float_vec& bias) :
        layer(name),
        units_(units),
        params_(generate_params(weights, bias))
    {
        assertion(bias.size() == units, "invalid bias count");
        assertion(weights.size() % units == 0, "invalid weight count");
    }
protected:
    matrix3ds apply_impl(const matrix3ds& inputs) const override
    {
        assertion(inputs.size() == 1, "invalid number of input tensors");
        const auto& input = inputs[0];
        assertion(input.size().height_ == 1, "input needs to be flattened");
        assertion(input.size().depth_ == 1, "input needs to be flattened");
        const auto bias_padded_input = bias_pad_input(input);
        return {matrix3d(size3d(1, 1, units_),
            multiply(bias_padded_input, params_).as_vector())};
    }
    static matrix2d bias_pad_input(const matrix3d& input)
    {
        return matrix2d(
            size2d(1, input.size().width_ + 1),
            fplus::append(input.as_vector(), {1}));
    }
    std::size_t units_;
    matrix2d params_;
};

} // namespace fd
