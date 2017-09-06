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
    fully_connected_layer(const std::string& name,
            std::size_t n_in, std::size_t n_out) :
        layer(name),
        params_(size2d(n_out, n_in + 1))
    {
    }
    matrix3ds apply(const matrix3ds& inputs) const override
    {
        assert(inputs.size() == 1);
        const auto& input = inputs[0];
        const auto input_slice_with_bias_neuron = bias_pad_input(input);
        //return matrix3d(size_out_, matrix2d_to_matrix3d(
            //multiply(params_, input_slice_with_bias_neuron)).as_vector());
        return {input};
    }
protected:
    static matrix2d bias_pad_input(const matrix3d& input)
    {
        return matrix2d(
            size2d(input.size().depth_ + 1, 1),
            fplus::append(input.as_vector(), {1}));
    }
    matrix2d params_;
};

} // namespace fd
