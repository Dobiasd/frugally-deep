// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class upsampling_1d_layer : public layer
{
public:
    explicit upsampling_1d_layer(const std::string& name,
        const std::size_t size) :
    layer(name),
    size_(size)
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override final
    {
        const auto& input = single_tensor_from_tensors(inputs);
        assertion(input.shape().rank() == 2, "invalid input shape for Upsampling1D");
        return {upsampling_1d_rank_2(input)};
    }

    tensor upsampling_1d_rank_2(const tensor& input) const
    {
        assertion(input.shape().rank() == 2, "invalid rank for upsampling");
        tensor out_vol(tensor_shape(
            input.shape().width_ * size_,
            input.shape().depth_), 0);
        for (std::size_t x = 0; x < out_vol.shape().width_; ++x)
        {
            for (std::size_t z = 0; z < out_vol.shape().depth_; ++z)
            {
                const std::size_t x_in = x / size_;
                out_vol.set(tensor_pos(x, z), input.get(tensor_pos(x_in, z)));
            }
        }
        return {out_vol};
    }

    std::size_t size_;
};

} } // namespace fdeep, namespace internal
