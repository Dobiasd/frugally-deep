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
    tensor5s apply_impl(const tensor5s& inputs) const override final
    {
        assertion(inputs.size() == 1, "invalid number of inputs tensors");
        const auto& input = inputs.front();

        if (input.shape().rank() == 1)
        {
            return {upsampling_1d_rank_1(input)};
        }
        else if (input.shape().rank() == 2)
        {
            return {upsampling_1d_rank_2(input)};
        }
        else
        {
            raise_error("invalid input shape for Upsampling1D");
            return inputs;
        }
    }

    tensor5 upsampling_1d_rank_1(const tensor5& input) const
    {
        tensor5 out_vol(shape5(1, 1,
            input.shape().height_,
            input.shape().width_,
            input.shape().depth_ * size_), 0);
        for (std::size_t y = 0; y < out_vol.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < out_vol.shape().width_; ++x)
            {
                for (std::size_t z = 0; z < out_vol.shape().depth_; ++z)
                {
                    const std::size_t z_in = z / size_;
                    out_vol.set(0, 0, y, x, z, input.get(0, 0, y, x, z_in));
                }
            }
        }
        return {out_vol};
    }

    tensor5 upsampling_1d_rank_2(const tensor5& input) const
    {
        tensor5 out_vol(shape5(1, 1,
            input.shape().height_,
            input.shape().width_ * size_,
            input.shape().depth_), 0);
        for (std::size_t y = 0; y < out_vol.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < out_vol.shape().width_; ++x)
            {
                for (std::size_t z = 0; z < out_vol.shape().depth_; ++z)
                {
                    const std::size_t x_in = x / size_;
                    out_vol.set(0, 0, y, x, z, input.get(0, 0, y, x_in, z));
                }
            }
        }
        return {out_vol};
    }

    std::size_t size_;
};

} } // namespace fdeep, namespace internal
