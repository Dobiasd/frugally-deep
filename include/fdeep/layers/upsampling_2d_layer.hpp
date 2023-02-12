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

class upsampling_2d_layer : public layer
{
public:
    explicit upsampling_2d_layer(const std::string& name,
        const shape2& scale_factor, const std::string& interpolation) :
    layer(name),
    scale_factor_(scale_factor),
    interpolation_(interpolation)
    {
    }
protected:
    tensors apply_impl(const tensors& inputs) const override final
    {
        const auto& input = single_tensor_from_tensors(inputs);
        if (interpolation_ == "nearest")
        {
            return {resize2d_nearest(
                input, shape2(scale_factor_.height_ * input.shape().height_, scale_factor_.width_ * input.shape().width_))};
        }
        else if (interpolation_ == "bilinear")
        {
            return {resize2d_bilinear(
                input, shape2(scale_factor_.height_ * input.shape().height_, scale_factor_.width_ * input.shape().width_))};
        }
        else
        {
            raise_error("Invalid interpolation method: " + interpolation_);
            return inputs;
        }
    }
    shape2 scale_factor_;
    std::string interpolation_;
};

} } // namespace fdeep, namespace internal
