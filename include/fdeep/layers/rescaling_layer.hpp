// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>
#include <vector>

namespace fdeep { namespace internal
{

class rescaling_layer : public layer
{
public:
    explicit rescaling_layer(const std::string& name,
        float_type scale, float_type offset)
        : layer(name),
        scale_(scale),
        offset_(offset)
    {
    }
protected:
    float_type scale_;
    float_type offset_;
    static float_type rescale_value(float_type scale, float_type offset, float_type x)
    {
        return scale * x + offset;
    }
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto f = fplus::bind_1st_and_2nd_of_3(rescale_value, scale_, offset_);
        const auto rescale_tensor = fplus::bind_1st_of_2(transform_tensor<decltype(f)>, f);
        return fplus::transform(rescale_tensor, inputs);
    }
};

} } // namespace fdeep, namespace internal
