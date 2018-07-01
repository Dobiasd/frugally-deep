// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <string>

namespace fdeep { namespace internal
{

class prelu_layer : public layer
{
public:
    explicit prelu_layer(const std::string& name, const float_vec& alpha, std::vector<std::size_t> shared_axes = std::vector<std::size_t>()) :
        layer(name), alpha_(fplus::make_shared_ref<float_vec>(alpha)), shared_axes_(shared_axes)
    {
    }
protected:
    fdeep::shared_float_vec alpha_;
    std::vector<std::size_t> shared_axes_;
    tensor3s apply_impl(const tensor3s& input) const override
    {
        const bool height_shared = fplus::is_elem_of(1, shared_axes_);
        const bool width_shared = fplus::is_elem_of(2, shared_axes_);
        const bool channels_shared = fplus::is_elem_of(3, shared_axes_);
        size_t width = width_shared ? 1 : input[0].shape().width_;
        size_t height = height_shared ? 1 : input[0].shape().height_;

        fdeep::tensor3 out(input[0].shape(), 1.0f);
        for (std::size_t z = 0; z < out.shape().depth_; ++z)
        {
            for (std::size_t y = 0; y < out.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < out.shape().width_; ++x)
                {
                    if (input[0].get(z, y, x) > 0)
                    {
                        out.set(z, y, x, input[0].get(z, y, x));
                    }
                    else
                    {
                        size_t y_temp = height_shared ? 0 : y;
                        size_t x_temp = width_shared ? 0 : x;
                        size_t z_temp = channels_shared ? 0 : z;
                        size_t pos = z_temp * height * width + y_temp * width + x_temp;
                        out.set(z, y, x, alpha_->at(pos) * input[0].get(z, y, x));
                    }
                }
            }
        }
        return { out };
    }
};

} } // namespace fdeep, namespace internal