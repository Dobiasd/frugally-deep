// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

namespace fdeep { namespace internal
{

class prelu_layer : public layer
{
public:
    explicit prelu_layer(const std::string& name, const float_vec& alpha) :
        layer(name), alpha_(fplus::make_shared_ref<float_vec>(alpha))
    {
    }
protected:
    fdeep::shared_float_vec alpha_;
    tensor3s apply_impl(const tensor3s& input) const override
    {
        fdeep::tensor3 alpha_tensor3(input[0].shape(), alpha_);
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
                        out.set(z, y, x, alpha_tensor3.get(z, y, x) * input[0].get(z, y, x));
                    }
                }
            }
        }
        return { out };
    }
};

} } // namespace fdeep, namespace internal