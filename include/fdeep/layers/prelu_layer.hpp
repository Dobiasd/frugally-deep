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
    explicit prelu_layer(const std::string& name, const float_vec& alpha,
            std::vector<std::size_t> shared_axes)
        : layer(name),
        alpha_(fplus::make_shared_ref<float_vec>(alpha)),
        shared_axes_(shared_axes)
    {
    }
protected:
    fdeep::shared_float_vec alpha_;
    std::vector<std::size_t> shared_axes_;
    tensors apply_impl(const tensors& input) const override
    {
        // We need to shift shared_axes if the original Keras tensor
        // was one or two dimensional.
        // We detect this by checking if the axes indicated in shared_axes
        // has length 1.
        // For this to work we need to remove axes with length 1
        // from shared axes in Python.
        std::vector<std::size_t> shared_axes_shifted;
        std::size_t shift = 0;
        for (std::size_t i = 0; i < shared_axes_.size(); ++i)
        {
            if ((shared_axes_[i] == 1 && input[0].shape().height_ == 1) ||
                (shared_axes_[i] == 2 && input[0].shape().width_ == 1))
            {
                shift++;
            }
            shared_axes_shifted.push_back(shared_axes_[i] + shift);
        }

        const bool height_shared = fplus::is_elem_of(1, shared_axes_shifted);
        const bool width_shared = fplus::is_elem_of(2, shared_axes_shifted);
        const bool channels_shared = fplus::is_elem_of(3, shared_axes_shifted);
        const size_t width = width_shared ? 1 : input[0].shape().width_;
        const size_t depth = channels_shared ? 1 : input[0].shape().depth_;

        fdeep::tensor out(input[0].shape(), 1.0f);
        for (std::size_t y = 0; y < out.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < out.shape().width_; ++x)
            {
                for (std::size_t z = 0; z < out.shape().depth_; ++z)
                {
                    if (input[0].get_ignore_rank(tensor_pos(y, x, z)) > 0)
                    {
                        out.set_ignore_rank(tensor_pos(y, x, z),
                            input[0].get_ignore_rank(tensor_pos(y, x, z)));
                    }
                    else
                    {
                        const size_t y_temp = height_shared ? 0 : y;
                        const size_t x_temp = width_shared ? 0 : x;
                        const size_t z_temp = channels_shared ? 0 : z;
                        const size_t pos =
                            y_temp * width * depth +
                            x_temp * depth +
                            z_temp;
                        out.set_ignore_rank(tensor_pos(y, x, z), (*alpha_)[pos] *
                            input[0].get_ignore_rank(tensor_pos(y, x, z)));
                    }
                }
            }
        }
        return { out };
    }
};

} } // namespace fdeep, namespace internal