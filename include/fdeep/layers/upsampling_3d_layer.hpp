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

namespace fdeep {
namespace internal {

    class upsampling_3d_layer : public layer {
    public:
        explicit upsampling_3d_layer(const std::string& name,
            const shape3& scale_factor)
            : layer(name)
            , scale_factor_(scale_factor)
        {
        }

    protected:
        tensors apply_impl(const tensors& inputs) const override final
        {
            const auto& input = single_tensor_from_tensors(inputs);
            return { resize3d_nearest(
                input,
                shape3(
                    scale_factor_.size_dim_4_ * input.shape().size_dim_4_,
                    scale_factor_.height_ * input.shape().height_,
                    scale_factor_.width_ * input.shape().width_)) };
        }
        shape3 scale_factor_;
    };

}
}
