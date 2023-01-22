// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/convolution3d.hpp"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

namespace fdeep { namespace internal
{

typedef void (*inner_pooling_func)(
    const tensor&, tensor& out,
    std::size_t, std::size_t, std::size_t,
    std::size_t, std::size_t, std::size_t,
    std::size_t, std::size_t, std::size_t, std::size_t,
    int, int, int
);

// Abstract base class for pooling layers
class pooling_3d_layer : public layer
{
public:
    explicit pooling_3d_layer(const std::string& name,
        const shape3& pool_size, const shape3& strides,
        padding p, const inner_pooling_func inner_f) :
        layer(name),
        pool_size_(pool_size),
        strides_(strides),
        padding_(p),
        inner_f_(inner_f)
    {
    }
protected:
    tensor pool(const tensor& in) const
    {
        const auto conv_cfg = preprocess_convolution_3d(
            shape3(pool_size_.size_dim_4_ , pool_size_.height_, pool_size_.width_),
            shape3(strides_.size_dim_4_, strides_.height_, strides_.width_),
            padding_, in.shape().size_dim_4_, in.shape().height_, in.shape().width_);

        int pad_front_int = static_cast<int>(conv_cfg.pad_front_);
        int pad_top_int = static_cast<int>(conv_cfg.pad_top_);
        int pad_left_int = static_cast<int>(conv_cfg.pad_left_);

        const std::size_t out_size_d4 = conv_cfg.out_size_d4_;
        const std::size_t out_height = conv_cfg.out_height_;
        const std::size_t out_width = conv_cfg.out_width_;

        tensor out(
            tensor_shape_with_changed_rank(
                tensor_shape(out_size_d4, out_height, out_width, in.shape().depth_),
                in.shape().rank()),
            0);

        for (std::size_t d4 = 0; d4 < out_size_d4; ++d4)
        {
            for (std::size_t y = 0; y < out_height; ++y)
            {
                for (std::size_t x = 0; x < out_width; ++x)
                {
                    for (std::size_t z = 0; z < in.shape().depth_; ++z)
                    {
                        inner_f_(in, out,
                            pool_size_.size_dim_4_, pool_size_.height_, pool_size_.width_,
                            strides_.size_dim_4_, strides_.height_, strides_.width_,
                            d4, y, x, z,
                            pad_front_int, pad_top_int, pad_left_int);
                    }
                }
            }
        }
        return out;
    }

    tensors apply_impl(const tensors& inputs) const override final
    {
        const auto& input = single_tensor_from_tensors(inputs);
        return {pool(input)};
    }

    shape3 pool_size_;
    shape3 strides_;
    padding padding_;
    inner_pooling_func inner_f_;
};

} } // namespace fdeep, namespace internal
