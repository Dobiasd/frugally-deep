// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/convolution.h"
#include "frugally_deep/convolution_transpose.h"
#include "frugally_deep/filter.h"
#include "frugally_deep/size2d.h"
#include "frugally_deep/size3d.h"
#include "frugally_deep/layers/layer.h"

#include <fplus/fplus.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

namespace fd
{

// upconvolution layer
// aka. convolution transpose
// aka. backward strided convolution
// aka. fractionally strided convolution
// aka. deconvolution
class convolution_transpose_layer : public layer
{
public:
    static std::vector<filter> generate_filters(
        std::size_t depth, const size2d& filter_size, std::size_t k)
    {
        return std::vector<filter>(k, filter(matrix3d(
            size3d(depth, filter_size.height_, filter_size.width_)), 0));
    }
    explicit convolution_transpose_layer(
            const std::string& name, const size2d& filter_size,
            std::size_t k, std::size_t stride)
        : layer(name)
        //filters_(generate_filters(size_in.depth_, filter_size, k)),
        //padding_y_((size_out_.height_ / stride - size_in.height_ + filter_size.height_ - stride) / 2),
        //padding_x_((size_out_.width_ / stride - size_in.width_ + filter_size.width_ - stride) / 2),
        //stride_(stride)
    {
        assert(k != 0);
        assert(filter_size.width_ > 0); // todo remove
        assert(stride > 0); // todo remove
        //assert((size_out_.height_ / stride - size_in.height_ + filter_size.height_ - stride) % 2 == 0);
        //assert((size_out_.width_ / stride - size_in.width_ + filter_size.width_ - stride) % 2 == 0);
    }

    matrix3ds apply(const matrix3ds& inputs) const override
    {
        assert(inputs.size() == 1);
        const auto& input = inputs[0];
        return {convolve_transpose(
            stride_, padding_x_, padding_y_, filters_, input)};
    }
    filter_vec filters_;
    std::size_t padding_y_;
    std::size_t padding_x_;
    std::size_t stride_;
};

} // namespace fd
