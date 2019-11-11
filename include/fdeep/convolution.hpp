// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/filter.hpp"
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace fdeep { namespace internal
{

enum class padding { valid, same, causal };

struct convolution_config
{
    std::size_t pad_top_;
    std::size_t pad_bottom_;
    std::size_t pad_left_;
    std::size_t pad_right_;
    std::size_t out_height_;
    std::size_t out_width_;
};

inline convolution_config preprocess_convolution(
    const shape2& filter_shape,
    const shape2& strides,
    padding pad_type,
    std::size_t input_shape_height,
    std::size_t input_shape_width)
{
    // https://www.tensorflow.org/api_guides/python/nn#Convolution
    const int filter_height = static_cast<int>(filter_shape.height_);
    const int filter_width = static_cast<int>(filter_shape.width_);
    const int in_height = static_cast<int>(input_shape_height);
    const int in_width = static_cast<int>(input_shape_width);
    const int strides_y = static_cast<int>(strides.height_);
    const int strides_x = static_cast<int>(strides.width_);

    int out_height = 0;
    int out_width = 0;

    if (pad_type == padding::same || pad_type == padding::causal)
    {
        out_height = fplus::ceil(static_cast<float>(in_height) / static_cast<float>(strides_y) - 0.001);
        out_width  = fplus::ceil(static_cast<float>(in_width) / static_cast<float>(strides_x) - 0.001);
    }
    else
    {
        out_height = fplus::ceil(static_cast<float>(in_height - filter_height + 1) / static_cast<float>(strides_y) - 0.001);
        out_width = fplus::ceil(static_cast<float>(in_width - filter_width + 1) / static_cast<float>(strides_x) - 0.001);
    }

    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;

    if (pad_type == padding::same)
    {
        int pad_along_height = 0;
        int pad_along_width = 0;

        if (in_height % strides_y == 0)
            pad_along_height = std::max(filter_height - strides_y, 0);
        else
            pad_along_height = std::max(filter_height - (in_height % strides_y), 0);
        if (in_width % strides_x == 0)
            pad_along_width = std::max(filter_width - strides_x, 0);
        else
            pad_along_width = std::max(filter_width - (in_width % strides_x), 0);

        pad_top = pad_along_height / 2;
        pad_bottom = pad_along_height - pad_top;
        pad_left = pad_along_width / 2;
        pad_right = pad_along_width - pad_left;
    }
    else if (pad_type == padding::causal)
    {
        pad_top = filter_height - 1;
        pad_left = filter_width - 1;
    }

    std::size_t out_height_size_t = fplus::integral_cast_throw<std::size_t>(out_height);
    std::size_t out_width_size_t = fplus::integral_cast_throw<std::size_t>(out_width);
    std::size_t pad_top_size_t = fplus::integral_cast_throw<std::size_t>(pad_top);
    std::size_t pad_bottom_size_t = fplus::integral_cast_throw<std::size_t>(pad_bottom);
    std::size_t pad_left_size_t = fplus::integral_cast_throw<std::size_t>(pad_left);
    std::size_t pad_right_size_t = fplus::integral_cast_throw<std::size_t>(pad_right);

    return {pad_top_size_t, pad_bottom_size_t,
        pad_left_size_t, pad_right_size_t,
        out_height_size_t, out_width_size_t};
}

inline tensor5 convolve(
    const shape5 filter_shape,
    std::size_t k,
    const float_vec weights,
    const float_vec bias,
    const shape2 strides,
    const padding pad_type,
    const shape2 dilation_rate_,
    const tensor5& input)
{
    const auto conv_cfg = preprocess_convolution(
        filter_shape.without_depth(),
        strides, pad_type, input.shape().height_, input.shape().width_);

    const Eigen::TensorMap<Eigen::Tensor<float_type, 3>> t_orig(
        const_cast<float *>(input.as_vector()->data()),
        static_cast<int>(input.depth()),
        static_cast<int>(input.width()),
        static_cast<int>(input.height()));

    Eigen::array<std::pair<int, int>, 2> eigen_paddings;
    eigen_paddings[0] = std::make_pair(conv_cfg.pad_left_, conv_cfg.pad_right_);
    eigen_paddings[1] = std::make_pair(conv_cfg.pad_top_, conv_cfg.pad_bottom_);
    Eigen::array<Eigen::DenseIndex, 2> eigen_strides({strides.height_, strides.width_});
    const auto t = t_orig.pad(eigen_paddings).stride(strides);

    tensor5 output(shape5(1, 1, conv_cfg.out_height_, conv_cfg.out_width_, k), static_cast<float_type>(0));

    // see: https://stackoverflow.com/questions/58788433/how-to-use-eigentensorconvolve-with-multiple-kernels
    std::vector<tensor5> output_slices;

    // todo: Can be done in parallel.
    for (std::size_t i = 0; i < k; ++i) {
        const Eigen::TensorMap<Eigen::Tensor<float_type, 3>> f(
            const_cast<float *>(weights.data() + i * filter_shape.volume()),
            static_cast<int>(filter_shape.depth_),
            static_cast<int>(filter_shape.width_),
            static_cast<int>(filter_shape.height_));

        output_slices.push_back(tensor5(shape5(1, 1, conv_cfg.out_height_, conv_cfg.out_width_, 1), static_cast<float_type>(0)));
        Eigen::TensorMap<Eigen::Tensor<float_type, 3>> output_slice_map(
            const_cast<float *>(output_slices.back().as_vector()->data()),
            static_cast<int>(1),
            static_cast<int>(conv_cfg.out_width_),
            static_cast<int>(conv_cfg.out_height_));

        Eigen::array<ptrdiff_t, 3> dims({0, 1, 2});
        output_slice_map = t.convolve(f, dims);
    }

    return tensor5_from_depth_slices(output_slices);
}

} } // namespace fdeep, namespace internal
