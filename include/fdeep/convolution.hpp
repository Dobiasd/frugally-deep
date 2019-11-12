// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/filter.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <numeric>
#include <vector>

namespace fdeep { namespace internal
{

// todo: Remove. Just save raw filters on layers.
struct im2col_filter_matrix
{
    shape5 filter_shape_;
    std::size_t filter_count_;
    std::vector<filter> filters;
};

inline im2col_filter_matrix generate_im2col_filter_matrix(
    const std::vector<filter>& filters)
{
    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(filter, shape, shape5), filters),
        "all filters must have the same shape");

    return {filters.front().shape(), filters.size(), filters};
}

inline im2col_filter_matrix generate_im2col_single_filter_matrix(
    const filter& filter)
{
    return generate_im2col_filter_matrix(filter_vec(1, filter));
}

inline float_type dot_product(
    const float_type* xs,
    const float_type* ys,
    std::size_t n)
{
    // todo: respect float_type
    Eigen::Map<Eigen::VectorXf> vx(const_cast<float*>(xs), static_cast<EigenIndex>(n));
    Eigen::Map<Eigen::VectorXf> vy(const_cast<float*>(ys), static_cast<EigenIndex>(n));
    return vx.adjoint() * vy;
}

inline tensor5 convolve_accumulative(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    const im2col_filter_matrix& filter_mat,
    const tensor5& in)
{
    assertion(in.shape().rank() <= 3, "invalid rank for input tensor");

    const std::vector<filter>& filters = filter_mat.filters;
    const auto f_height = filter_mat.filter_shape_.height_;
    const auto f_width = filter_mat.filter_shape_.width_;
    const auto f_depth = filter_mat.filter_shape_.depth_;
    const auto out_depth = filters.size();

    assertion(f_depth == in.shape().depth_, "filter depth does not match input");

    tensor5 output(shape5(1, 1, out_height, out_width, out_depth), static_cast<float>(0));

    for (std::size_t z_out = 0; z_out < out_depth; ++z_out)
    {
        for (std::size_t y_filt = 0; y_filt < f_height; ++y_filt)
        {
            const float_type* filter_ptr = &(filters[z_out].get_tensor5().get_ref(0, 0, y_filt, 0, 0));

            for (std::size_t y = 0, y_out = 0; y < in.shape().height_ + 1 - f_height; y += strides_y, ++y_out)
            {
                for (std::size_t x = 0, x_out = 0; x < in.shape().width_ + 1 - f_width; x += strides_x, ++x_out)
                {
                    auto current = output.get(0, 0, y_out, x_out, z_out);
                    const float_type* input_ptr = &in.get_ref(0, 0, y + y_filt, x, 0);
                    const auto addend = dot_product(input_ptr, filter_ptr, f_width * f_depth);
                    output.set(0, 0, y_out, x_out, z_out, current + addend);
                }
            }
        }
    }

    for (std::size_t y_out = 0; y_out < out_height; ++y_out)
    {
        for (std::size_t x_out = 0; x_out < out_width; ++x_out)
        {
            for (std::size_t z_out = 0; z_out < out_depth; ++z_out)
            {
                const float_type bias = filters[z_out].get_bias();
                output.set(0, 0, y_out, x_out, z_out, output.get(0, 0, y_out, x_out, z_out) + bias);
            }
        }
    }
    return output;
}

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
    const shape2& strides,
    const padding& pad_type,
    const im2col_filter_matrix& filter_mat,
    const tensor5& input)
{
    assertion(filter_mat.filter_shape_.depth_ == input.shape().depth_,
        "invalid filter depth");

    const auto conv_cfg = preprocess_convolution(
        filter_mat.filter_shape_.without_depth(),
        strides, pad_type, input.shape().height_, input.shape().width_);

    const std::size_t out_height = conv_cfg.out_height_;
    const std::size_t out_width = conv_cfg.out_width_;

    const auto in_padded = pad_tensor5(0,
        conv_cfg.pad_top_, conv_cfg.pad_bottom_, conv_cfg.pad_left_, conv_cfg.pad_right_,
        input);

    return convolve_accumulative(
        out_height, out_width,
        strides.height_, strides.width_,
        filter_mat, in_padded);
}

} } // namespace fdeep, namespace internal
