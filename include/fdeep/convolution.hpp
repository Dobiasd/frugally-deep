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
#include <vector>

namespace fdeep { namespace internal
{

struct im2col_filter_matrix
{
    ColMajorMatrixXf mat_;
    shape5 filter_shape_;
    std::size_t filter_count_;
};

inline im2col_filter_matrix generate_im2col_filter_matrix(
    const std::vector<filter>& filters)
{
    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(filter, shape, shape5), filters),
        "all filters must have the same shape");

    const std::size_t fy = filters.front().shape().height_;
    const std::size_t fx = filters.front().shape().width_;
    const std::size_t fz = filters.front().shape().depth_;
    ColMajorMatrixXf b(filters.size(), fy * fx * fz + 1);
    EigenIndex b_y = 0;
    EigenIndex b_x = 0;
    for (std::size_t f = 0; f < filters.size(); ++f)
    {
        b_x = 0;
        const filter& filter = filters[f];
        for (std::size_t yf = 0; yf < fy; ++yf)
        {
            for (std::size_t xf = 0; xf < fx; ++xf)
            {
                for (std::size_t zf = 0; zf < fz; ++zf)
                {
                    b(b_y, b_x++) = filter.get(yf, xf, zf);
                }
            }
        }
        b(b_y, b_x++) = filter.get_bias();
        ++b_y;
    }
    return {b, filters.front().shape(), filters.size()};
}

inline im2col_filter_matrix generate_im2col_single_filter_matrix(
    const filter& filter)
{
    return generate_im2col_filter_matrix(filter_vec(1, filter));
}

// GEMM convolution, faster but uses more RAM
// https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
// https://github.com/tensorflow/tensorflow/blob/a0d784bdd31b27e013a7eac58a86ba62e86db299/tensorflow/core/kernels/conv_ops_using_gemm.cc
// http://www.youtube.com/watch?v=pA4BsUK3oP4&t=36m22s
inline tensor5 convolve_im2col(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    std::size_t offset_y,
    std::size_t offset_x,
    const im2col_filter_matrix& filter_mat,
    const tensor5& in_padded)
{
    const auto fy = filter_mat.filter_shape_.height_;
    const auto fx = filter_mat.filter_shape_.width_;
    const auto fz = filter_mat.filter_shape_.depth_;
    ColMajorMatrixXf a(fy * fx * fz + 1, out_height * out_width);
    EigenIndex a_x = 0;
    for (std::size_t y = 0; y < out_height; ++y)
    {
        for (std::size_t x = 0; x < out_width; ++x)
        {
            EigenIndex a_y = 0;
            for (std::size_t yf = 0; yf < fy; ++yf)
            {
                for (std::size_t xf = 0; xf < fx; ++xf)
                {
                    for (std::size_t zf = 0; zf < fz; ++zf)
                    {
                        a(a_y++, a_x) = in_padded.get(0, 0,
                                offset_y + strides_y * y + yf,
                                offset_x + strides_x * x + xf,
                                zf);
                    }
                }
                a(a_y, a_x) = static_cast<float_type>(1);
            }
            ++a_x;
        }
    }

    const std::size_t val_cnt =
        static_cast<std::size_t>(filter_mat.mat_.rows() * a.cols());
    assertion(val_cnt % (out_height * out_width) == 0,
        "Can not calculate out_depth");

    const std::size_t out_depth = val_cnt / (out_height * out_width);
    assertion(val_cnt == out_depth * out_height * out_width,
        "Invalid target size");

    shared_float_vec res_vec = fplus::make_shared_ref<float_vec>();
    res_vec->resize(static_cast<std::size_t>(out_depth * out_height * out_width));

    Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned> out_mat_map(
        res_vec->data(),
        static_cast<EigenIndex>(filter_mat.mat_.rows()),
        static_cast<EigenIndex>(a.cols()));

    // https://stackoverflow.com/questions/48644724/multiply-two-eigen-matrices-directly-into-memory-of-target-matrix
    out_mat_map.noalias() = filter_mat.mat_ * a;

    return tensor5(shape5(1, 1, out_height, out_width, out_depth), res_vec);
}

enum class padding { valid, same };

struct convolution_config
{
    std::size_t pad_top_;
    std::size_t pad_bottom_;
    std::size_t pad_left_;
    std::size_t pad_right_;
    std::size_t offset_y_;
    std::size_t offset_x_;
    std::size_t out_height_;
    std::size_t out_width_;
};

inline convolution_config preprocess_convolution(
    const shape2& filter_shape,
    const shape2& strides,
    padding pad_type,
    bool use_offset,
    const shape5& input_shape)
{
    // https://www.tensorflow.org/api_guides/python/nn#Convolution
    const int filter_height = static_cast<int>(filter_shape.height_);
    const int filter_width = static_cast<int>(filter_shape.width_);
    const int in_height = static_cast<int>(input_shape.height_);
    const int in_width = static_cast<int>(input_shape.width_);
    const int strides_y = static_cast<int>(strides.height_);
    const int strides_x = static_cast<int>(strides.width_);

    int out_height = fplus::ceil(static_cast<float>(in_height - filter_height + 1) / static_cast<float>(strides_y) - 0.001);
    int out_width = fplus::ceil(static_cast<float>(in_width - filter_width + 1) / static_cast<float>(strides_x) - 0.001);
    int pad_along_height = 0;
    int pad_along_width = 0;

    if (pad_type == padding::same)
    {
        out_height = fplus::ceil(static_cast<float>(in_height) / static_cast<float>(strides_y) - 0.001);
        out_width  = fplus::ceil(static_cast<float>(in_width) / static_cast<float>(strides_x) - 0.001);

        if (in_height % strides_y == 0)
            pad_along_height = std::max(filter_height - strides_y, 0);
        else
            pad_along_height = std::max(filter_height - (in_height % strides_y), 0);
        if (in_width % strides_x == 0)
            pad_along_width = std::max(filter_width - strides_x, 0);
        else
            pad_along_width = std::max(filter_width - (in_width % strides_x), 0);
    }
    const int pad_top = pad_along_height / 2;
    const int pad_bottom = pad_along_height - pad_top;
    const int pad_left = pad_along_width / 2;
    const int pad_right = pad_along_width - pad_left;

    int offset_y = 0;
    int offset_x = 0;

    if (use_offset)
    {
        offset_y = ((in_height + pad_top + pad_bottom - filter_height) % strides_y) / 2;
    }
    if (use_offset)
    {
        offset_x = ((in_width + pad_left + pad_right - filter_width) % strides_x) / 2;
    }

    std::size_t out_height_size_t = fplus::integral_cast_throw<std::size_t>(out_height);
    std::size_t out_width_size_t = fplus::integral_cast_throw<std::size_t>(out_width);
    std::size_t offset_y_size_t = fplus::integral_cast_throw<std::size_t>(offset_y);
    std::size_t offset_x_size_t = fplus::integral_cast_throw<std::size_t>(offset_x);
    std::size_t pad_top_size_t = fplus::integral_cast_throw<std::size_t>(pad_top);
    std::size_t pad_bottom_size_t = fplus::integral_cast_throw<std::size_t>(pad_bottom);
    std::size_t pad_left_size_t = fplus::integral_cast_throw<std::size_t>(pad_left);
    std::size_t pad_right_size_t = fplus::integral_cast_throw<std::size_t>(pad_right);

    return {pad_top_size_t, pad_bottom_size_t,
        pad_left_size_t, pad_right_size_t,
        offset_y_size_t, offset_x_size_t,
        out_height_size_t, out_width_size_t};
}

inline tensor5 convolve(
    const shape2& strides,
    const padding& pad_type,
    bool use_offset,
    const im2col_filter_matrix& filter_mat,
    const tensor5& input)
{
    assertion(filter_mat.filter_shape_.depth_ == input.shape().depth_,
        "invalid filter depth");

    const auto conv_cfg = preprocess_convolution(
        filter_mat.filter_shape_.without_depth(),
        strides, pad_type, use_offset, input.shape());

    const std::size_t offset_y = conv_cfg.offset_y_;
    const std::size_t offset_x = conv_cfg.offset_x_;
    const std::size_t out_height = conv_cfg.out_height_;
    const std::size_t out_width = conv_cfg.out_width_;

    const auto in_padded = pad_tensor5(0,
        conv_cfg.pad_top_, conv_cfg.pad_bottom_, conv_cfg.pad_left_, conv_cfg.pad_right_,
        input);

    return convolve_im2col(
        out_height, out_width,
        strides.height_, strides.width_,
        offset_y, offset_x,
        filter_mat, in_padded);
}

} } // namespace fdeep, namespace internal
