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
    tensor_shape filter_shape_;
    std::size_t filter_count_;
};

inline im2col_filter_matrix generate_im2col_filter_matrix(
    const std::vector<filter>& filters)
{
    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(filter, shape, tensor_shape), filters),
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
                    b(b_y, b_x++) = filter.get(tensor_pos(yf, xf, zf));
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
inline tensor convolve_im2col(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    const im2col_filter_matrix& filter_mat,
    const tensor& in_padded)
{
    const auto fy = filter_mat.filter_shape_.height_;
    const auto fx = filter_mat.filter_shape_.width_;
    const auto fz = filter_mat.filter_shape_.depth_;
    const EigenIndex a_cols = static_cast<EigenIndex>(out_height * out_width);

    const std::size_t val_cnt =
        static_cast<std::size_t>(filter_mat.mat_.rows() * a_cols);
    assertion(val_cnt % (out_height * out_width) == 0,
        "Can not calculate out_depth");

    const std::size_t out_depth = val_cnt / (out_height * out_width);
    assertion(val_cnt == out_depth * out_height * out_width,
        "Invalid target size");

    shared_float_vec res_vec = fplus::make_shared_ref<float_vec>();
    res_vec->resize(static_cast<std::size_t>(out_depth * out_height * out_width));

    const EigenIndex a_rows = static_cast<EigenIndex>(fy * fx * fz + 1);
    const EigenIndex a_max_size_bytes = 16 * 1024 * 1024;
    EigenIndex step_size = a_max_size_bytes / (a_rows * static_cast<EigenIndex>(sizeof(float_type)));
    EigenIndex alignment_step = 64 / sizeof(float_type);
    step_size = (step_size / alignment_step) * alignment_step;
    step_size = std::max(static_cast<EigenIndex>(1), step_size);

    ColMajorMatrixXf a(a_rows, step_size);
    EigenIndex a_x_virtual = 0;
    EigenIndex last_gem_a_x = 0;
    for (std::size_t y = 0; y < out_height; ++y)
    {
        for (std::size_t x = 0; x < out_width; ++x)
        {
            EigenIndex a_y = 0;
            for (std::size_t yf = 0; yf < fy; ++yf)
            {
                const auto p = &(in_padded.get_ref_ignore_rank(tensor_pos(
                        strides_y * y + yf,
                        strides_x * x,
                        0)));
                const auto a_x = a_x_virtual % step_size;
                // https://stackoverflow.com/a/9980859/1866775
                std::copy(p, p + fx * fz, &a(a_y, a_x));
                a_y += static_cast<EigenIndex>(fx * fz);
                a(a_y, a_x) = static_cast<float_type>(1);
            }
            ++a_x_virtual;
            if (a_x_virtual >= last_gem_a_x + step_size)
            {
                MappedColMajorMatrixXf out_mat_map(
                    res_vec->data() + filter_mat.mat_.rows() * last_gem_a_x,
                    static_cast<EigenIndex>(filter_mat.mat_.rows()),
                    static_cast<EigenIndex>(a_x_virtual - last_gem_a_x));
                out_mat_map.noalias() = filter_mat.mat_ * a;
                last_gem_a_x = a_x_virtual;
            }
        }
    }
    if (a_x_virtual != last_gem_a_x)
    {
        EigenIndex fields_left = a_x_virtual - last_gem_a_x;
        MappedColMajorMatrixXf a_map(a.data(), a.rows(), fields_left);
        MappedColMajorMatrixXf out_mat_map(
            res_vec->data() + filter_mat.mat_.rows() * last_gem_a_x,
            static_cast<EigenIndex>(filter_mat.mat_.rows()),
            static_cast<EigenIndex>(fields_left));
        // https://stackoverflow.com/questions/48644724/multiply-two-eigen-matrices-directly-into-memory-of-target-matrix
        out_mat_map.noalias() = filter_mat.mat_ * a_map;
    }

    return tensor(
        tensor_shape_with_changed_rank(
            tensor_shape(out_height, out_width, out_depth),
            in_padded.shape().rank()),
        res_vec);
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

inline tensor convolve(
    const shape2& strides,
    const padding& pad_type,
    const im2col_filter_matrix& filter_mat,
    const tensor& input)
{
    assertion(filter_mat.filter_shape_.depth_ == input.shape().depth_,
        "invalid filter depth");

    const auto conv_cfg = preprocess_convolution(
        filter_mat.filter_shape_.without_depth(),
        strides, pad_type, input.shape().height_, input.shape().width_);

    const std::size_t out_height = conv_cfg.out_height_;
    const std::size_t out_width = conv_cfg.out_width_;

    const auto in_padded = pad_tensor(0,
        conv_cfg.pad_top_, conv_cfg.pad_bottom_, conv_cfg.pad_left_, conv_cfg.pad_right_,
        input);

    // todo: Find out under which circumstances this version is faster compared to the accumulative convolution and why.
    return convolve_im2col(
        out_height, out_width,
        strides.height_, strides.width_,
        filter_mat, in_padded);
}

} } // namespace fdeep, namespace internal
