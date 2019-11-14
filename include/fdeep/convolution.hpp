// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/filter.hpp"

#include <immintrin.h>

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
    std::vector<filter> filters_;
    float_vec biases_;
    bool use_bias_;
    tensor5 filter_tensor_;
};

inline im2col_filter_matrix generate_im2col_filter_matrix(
    const std::vector<filter>& filters)
{
    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(filter, shape, shape5), filters),
        "all filters must have the same shape");

    const auto biases = fplus::transform_convert<float_vec>(
        fplus_c_mem_fn_t(filter, get_bias, float_type),
        filters);

    const bool use_bias =
        fplus::sum(biases) != static_cast<float_type>(0) ||
        !fplus::all_the_same(biases);

    const auto shape = filters.front().shape();

    tensor5 filter_tensor(
        shape5(1, shape.height_, filters.size(), shape.width_, shape.depth_),
        static_cast<float>(0));

    for (std::size_t y = 0; y < shape.height_; ++y)
    {
        for (std::size_t n = 0; n < filters.size(); ++n)
        {
            for (std::size_t x = 0; x < shape.width_; ++x)
            {
                for (std::size_t z = 0; z < shape.depth_; ++z)
                {
                    filter_tensor.set(
                        0, y, n, x, z,
                        filters[n].get(y, x, z));
                }
            }
        }
    }

    return {shape, filters.size(), filters, biases, use_bias, filter_tensor};
}

inline im2col_filter_matrix generate_im2col_single_filter_matrix(
    const filter& filter)
{
    return generate_im2col_filter_matrix(filter_vec(1, filter));
}


FDEEP_FORCE_INLINE float_type dot_product(
    const float_type* xs,
    const float_type* ys,
    int n_div_8)
{
    const auto xs_aligned = reinterpret_cast<float_type*>(__builtin_assume_aligned(xs, EIGEN_MAX_ALIGN_BYTES));
    const auto ys_aligned = reinterpret_cast<float_type*>(__builtin_assume_aligned(ys, EIGEN_MAX_ALIGN_BYTES));

    // Naive version: Works.
    /*
    float result = 0;
    for (int i = 0; i < 8 * n_div_8; ++i)
    {
        result += xs_aligned[i] * ys_aligned[i];
    }
    return result;
    */

    // Eigen version: Works.
    /*
    Eigen::Map<Eigen::Matrix<float_type, 1, Eigen::Dynamic>, Eigen::Aligned> vx(xs_aligned, static_cast<EigenIndex>(8 * n_div_8));
    Eigen::Map<Eigen::Matrix<float_type, Eigen::Dynamic, 1>, Eigen::Aligned> vy(ys_aligned, static_cast<EigenIndex>(8 * n_div_8));
    return vx * vy;
    */

    // AVX-256 version: Works.
    // https://stackoverflow.com/questions/13000316/how-to-access-components-of-the-256-bit-ps-vector
    // todo: respect float type, or drop support for double
    // todo: if this is fast, maybe get rid of Eigen as dependency
    float result = 0;
    for (int i = 0; i < n_div_8; ++i)
    {
        const auto xs8 = _mm256_load_ps(&(xs_aligned[8*i]));
        const auto ys8 = _mm256_load_ps(&(ys_aligned[8*i]));
        const auto res = _mm256_dp_ps(xs8, ys8, 0xff);
        result += res[0] + res[4];
    }
    return result;
}

FDEEP_FORCE_INLINE tensor5 convolve_accumulative(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    const im2col_filter_matrix& filter_mat,
    const tensor5& in)
{
    assertion(in.shape().rank() <= 3, "invalid rank for input tensor");

    const std::vector<filter>& filters = filter_mat.filters_;
    const tensor5& filter_tensor = filter_mat.filter_tensor_;
    const auto f_height = filter_mat.filter_shape_.height_;
    const auto f_width = filter_mat.filter_shape_.width_;
    const auto f_depth = filter_mat.filter_shape_.depth_;
    const auto out_depth = filters.size();

    assertion(f_depth == in.shape().depth_, "filter depth does not match input");

    const auto f_memory_depth = filter_mat.filter_shape_.depth_in_memory();
    const auto in_memory_depth = in.shape().depth_in_memory();
    assertion(f_memory_depth == in_memory_depth, "invalid alignment");

    tensor5 output(shape5(1, 1, out_height, out_width, out_depth), static_cast<float_type>(0));
    const int dot_product_dims = static_cast<int>(f_width * f_memory_depth);
    assertion(dot_product_dims % 8 == 0, "alignment does not match dot-product dimensions");
    const int dot_product_dims_div_8 = dot_product_dims / 8;

    static_assert(EIGEN_MAX_ALIGN_BYTES % 32 == 0, "invalid alignment");

    //std::cout << dot_product_dims << ": " << filter_mat.filter_shape_.volume() << " vs. " << in.shape().volume() << " vs. " << out_depth * out_height * out_width << std::endl;

    // todo allow prefetch for other compilers too
    for (std::size_t y_filt = 0; y_filt < f_height; ++y_filt)
    {
        //__builtin_prefetch(&(filter_tensor.get_ref(0, y_filt, 0, 0, 0)));
        for (std::size_t y = 0, y_out = 0; y < in.shape().height_ + 1 - f_height; y += strides_y, ++y_out)
        {
            for (std::size_t x = 0, x_out = 0; x < in.shape().width_ + 1 - f_width; x += strides_x, ++x_out)
            {
                const float_type* input_ptr = &in.get_ref(0, 0, y + y_filt, x, 0);
                //__builtin_prefetch(input_ptr);
                for (std::size_t z_out = 0; z_out < out_depth; ++z_out)
                {
                    const float_type* filter_ptr = &(filter_tensor.get_ref(0, y_filt, z_out, 0, 0));
                    output.get_ref(0, 0, y_out, x_out, z_out) += dot_product(filter_ptr, input_ptr, dot_product_dims_div_8);
                }
            }
        }
    }

    if (filter_mat.use_bias_) {
        for (std::size_t y_out = 0; y_out < out_height; ++y_out)
        {
            for (std::size_t x_out = 0; x_out < out_width; ++x_out)
            {
                for (std::size_t z_out = 0; z_out < out_depth; ++z_out)
                {
                    output.get_ref(0, 0, y_out, x_out, z_out) += filter_mat.biases_[z_out];
                }
            }
        }
    }
    //std::cout << "---" << std::endl;
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
        filter_mat,
        in_padded);
}

} } // namespace fdeep, namespace internal
