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

struct convolution_filter_matrices
{
    tensor_shape filter_shape_;
    std::size_t filter_count_;
    float_vec biases_;
    bool use_bias_;
    tensor filter_mats_;
};

inline convolution_filter_matrices generate_im2col_filter_matrix(
    const std::vector<filter>& filters)
{
    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(filter, shape, tensor_shape), filters),
        "all filters must have the same shape");

    const auto biases = fplus::transform_convert<float_vec>(
        fplus_c_mem_fn_t(filter, get_bias, float_type),
        filters);

    const bool use_bias =
        fplus::sum(biases) != static_cast<float_type>(0) ||
        !fplus::all_the_same(biases);

    const auto shape = filters.front().shape();

    tensor filter_mats = tensor(
        tensor_shape(shape.height_, shape.width_, shape.depth_, filters.size()),
        static_cast<float_type>(0));

    for (std::size_t y = 0; y < shape.height_; ++y)
    {
        for (std::size_t n = 0; n < filters.size(); ++n)
        {
            for (std::size_t x = 0; x < shape.width_; ++x)
            {
                for (std::size_t z = 0; z < shape.depth_; ++z)
                {
                    filter_mats.set(tensor_pos(y, x, z, n),
                        filters[n].get(tensor_pos(y, x, z)));
                }
            }
        }
    }

    return {shape, filters.size(), biases, use_bias, filter_mats};
}

inline convolution_filter_matrices generate_im2col_single_filter_matrix(
    const filter& filter)
{
    return generate_im2col_filter_matrix(filter_vec(1, filter));
}

inline tensor init_conv_output_tensor(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t out_depth,
    std::size_t rank,
    const convolution_filter_matrices& filter_mat)
{
    tensor output(tensor_shape_with_changed_rank(
            tensor_shape(out_height, out_width, out_depth),
            rank),
        static_cast<float_type>(0));
    if (filter_mat.use_bias_) {
        const auto bias_ptr = &filter_mat.biases_.front();
        const auto bias_ptr_end = bias_ptr + out_depth;
        for (std::size_t y_out = 0; y_out < out_height; ++y_out)
        {
            for (std::size_t x_out = 0; x_out < out_width; ++x_out)
            {
                auto output_ptr = &output.get_ref_ignore_rank(tensor_pos(0, 0, y_out, x_out, 0));
                std::copy(bias_ptr, bias_ptr_end, output_ptr);
            }
        }
    }
    return output;
}

inline Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned, Eigen::OuterStride<>> get_im2col_mapping(
    const tensor& in,
    std::size_t f_width,
    std::size_t f_depth,
    std::size_t strides_x,
    std::size_t out_width,
    std::size_t y,
    std::size_t y_filt)
{
    // To avoid using too much RAM, the input tensor is not materializezd
    // as an actual im2col matrix, but instead the too-small outer stride
    // of the matrix mapping is utilized to achieve the overlap the receptive fields.
    return Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned, Eigen::OuterStride<>>(
            const_cast<float_type*>(&in.get_ref_ignore_rank(tensor_pos(0, 0, y + y_filt, 0, 0))),
            static_cast<EigenIndex>(f_width * f_depth),
            static_cast<EigenIndex>(out_width),
            Eigen::OuterStride<>(static_cast<EigenIndex>(f_depth * strides_x)));
}

// Special version for convolution with strides_x == 1 and strides_y == 1.
// Reduces the forward-pass runtime of VGG19 about 15%, by using fewer but larger GEMMs.
inline tensor convolve_accumulative_s1x1(
    std::size_t out_height,
    std::size_t out_width,
    const convolution_filter_matrices& filter_mat,
    const tensor& in)
{
    const tensor& filter_mats = filter_mat.filter_mats_;
    const auto f_height = filter_mat.filter_shape_.height_;
    const auto f_width = filter_mat.filter_shape_.width_;
    const auto f_depth = filter_mat.filter_shape_.depth_;
    const auto out_depth = filter_mat.filter_count_;

    assertion(f_depth == in.shape().depth_, "filter depth does not match input");
    assertion(filter_mats.shape().size_dim_4_ == f_height, "incorrect number of filter levels in y direction");
    assertion(out_width == (in.shape().width_ - f_width) + 1, "output width does not match");
    assertion(out_depth == filter_mat.biases_.size(), "invlid bias count");

    tensor output = init_conv_output_tensor(out_height, out_width, out_depth, in.shape().rank(), filter_mat);

    const std::size_t out_width_temp = out_width + f_width - 1;
    tensor output_temp(tensor_shape_with_changed_rank(
                tensor_shape(out_height, out_width_temp, out_depth),
                in.shape().rank()),
            static_cast<float_type>(0));

    const auto mapping_width = out_width_temp * (out_height - 1) + out_width;

    for (std::size_t y_filt = 0; y_filt < f_height; ++y_filt)
    {
        const Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned>
            filter(const_cast<float_type*>(&filter_mats.get_ref_ignore_rank(tensor_pos(0, y_filt, 0, 0, 0))),
                static_cast<EigenIndex>(out_depth),
                static_cast<EigenIndex>(f_width * f_depth));
    
        const auto input = get_im2col_mapping(in, f_width, f_depth, 1, mapping_width, 0, y_filt);

        Eigen::Map<Eigen::Matrix<float_type, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Unaligned>
            output_temp_map(&output_temp.get_ref_ignore_rank(tensor_pos(0, 0, 0, 0, 0)),
            static_cast<EigenIndex>(out_depth),
            static_cast<EigenIndex>(mapping_width));
            
        output_temp_map.noalias() += filter * input;
    }

    // Dropping the superfluous results from "between" the rows.
    for (std::size_t y_out = 0; y_out < out_height; ++y_out)
    {
        for (std::size_t x_out = 0; x_out < out_width; ++x_out)
        {
            for (std::size_t z_out = 0; z_out < out_depth; ++z_out)
            {
                output.get_ref_ignore_rank(tensor_pos(0, 0, y_out, x_out, z_out)) +=
                output_temp.get_ref_ignore_rank(tensor_pos(0, 0, y_out, x_out, z_out));
            }
        }
    }
    
    return output;
}

inline tensor convolve_accumulative(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    const convolution_filter_matrices& filter_mat,
    const tensor& in)
{
    // Using the im2col method, the convolution is expressed as GEMMs for performance.
    // https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
    // https://github.com/tensorflow/tensorflow/blob/a0d784bdd31b27e013a7eac58a86ba62e86db299/tensorflow/core/kernels/conv_ops_using_gemm.cc
    // http://www.youtube.com/watch?v=pA4BsUK3oP4&t=36m22s
    const tensor& filter_mats = filter_mat.filter_mats_;
    const auto f_height = filter_mat.filter_shape_.height_;
    const auto f_width = filter_mat.filter_shape_.width_;
    const auto f_depth = filter_mat.filter_shape_.depth_;
    const auto out_depth = filter_mat.filter_count_;

    assertion(f_depth == in.shape().depth_, "filter depth does not match input");
    assertion(filter_mats.shape().size_dim_4_ == f_height, "incorrect number of filter levels in y direction");
    assertion(out_width == (in.shape().width_ - f_width) / strides_x + 1, "output width does not match");
    assertion(out_depth == filter_mat.biases_.size(), "invlid bias count");

    if (strides_x == 1 && strides_y == 1)
    {
        return convolve_accumulative_s1x1(out_height, out_width, filter_mat, in);
    }

    tensor output = init_conv_output_tensor(out_height, out_width, out_depth, in.shape().rank(), filter_mat);

    for (std::size_t y_filt = 0; y_filt < f_height; ++y_filt)
    {
        const Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned>
            filter(const_cast<float_type*>(&filter_mats.get_ref_ignore_rank(tensor_pos(0, y_filt, 0, 0, 0))),
                static_cast<EigenIndex>(out_depth),
                static_cast<EigenIndex>(f_width * f_depth));
        for (std::size_t y = 0, y_out = 0; y < in.shape().height_ + 1 - f_height; y += strides_y, ++y_out)
        {
            const auto input = get_im2col_mapping(in, f_width, f_depth, strides_x, out_width, y, y_filt);
            Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned>
                output_map(&output.get_ref_ignore_rank(tensor_pos(0, 0, y_out, 0, 0)),
                    static_cast<EigenIndex>(out_depth),
                    static_cast<EigenIndex>(out_width));
            
            output_map.noalias() += filter * input;
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

inline tensor convolve(
    const shape2& strides,
    const padding& pad_type,
    const convolution_filter_matrices& filter_mat,
    const tensor& input)
{
    assertion(filter_mat.filter_shape_.depth_ == input.shape().depth_,
        "invalid filter depth");

    const auto conv_cfg = preprocess_convolution(
        filter_mat.filter_shape_.without_depth(),
        strides, pad_type, input.shape().height_, input.shape().width_);

    // The padding step usually (on a VGG19 net) only takes about 1% of the overall runtime.
    // So the increased code complexity of doing it inside the convolution step
    // is probably not worth the small potential performance gain.
    const auto in_padded = pad_tensor(0, 0, 0,
        conv_cfg.pad_top_, conv_cfg.pad_bottom_, conv_cfg.pad_left_, conv_cfg.pad_right_,
        input);

    return convolve_accumulative(
        conv_cfg.out_height_, conv_cfg.out_width_,
        strides.height_, strides.width_,
        filter_mat,
        in_padded);
}

} } // namespace fdeep, namespace internal
