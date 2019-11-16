// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/filter.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "tensorflow/core/kernels/eigen_spatial_convolutions-inl.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace fdeep { namespace internal
{

struct filter_tensor
{
    typedef Eigen::Tensor<float_type, 4>::Index EigenTensorIndex;
    Eigen::Tensor<float_type, 4> tensor_;
    float_vec bias_;
    shape5 filter_shape() const
    {
        return shape5(
            1,
            1,
            static_cast<std::size_t>(tensor_.dimensions()[2]),
            static_cast<std::size_t>(tensor_.dimensions()[3]),
            static_cast<std::size_t>(tensor_.dimensions()[1]));
    }
    std::size_t filter_count() const
    {
        return static_cast<std::size_t>(tensor_.dimensions()[0]);
    }
};

inline filter_tensor generate_filter_tensor(
    const std::vector<filter>& filters)
{
    assertion(fplus::all_the_same_on(
        fplus_c_mem_fn_t(filter, shape, shape5), filters),
        "all filters must have the same shape");

    const std::size_t fy = filters.front().shape().height_;
    const std::size_t fx = filters.front().shape().width_;
    const std::size_t fz = filters.front().shape().depth_;
    Eigen::Tensor<float_type, 4> t(
        static_cast<filter_tensor::EigenTensorIndex>(filters.size()),
        static_cast<filter_tensor::EigenTensorIndex>(fz),
        static_cast<filter_tensor::EigenTensorIndex>(fy),
        static_cast<filter_tensor::EigenTensorIndex>(fx));
    for (std::size_t f = 0; f < filters.size(); ++f)
    {
        for (std::size_t yf = 0; yf < fy; ++yf)
        {
            for (std::size_t xf = 0; xf < fx; ++xf)
            {
                for (std::size_t zf = 0; zf < fz; ++zf)
                {
                    t(
                        static_cast<filter_tensor::EigenTensorIndex>(f),
                        static_cast<filter_tensor::EigenTensorIndex>(zf),
                        static_cast<filter_tensor::EigenTensorIndex>(yf),
                        static_cast<filter_tensor::EigenTensorIndex>(xf)) =
                            filters[f].get(yf, xf, zf);
                }
            }
        }
    }

    float_vec bias = fplus::transform(
        fplus_c_mem_fn_t(filter, get_bias, float_type), filters);

    return {t, bias};
}

inline filter_tensor generate_im2col_single_filter_matrix(
    const filter& filter)
{
    return generate_filter_tensor(filter_vec(1, filter));
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
    const filter_tensor& filter_mat,
    const tensor5& input)
{
    assertion(input.shape().size_dim_5_ == 1, "invalid input shape");
    assertion(input.shape().size_dim_4_ == 1, "invalid input shape");
    assertion(filter_mat.filter_shape().depth_ == input.shape().depth_,
        "invalid filter depth");

    const auto conv_cfg = preprocess_convolution(
        filter_mat.filter_shape().without_depth(),
        strides, pad_type, input.shape().height_, input.shape().width_);

    const std::size_t out_height = conv_cfg.out_height_;
    const std::size_t out_width = conv_cfg.out_width_;
    const std::size_t filter_count = filter_mat.filter_count();

    const auto in_padded = pad_tensor5(0,
        conv_cfg.pad_top_, conv_cfg.pad_bottom_, conv_cfg.pad_left_, conv_cfg.pad_right_,
        input);

    tensor5 result(shape5(1, 1, out_height, out_width, filter_count),
        static_cast<float_type>(0));

    typedef Eigen::Tensor<float_type, 3> EigenTensor3;
    typedef EigenTensor3::Index EigenTensor3Index;

    //Eigen::Map<Eigen::Tensor<float_type, 3>> source(
        //in_padded.get_eigen_tensor().data,
    Eigen::Tensor<float_type, 3> source(
        static_cast<EigenTensor3Index>(in_padded.shape().depth_),
        static_cast<EigenTensor3Index>(in_padded.shape().height_),
        static_cast<EigenTensor3Index>(in_padded.shape().width_));

    for (std::size_t z = 0; z < in_padded.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < in_padded.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < in_padded.shape().width_; ++x)
            {
                source(
                    static_cast<EigenTensor3Index>(z),
                    static_cast<EigenTensor3Index>(y),
                    static_cast<EigenTensor3Index>(x)) = in_padded.get(0, 0, y, x, z);
            }
        }
    }

    // todo: move padding here
    // todo: move bias here
    const auto eigen_padding = Eigen::PADDING_VALID;
    Eigen::Tensor<float_type, 3> dest(
        static_cast<EigenTensor3Index>(filter_count),
        static_cast<EigenTensor3Index>(out_height),
        static_cast<EigenTensor3Index>(out_width));
    dest = SpatialConvolution(source, filter_mat.tensor_,
        static_cast<Eigen::Index>(strides.height_),
        static_cast<Eigen::Index>(strides.width_),
        eigen_padding,
        1,
        1,
        Eigen::NoOpOutputKernel(),
        0, 0,
        0, 0);

    for (std::size_t z = 0; z < result.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < result.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < result.shape().width_; ++x)
            {
                result.set(0, 0, y, x, z,
                    dest(
                        static_cast<EigenTensor3Index>(z),
                        static_cast<EigenTensor3Index>(y),
                        static_cast<EigenTensor3Index>(x)) + filter_mat.bias_[z]);
            }
        }
    }

    return result;
}

} } // namespace fdeep, namespace internal
