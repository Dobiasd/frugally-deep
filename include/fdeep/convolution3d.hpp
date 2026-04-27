// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"

#include "fdeep/convolution.hpp"
#include "fdeep/filter.hpp"
#include "fdeep/shape3.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace fdeep {
namespace internal {

    struct convolution3d_config {
        std::size_t pad_front_;
        std::size_t pad_back_;
        std::size_t pad_top_;
        std::size_t pad_bottom_;
        std::size_t pad_left_;
        std::size_t pad_right_;
        std::size_t out_size_d4_;
        std::size_t out_height_;
        std::size_t out_width_;
    };

    inline convolution3d_config preprocess_convolution_3d(
        const shape3& filter_shape,
        const shape3& strides,
        padding pad_type,
        std::size_t input_shape_size_d4,
        std::size_t input_shape_height,
        std::size_t input_shape_width)
    {
        const int filter_size_d4 = static_cast<int>(filter_shape.size_dim_4_);
        const int filter_height = static_cast<int>(filter_shape.height_);
        const int filter_width = static_cast<int>(filter_shape.width_);
        const int in_size_d4 = static_cast<int>(input_shape_size_d4);
        const int in_height = static_cast<int>(input_shape_height);
        const int in_width = static_cast<int>(input_shape_width);
        const int strides_d4 = static_cast<int>(strides.size_dim_4_);
        const int strides_y = static_cast<int>(strides.height_);
        const int strides_x = static_cast<int>(strides.width_);

        int out_size_d4 = 0;
        int out_height = 0;
        int out_width = 0;

        if (pad_type == padding::same || pad_type == padding::causal) {
            out_size_d4 = fplus::ceil(static_cast<float>(in_size_d4) / static_cast<float>(strides_d4) - 0.001);
            out_height = fplus::ceil(static_cast<float>(in_height) / static_cast<float>(strides_y) - 0.001);
            out_width = fplus::ceil(static_cast<float>(in_width) / static_cast<float>(strides_x) - 0.001);
        } else {
            out_size_d4 = fplus::ceil(static_cast<float>(in_size_d4 - filter_size_d4 + 1) / static_cast<float>(strides_d4) - 0.001);
            out_height = fplus::ceil(static_cast<float>(in_height - filter_height + 1) / static_cast<float>(strides_y) - 0.001);
            out_width = fplus::ceil(static_cast<float>(in_width - filter_width + 1) / static_cast<float>(strides_x) - 0.001);
        }

        int pad_front = 0;
        int pad_back = 0;
        int pad_top = 0;
        int pad_bottom = 0;
        int pad_left = 0;
        int pad_right = 0;

        if (pad_type == padding::same) {
            int pad_along_d4 = 0;
            int pad_along_height = 0;
            int pad_along_width = 0;

            if (in_size_d4 % strides_d4 == 0)
                pad_along_d4 = std::max(filter_size_d4 - strides_d4, 0);
            else
                pad_along_d4 = std::max(filter_size_d4 - (in_size_d4 % strides_d4), 0);
            if (in_height % strides_y == 0)
                pad_along_height = std::max(filter_height - strides_y, 0);
            else
                pad_along_height = std::max(filter_height - (in_height % strides_y), 0);
            if (in_width % strides_x == 0)
                pad_along_width = std::max(filter_width - strides_x, 0);
            else
                pad_along_width = std::max(filter_width - (in_width % strides_x), 0);

            pad_front = pad_along_d4 / 2;
            pad_back = pad_along_d4 - pad_front;
            pad_top = pad_along_height / 2;
            pad_bottom = pad_along_height - pad_top;
            pad_left = pad_along_width / 2;
            pad_right = pad_along_width - pad_left;
        } else if (pad_type == padding::causal) {
            pad_front = filter_size_d4 - 1;
            pad_top = filter_height - 1;
            pad_left = filter_width - 1;
        }

        std::size_t out_size_d4_size_t = fplus::integral_cast_throw<std::size_t>(out_size_d4);
        std::size_t out_height_size_t = fplus::integral_cast_throw<std::size_t>(out_height);
        std::size_t out_width_size_t = fplus::integral_cast_throw<std::size_t>(out_width);
        std::size_t pad_front_size_t = fplus::integral_cast_throw<std::size_t>(pad_front);
        std::size_t pad_back_size_t = fplus::integral_cast_throw<std::size_t>(pad_back);
        std::size_t pad_top_size_t = fplus::integral_cast_throw<std::size_t>(pad_top);
        std::size_t pad_bottom_size_t = fplus::integral_cast_throw<std::size_t>(pad_bottom);
        std::size_t pad_left_size_t = fplus::integral_cast_throw<std::size_t>(pad_left);
        std::size_t pad_right_size_t = fplus::integral_cast_throw<std::size_t>(pad_right);

        return { pad_front_size_t, pad_back_size_t,
            pad_top_size_t, pad_bottom_size_t,
            pad_left_size_t, pad_right_size_t,
            out_size_d4_size_t, out_height_size_t, out_width_size_t };
    }

    struct convolution3d_filter_matrices {
        tensor_shape filter_shape_;
        std::size_t filter_count_;
        float_vec biases_;
        bool use_bias_;
        tensor filter_mats_;
    };

    inline tensor dilate_tensor_3d(const shape3& dilation_rate, const tensor& in)
    {
        if (dilation_rate == shape3(1, 1, 1)) {
            return in;
        }
        assertion(in.shape().rank() == 4, "Invalid rank for 3d dilation");

        const auto in_shape = in.shape();
        const tensor_shape dilated_shape(
            (in_shape.size_dim_4_ - 1) * dilation_rate.size_dim_4_ + 1,
            (in_shape.height_ - 1) * dilation_rate.height_ + 1,
            (in_shape.width_ - 1) * dilation_rate.width_ + 1,
            in_shape.depth_);
        tensor result(dilated_shape, static_cast<float_type>(0));
        for (std::size_t d4 = 0; d4 < in_shape.size_dim_4_; ++d4) {
            for (std::size_t y = 0; y < in_shape.height_; ++y) {
                for (std::size_t x = 0; x < in_shape.width_; ++x) {
                    for (std::size_t z = 0; z < in_shape.depth_; ++z) {
                        result.set_ignore_rank(tensor_pos(
                                                   d4 * dilation_rate.size_dim_4_,
                                                   y * dilation_rate.height_,
                                                   x * dilation_rate.width_,
                                                   z),
                            in.get_ignore_rank(tensor_pos(d4, y, x, z)));
                    }
                }
            }
        }
        return result;
    }

    inline filter dilate_filter_3d(const shape3& dilation_rate, const filter& undilated)
    {
        return filter(dilate_tensor_3d(dilation_rate, undilated.get_tensor()),
            undilated.get_bias());
    }

    inline filter_vec generate_filters_3d(
        const shape3& dilation_rate,
        const tensor_shape& filter_shape, std::size_t k,
        const float_vec& weights, const float_vec& bias)
    {
        filter_vec filters(k, filter(tensor(filter_shape, 0), 0));

        assertion(!filters.empty(), "at least one filter needed");
        const std::size_t param_count = fplus::sum(fplus::transform(
            fplus_c_mem_fn_t(filter, volume, std::size_t), filters));

        assertion(static_cast<std::size_t>(weights.size()) == param_count,
            "invalid weight size");
        const auto filter_param_cnt = filters.front().shape().volume();

        auto filter_weights = fplus::split_every(filter_param_cnt, weights);
        assertion(filter_weights.size() == filters.size(),
            "invalid size of filter weights");
        assertion(bias.size() == filters.size(), "invalid bias size");
        auto it_filter_val = std::begin(filter_weights);
        auto it_filter_bias = std::begin(bias);
        for (auto& filt : filters) {
            filt.set_params(*it_filter_val, *it_filter_bias);
            filt = dilate_filter_3d(dilation_rate, filt);
            ++it_filter_val;
            ++it_filter_bias;
        }

        return filters;
    }

    inline convolution3d_filter_matrices generate_im2col_filter_matrix_3d(
        const std::vector<filter>& filters)
    {
        assertion(fplus::all_the_same_on(
                      fplus_c_mem_fn_t(filter, shape, tensor_shape), filters),
            "all filters must have the same shape");

        const auto biases = fplus::transform_convert<float_vec>(
            fplus_c_mem_fn_t(filter, get_bias, float_type),
            filters);

        const bool use_bias = fplus::sum(biases) != static_cast<float_type>(0) || !fplus::all_the_same(biases);

        const auto shape = filters.front().shape();

        tensor filter_mats = tensor(
            tensor_shape(shape.size_dim_4_, shape.height_, shape.width_, shape.depth_, filters.size()),
            static_cast<float_type>(0));

        for (std::size_t d4 = 0; d4 < shape.size_dim_4_; ++d4) {
            for (std::size_t y = 0; y < shape.height_; ++y) {
                for (std::size_t n = 0; n < filters.size(); ++n) {
                    for (std::size_t x = 0; x < shape.width_; ++x) {
                        for (std::size_t z = 0; z < shape.depth_; ++z) {
                            filter_mats.set(tensor_pos(d4, y, x, z, n),
                                filters[n].get(tensor_pos(d4, y, x, z)));
                        }
                    }
                }
            }
        }

        return { shape, filters.size(), biases, use_bias, filter_mats };
    }

    inline tensor init_conv_output_tensor_3d(
        std::size_t out_size_d4,
        std::size_t out_height,
        std::size_t out_width,
        std::size_t out_depth,
        std::size_t rank,
        const convolution3d_filter_matrices& filter_mat)
    {
        tensor output(tensor_shape_with_changed_rank(
                          tensor_shape(out_size_d4, out_height, out_width, out_depth),
                          rank),
            static_cast<float_type>(0));
        if (filter_mat.use_bias_) {
            const auto bias_ptr = &filter_mat.biases_.front();
            const auto bias_ptr_end = bias_ptr + out_depth;
            for (std::size_t d4_out = 0; d4_out < out_size_d4; ++d4_out) {
                for (std::size_t y_out = 0; y_out < out_height; ++y_out) {
                    for (std::size_t x_out = 0; x_out < out_width; ++x_out) {
                        auto output_ptr = &output.get_ref_ignore_rank(tensor_pos(0, d4_out, y_out, x_out, 0));
                        std::copy(bias_ptr, bias_ptr_end, output_ptr);
                    }
                }
            }
        }
        return output;
    }

    inline Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned, Eigen::OuterStride<>> get_im2col_mapping_3d(
        const tensor& in,
        std::size_t f_width,
        std::size_t f_depth,
        std::size_t strides_x,
        std::size_t out_width,
        std::size_t d4,
        std::size_t y,
        std::size_t d4_filt,
        std::size_t y_filt)
    {
        // Same trick as in the 2D case: avoid materializing the im2col matrix
        // by using an outer stride smaller than the row count, so adjacent
        // columns share data along the receptive field.
        return Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned, Eigen::OuterStride<>>(
            const_cast<float_type*>(&in.get_ref_ignore_rank(tensor_pos(0, d4 + d4_filt, y + y_filt, 0, 0))),
            static_cast<EigenIndex>(f_width * f_depth),
            static_cast<EigenIndex>(out_width),
            Eigen::OuterStride<>(static_cast<EigenIndex>(f_depth * strides_x)));
    }

    inline tensor convolve_accumulative_3d(
        std::size_t out_size_d4,
        std::size_t out_height,
        std::size_t out_width,
        std::size_t strides_d4,
        std::size_t strides_y,
        std::size_t strides_x,
        const convolution3d_filter_matrices& filter_mat,
        const tensor& in)
    {
        const tensor& filter_mats = filter_mat.filter_mats_;
        const auto f_size_d4 = filter_mat.filter_shape_.size_dim_4_;
        const auto f_height = filter_mat.filter_shape_.height_;
        const auto f_width = filter_mat.filter_shape_.width_;
        const auto f_depth = filter_mat.filter_shape_.depth_;
        const auto out_depth = filter_mat.filter_count_;

        assertion(f_depth == in.shape().depth_, "filter depth does not match input");
        assertion(filter_mats.shape().size_dim_5_ == f_size_d4, "incorrect number of filter levels in d4 direction");
        assertion(filter_mats.shape().size_dim_4_ == f_height, "incorrect number of filter levels in y direction");
        assertion(out_width == (in.shape().width_ - f_width) / strides_x + 1, "output width does not match");
        assertion(out_depth == filter_mat.biases_.size(), "invalid bias count");

        tensor output = init_conv_output_tensor_3d(out_size_d4, out_height, out_width, out_depth, in.shape().rank(), filter_mat);

        for (std::size_t d4_filt = 0; d4_filt < f_size_d4; ++d4_filt) {
            for (std::size_t y_filt = 0; y_filt < f_height; ++y_filt) {
                const Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned>
                    filter(const_cast<float_type*>(&filter_mats.get_ref_ignore_rank(tensor_pos(d4_filt, y_filt, 0, 0, 0))),
                        static_cast<EigenIndex>(out_depth),
                        static_cast<EigenIndex>(f_width * f_depth));
                for (std::size_t d4 = 0, d4_out = 0; d4 < in.shape().size_dim_4_ + 1 - f_size_d4; d4 += strides_d4, ++d4_out) {
                    for (std::size_t y = 0, y_out = 0; y < in.shape().height_ + 1 - f_height; y += strides_y, ++y_out) {
                        const auto input = get_im2col_mapping_3d(in, f_width, f_depth, strides_x, out_width, d4, y, d4_filt, y_filt);
                        Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned>
                            output_map(&output.get_ref_ignore_rank(tensor_pos(0, d4_out, y_out, 0, 0)),
                                static_cast<EigenIndex>(out_depth),
                                static_cast<EigenIndex>(out_width));

                        output_map.noalias() += filter * input;
                    }
                }
            }
        }

        return output;
    }

    inline tensor convolve_3d(
        const shape3& strides,
        const padding& pad_type,
        const convolution3d_filter_matrices& filter_mat,
        const tensor& input)
    {
        assertion(filter_mat.filter_shape_.depth_ == input.shape().depth_,
            "invalid filter depth");

        const shape3 filter_spatial_shape(
            filter_mat.filter_shape_.size_dim_4_,
            filter_mat.filter_shape_.height_,
            filter_mat.filter_shape_.width_);

        const auto conv_cfg = preprocess_convolution_3d(
            filter_spatial_shape,
            strides, pad_type,
            input.shape().size_dim_4_,
            input.shape().height_,
            input.shape().width_);

        const auto in_padded = pad_tensor(0,
            conv_cfg.pad_front_, conv_cfg.pad_back_,
            conv_cfg.pad_top_, conv_cfg.pad_bottom_,
            conv_cfg.pad_left_, conv_cfg.pad_right_,
            input);

        return convolve_accumulative_3d(
            conv_cfg.out_size_d4_, conv_cfg.out_height_, conv_cfg.out_width_,
            strides.size_dim_4_, strides.height_, strides.width_,
            filter_mat,
            in_padded);
    }

}
}
