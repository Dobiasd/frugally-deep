// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"
#include "fdeep/filter.hpp"
#include "fdeep/convolution.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace fdeep { namespace internal
{

inline tensor depthwise_convolve_accumulative(
    std::size_t out_height,
    std::size_t out_width,
    std::size_t strides_y,
    std::size_t strides_x,
    const convolution_filter_matrices& filter_mat,
    const tensor& in)
{
    const tensor& filter_mats = filter_mat.filter_mats_;
    const auto f_height = filter_mat.filter_shape_.height_;
    const auto f_width = filter_mat.filter_shape_.width_;
    const auto filters_count = filter_mat.filter_count_;
    const auto out_depth = filter_mat.filter_count_;

    assertion(filter_mat.filter_shape_.depth_ == 1, "filter depth must be 1");
    assertion(filters_count == in.shape().depth_, "filter count must match input depth");
    assertion(out_depth == in.shape().depth_, "number of filters does not match input depth");
    assertion(filter_mats.shape().size_dim_4_ == f_height, "incorrect number of filter levels in y direction");
    assertion(out_width == (in.shape().width_ - f_width) / strides_x + 1, "output width does not match");
    assertion(out_depth == filter_mat.biases_.size(), "invlid bias count");

    tensor output = init_conv_output_tensor(out_height, out_width, out_depth, in.shape().rank(), filter_mat);

    for (std::size_t y_filt = 0; y_filt < f_height; ++y_filt)
    {
        const auto filter = Eigen::Map<ArrayXf1D, Eigen::Unaligned>(
            const_cast<float_type*>(&filter_mats.get_ref_ignore_rank(tensor_pos(0, y_filt, 0, 0, 0))),
                static_cast<EigenIndex>(f_width * filters_count));

        for (std::size_t y = 0, y_out = 0; y < in.shape().height_ + 1 - f_height; y += strides_y, ++y_out)
        {
            const auto input = get_im2col_mapping(in, f_width, filters_count, strides_x, out_width, y, y_filt).array();

            // Not materialized to save memory and improve performance.
            const auto coefficient_wise_product = input.colwise() * filter;

            Eigen::Map<ArrayXf, Eigen::Unaligned>
                output_map(&output.get_ref_ignore_rank(tensor_pos(0, 0, y_out, 0, 0)),
                    static_cast<EigenIndex>(out_depth),
                    static_cast<EigenIndex>(out_width));

            for (EigenIndex x = 0; x < static_cast<EigenIndex>(out_width); ++x) {
                const ArrayXf col_materialized = coefficient_wise_product.col(x);
                const Eigen::Map<ArrayXf, Eigen::Unaligned>
                    cwp_reshaped(const_cast<float_type*>(col_materialized.data()),
                        static_cast<EigenIndex>(filters_count),
                        static_cast<EigenIndex>(f_width));
                output_map.col(x) += cwp_reshaped.rowwise().sum();
            }
        }
    }

    return output;
}


inline tensor depthwise_convolve(
    const shape2& strides,
    const padding& pad_type,
    const convolution_filter_matrices& filter_mat,
    const tensor& input)
{
    assertion(filter_mat.filter_shape_.depth_ == 1,
        "invalid filter depth");

    assertion(filter_mat.filter_count_ == input.shape().depth_,
        "invalid filter count");

    const auto conv_cfg = preprocess_convolution(
        filter_mat.filter_shape_.without_depth(),
        strides, pad_type, input.shape().height_, input.shape().width_);

    const auto in_padded = pad_tensor(0, 0, 0,
        conv_cfg.pad_top_, conv_cfg.pad_bottom_, conv_cfg.pad_left_, conv_cfg.pad_right_,
        input);

    return depthwise_convolve_accumulative(
        conv_cfg.out_height_, conv_cfg.out_width_,
        strides.height_, strides.width_,
        filter_mat,
        in_padded);
}

} } // namespace fdeep, namespace internal
