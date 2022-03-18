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

    tensor output(tensor_shape_with_changed_rank(
            tensor_shape(out_height, out_width, out_depth),
            in.shape().rank()),
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

    using MappedColMajorMatrixXfUnaligned = Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned>;
    using MappedColMajorMatrixXfUnalignedOuterStride = Eigen::Map<ColMajorMatrixXf, Eigen::Unaligned, Eigen::OuterStride<>>;
    
    for (std::size_t y_filt = 0; y_filt < f_height; ++y_filt)
    {
        const MappedColMajorMatrixXfUnalignedOuterStride
            filter(const_cast<float_type*>(&filter_mats.get_ref_ignore_rank(tensor_pos(0, y_filt, 0, 0, 0))),
                static_cast<EigenIndex>(filters_count),
                static_cast<EigenIndex>(f_width * filters_count),
                Eigen::OuterStride<>(static_cast<EigenIndex>(0)));

        for (std::size_t y = 0, y_out = 0; y < in.shape().height_ + 1 - f_height; y += strides_y, ++y_out)
        {
            const MappedColMajorMatrixXfUnalignedOuterStride
                input(const_cast<float_type*>(&in.get_ref_ignore_rank(tensor_pos(0, 0, y + y_filt, 0, 0))),
                    static_cast<EigenIndex>(f_width * filters_count),
                    static_cast<EigenIndex>(out_width),
                    Eigen::OuterStride<>(static_cast<EigenIndex>(filters_count * strides_x)));

            MappedColMajorMatrixXfUnaligned
                output_map(&output.get_ref_ignore_rank(tensor_pos(0, 0, y_out, 0, 0)),
                    static_cast<EigenIndex>(out_depth),
                    static_cast<EigenIndex>(out_width));

            // todo: remove
            /*
            const auto input_rows = input.rows();
            const auto input_cols = input.cols();
            const auto filter_rows = filter.rows();
            const auto filter_cols = filter.cols();
            const auto output_rows = output_map.rows();
            const auto output_cols = output_map.cols();
            std::cout << (input_rows == (input_cols == (filter_rows == (filter_cols == (output_rows == output_cols))))) << std::endl;
            */

            output_map.noalias() += filter * input;
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

    const std::size_t out_height = conv_cfg.out_height_;
    const std::size_t out_width = conv_cfg.out_width_;

    const auto in_padded = pad_tensor(0,
        conv_cfg.pad_top_, conv_cfg.pad_bottom_, conv_cfg.pad_left_, conv_cfg.pad_right_,
        input);

    return depthwise_convolve_accumulative(
        out_height, out_width,
        strides.height_, strides.width_,
        filter_mat,
        in_padded);
}

} } // namespace fdeep, namespace internal
