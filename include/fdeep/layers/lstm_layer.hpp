// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/recurrent_ops.hpp"

#include <string>
#include <functional>

namespace fdeep
{
namespace internal
{

class lstm_layer : public layer
{
  public:
    explicit lstm_layer(const std::string &name,
                        std::size_t n_units,
                        const std::string &activation,
                        const std::string &recurrent_activation,
                        const bool use_bias,
                        const bool return_sequences,
                        const RowMajorMatrixXf &W,
                        const RowMajorMatrixXf &U,
                        const RowMajorMatrixXf &bias)
        : layer(name),
          n_units_(n_units),
          activation_(activation),
          recurrent_activation_(recurrent_activation),
          use_bias_(use_bias),
          return_sequences_(return_sequences),
          W_(W),
          U_(U),
          bias_(bias)
    {
    }

  protected:
    tensor3s apply_impl(const tensor3s &inputs) const override final
    {
        const auto input_shapes = fplus::transform(fplus_c_mem_fn_t(tensor3, shape, shape_hwc), inputs);

        // ensure that tensor3 shape is (1, 1, n) when using multiple tensor3s elements as seq_len
        if (inputs.size() > 1)
        {
            assertion(inputs.front().shape().height_ == 1 && inputs.front().shape().width_ == 1,
                      "height and width dimension must be 1, but shape is '" + show_shape_hwcs(input_shapes) + "'");
        }
        else
        {
            assertion(inputs.front().shape().height_ == 1,
                      "width dimension must be 1, but shape is '" + show_shape_hwcs(input_shapes) + "'");
        }

        return lstm_impl(inputs, n_units_, use_bias_, return_sequences_, W_, U_, bias_, activation_, recurrent_activation_);
    }

    const std::size_t n_units_;
    const std::string activation_;
    const std::string recurrent_activation_;
    const bool use_bias_;
    const bool return_sequences_;
    const RowMajorMatrixXf W_;
    const RowMajorMatrixXf U_;
    const RowMajorMatrixXf bias_;
};

} // namespace internal
} // namespace fdeep
