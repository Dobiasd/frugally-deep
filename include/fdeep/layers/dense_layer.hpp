// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include "fdeep/tensor2.hpp"

#include <fplus/fplus.hpp>

#include <string>

namespace fdeep { namespace internal
{

// Takes a single stack volume (shape_hwc(1, 1, n)) as input.
class dense_layer : public layer
{
public:
    static RowMajorMatrixXf generate_params(std::size_t n_in,
        const float_vec& weights, const float_vec& bias)
    {
        assertion(weights.size() % bias.size() == 0, "invalid params");
        return eigen_row_major_mat_from_values(n_in + 1, bias.size(),
            fplus::append(weights, bias));
    }
    dense_layer(const std::string& name, std::size_t units,
            const float_vec& weights,
            const float_vec& bias) :
        layer(name),
        n_in_(weights.size() / bias.size()),
        n_out_(units),
        params_(generate_params(n_in_, weights, bias))
    {
        assertion(bias.size() == units, "invalid bias count");
        assertion(weights.size() % units == 0, "invalid weight count");
    }
protected:
    tensor3s apply_impl(const tensor3s& inputs) const override
    {
        assertion(inputs.size() == 1, "invalid number of input tensors");
        auto input = inputs.front();
        assertion(input.shape().width_ == 1 && input.shape().height_ == 1,
            "input not flattened");
        const auto bias_padded_input = bias_pad_input(input);
        const auto result = bias_padded_input * params_;
        assertion(result.rows() == 1, "invalid result size");
        return {tensor3(shape_hwc(1, 1, static_cast<std::size_t>(result.cols())),
            eigen_row_major_mat_to_values(result))};
    }
    static RowMajorMatrixXf bias_pad_input(const tensor3& input)
    {
        assertion(input.shape().width_ == 1 && input.shape().height_ == 1,
            "tensor not flattened");
        RowMajorMatrixXf m(1, input.shape().depth_ + 1);
        for (std::size_t z = 0; z < input.shape().depth_; ++z)
        {
            m(0, static_cast<EigenIndex>(z)) = input.get(tensor3_pos_yxz(0, 0, z));
        }
        m(0, static_cast<EigenIndex>(input.shape().depth_)) = 1;
        return m;
    }
    std::size_t n_in_;
    std::size_t n_out_;
    RowMajorMatrixXf params_;
};

} } // namespace fdeep, namespace internal
