// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"
#include "fdeep/tensor.hpp"

#include <fplus/fplus.hpp>

#include <string>

namespace fdeep { namespace internal
{

// Takes a single stack volume (tensor_shape(n)) as input.
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
    tensors apply_impl(const tensors& inputs) const override
    {
        const auto& input = single_tensor_from_tensors(inputs);
        // According to the Keras documentation
        // https://keras.io/layers/core/#dense
        // "if the input to the layer has a rank greater than 2,
        // then it is flattened prior to the initial dot product with kernel."
        // But this seems to not be the case.
        // Instead it does this: https://stackoverflow.com/a/43237727/1866775
        // Otherwise the following would need to be done:
        // if (input.shape().get_not_one_dimension_count() > 1)
        // {
        //     input = flatten_tensor(input);
        // }
        const auto input_parts = fplus::split_every(
            input.shape().depth_, *input.as_vector());

        const auto result_value_vectors = fplus::transform(
            [this](const auto& input_part) -> float_vec
                        {
                assertion(input_part.size() == n_in_,
                    "Invalid input value count.");
                const auto bias_padded_input = bias_pad_input(input_part);
                const auto result = bias_padded_input * params_;
                assertion(result.rows() == 1, "invalid result size.");
                return *eigen_row_major_mat_to_values(result);
            },
            input_parts);

        const auto result_values = fplus::concat(result_value_vectors);
        assertion(result_values.size() % n_out_ == 0,
            "Invalid number of output values.");

        return {tensor(change_tensor_shape_dimension_by_index(
                input.shape(), 4, n_out_),
            fplus::make_shared_ref<fdeep::float_vec>(result_values))};
    }
    static RowMajorMatrixXf bias_pad_input(const float_vec& input)
    {
        RowMajorMatrixXf m(1, input.size() + 1);
        for (std::size_t z = 0; z < input.size(); ++z)
        {
            m(0, static_cast<EigenIndex>(z)) = input[z];
        }
        m(0, static_cast<EigenIndex>(input.size())) = 1;
        return m;
    }
    std::size_t n_in_;
    std::size_t n_out_;
    RowMajorMatrixXf params_;
};

} } // namespace fdeep, namespace internal
