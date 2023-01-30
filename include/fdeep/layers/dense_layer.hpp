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
#include <utility>
#include <vector>

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
    tensors apply_impl(const tensors& inputs) const override {
        assertion(inputs.size() == 1, "invalid number of input tensors");
        auto input = inputs.front();
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

        const auto feature_arr = input.as_vector();
        const size_t size = feature_arr->size();
        const size_t depth = input.shape().depth_;
        assertion(depth == n_in_ && (size % depth) == 0, "Invalid input value count.");
        std::vector<float_type> result_values((input.shape().volume() / depth) * n_out_);
        const size_t n_of_parts = size / depth;

        Eigen::Map<const RowMajorMatrixXf, Eigen::Unaligned> params(
            params_.data(),
            static_cast<EigenIndex>(params_.rows() - 1),
            static_cast<EigenIndex>(params_.cols()));
        Eigen::Map<const RowMajorMatrixXf, Eigen::Unaligned> bias(
            params_.data() + (params_.rows() - 1) * params_.cols(),
            static_cast<EigenIndex>(1),
            static_cast<EigenIndex>(params_.cols()));

        for (size_t part_id = 0; part_id < n_of_parts; ++part_id) {
            Eigen::Map<const RowMajorMatrixXf, Eigen::Unaligned> m(
                &(*feature_arr)[part_id * depth],
                static_cast<EigenIndex>(1),
                static_cast<EigenIndex>(depth));
            Eigen::Map<RowMajorMatrixXf, Eigen::Unaligned> res_m(
                &result_values[part_id * n_out_],
                static_cast<EigenIndex>(1),
                static_cast<EigenIndex>(n_out_));
            res_m.noalias() = m * params + bias;
        }
        return {tensor(tensor_shape_with_changed_rank(
                tensor_shape(
                    input.shape().size_dim_5_,
                    input.shape().size_dim_4_,
                    input.shape().height_,
                    input.shape().width_,
                    n_out_),
                input.shape().rank()),
            std::move(result_values))};
    }

    std::size_t n_in_;
    std::size_t n_out_;
    RowMajorMatrixXf params_;
};

} } // namespace fdeep, namespace internal