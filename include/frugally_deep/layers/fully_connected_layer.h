// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/layers/layer.h"

#include "frugally_deep/matrix2d.h"

#include <fplus/fplus.h>

namespace fd
{

// Takes a single column volume (size3d(1, n, 1)) as input.
class fully_connected_layer : public layer
{
public:
    fully_connected_layer(std::size_t n_in, std::size_t n_out) :
        layer(size3d(1, n_in, 1), size3d(1, n_out, 1)),
        params_(size2d(n_out, n_in + 1))
    {
    }
    std::size_t param_count() const override
    {
        return params_.size().area();
    }
    float_vec get_params() const override
    {
        return params_.as_vector();
    }
    void set_params(const float_vec_const_it ps_begin,
        const float_vec_const_it ps_end) override
    {
        assert(static_cast<std::size_t>(std::distance(ps_begin, ps_end)) ==
            param_count());
        params_.overwrite_values(ps_begin, ps_end);
    }
protected:
    static matrix2d bias_pad_input(const matrix3d& input)
    {
        return matrix2d(
            size2d(input.size().height_ + 1, 1),
            fplus::append(input.as_vector(), {1}));
    }
    matrix3d forward_pass_impl(const matrix3d& input) const override
    {
        const auto input_slice_with_bias_neuron = bias_pad_input(input);
        return matrix2d_to_matrix3d(
            multiply(params_, input_slice_with_bias_neuron));
    }

    matrix3d backward_pass_impl(const matrix3d& input,
        float_vec& params_deltas_acc_reversed) const override
    {
        auto output_temp = matrix2d_to_matrix3d(
            multiply(transpose(params_), depth_slice(0, input)));
        auto output = matrix3d(input_size(),
            fplus::init(output_temp.as_vector()));
        matrix2d param_deltas(params_.size());
        const auto last_input_slice_with_bias_neuron =
            bias_pad_input(last_input_);
        assert(param_deltas.size().height_ == input.size().height_);
        assert(param_deltas.size().width_ ==
            last_input_slice_with_bias_neuron.size().height_);
        for (std::size_t y = 0; y < param_deltas.size().height_; ++y)
        {
            for (std::size_t x = 0; x < param_deltas.size().width_; ++x)
            {
                param_deltas.set(y, x,
                    input.get(0, y, 0) *
                    last_input_slice_with_bias_neuron.get(x, 0));
            }
        }
        const auto& param_deltas_vec = param_deltas.as_vector();

        params_deltas_acc_reversed.insert(std::end(params_deltas_acc_reversed),
            param_deltas_vec.rbegin(), param_deltas_vec.rend());
        return output;
    }

    matrix2d params_;
};

} // namespace fd
