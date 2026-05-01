// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/conv_3d_layer.hpp"
#include "fdeep/layers/layer.hpp"
#include "fdeep/recurrent_ops.hpp"

#include <cstring>
#include <string>

namespace fdeep {
namespace internal {

    class conv_lstm_3d_layer : public layer {
    public:
        explicit conv_lstm_3d_layer(const std::string& name,
            std::size_t units,
            const tensor_shape& filter_shape,
            const shape3& strides, padding pad_type, const shape3& dilation_rate,
            const float_vec& weights, const float_vec& recurrent_weights,
            const float_vec& bias,
            const std::string& activation, const std::string& recurrent_activation,
            bool return_sequences, bool return_state)
            : layer(name)
            , units_(units)
            , return_sequences_(return_sequences)
            , return_state_(return_state)
            , activation_(activation)
            , recurrent_activation_(recurrent_activation)
            , input_conv_(name + "_input_conv", filter_shape, units * 4,
                  strides, pad_type, dilation_rate, weights, bias)
            , recurrent_conv_(name + "_recurrent_conv",
                  tensor_shape(filter_shape.size_dim_4_,
                      filter_shape.height_, filter_shape.width_, units),
                  units * 4, shape3(1, 1, 1), padding::same, shape3(1, 1, 1),
                  recurrent_weights, float_vec(units * 4, 0))
        {
        }

    protected:
        std::size_t units_;
        bool return_sequences_;
        bool return_state_;
        std::string activation_;
        std::string recurrent_activation_;
        conv_3d_layer input_conv_;
        conv_3d_layer recurrent_conv_;

        static tensor extract_timestep(const tensor& input, std::size_t t)
        {
            const auto& sh = input.shape();
            const std::size_t D4 = sh.size_dim_4_;
            const std::size_t H = sh.height_;
            const std::size_t W = sh.width_;
            const std::size_t C = sh.depth_;
            const std::size_t step_size = D4 * H * W * C;
            const auto& src = *input.as_vector();
            float_vec slice_data(step_size);
            std::memcpy(slice_data.data(), src.data() + t * step_size,
                step_size * sizeof(float_type));
            return tensor(tensor_shape(D4, H, W, C), std::move(slice_data));
        }

        tensors apply_impl(const tensors& inputs) const override
        {
            const auto& input = single_tensor_from_tensors(inputs);
            const auto& sh = input.shape();
            const std::size_t T = sh.size_dim_5_;

            const auto act_func = get_activation_func(activation_);
            const auto rec_act_func = get_activation_func(recurrent_activation_);

            tensor X = input_conv_.apply({ extract_timestep(input, 0) }).front();
            const std::size_t D4_out = X.shape().size_dim_4_;
            const std::size_t H_out = X.shape().height_;
            const std::size_t W_out = X.shape().width_;
            const std::size_t spatial = D4_out * H_out * W_out;
            const std::size_t depth4 = units_ * 4;

            float_vec h_buf(spatial * units_, 0);
            float_vec c_buf(spatial * units_, 0);

            const auto step = [&](const tensor& X_t) {
                const tensor h_tensor(tensor_shape(D4_out, H_out, W_out, units_),
                    float_vec(h_buf));
                const tensor U_tensor = recurrent_conv_.apply({ h_tensor }).front();
                const auto& Xv = *X_t.as_vector();
                const auto& Uv = *U_tensor.as_vector();
                for (std::size_t s = 0; s < spatial; ++s) {
                    for (std::size_t k = 0; k < units_; ++k) {
                        const std::size_t base = s * depth4;
                        const float_type i_val = rec_act_func(Xv[base + k] + Uv[base + k]);
                        const float_type f_val = rec_act_func(Xv[base + units_ + k] + Uv[base + units_ + k]);
                        const float_type c_pre = act_func(Xv[base + 2 * units_ + k] + Uv[base + 2 * units_ + k]);
                        const float_type o_val = rec_act_func(Xv[base + 3 * units_ + k] + Uv[base + 3 * units_ + k]);
                        const std::size_t hpos = s * units_ + k;
                        c_buf[hpos] = f_val * c_buf[hpos] + i_val * c_pre;
                        h_buf[hpos] = o_val * act_func(c_buf[hpos]);
                    }
                }
            };

            float_vec all_h;
            if (return_sequences_)
                all_h.reserve(T * spatial * units_);

            for (std::size_t t = 0; t < T; ++t) {
                if (t > 0)
                    X = input_conv_.apply({ extract_timestep(input, t) }).front();
                step(X);
                if (return_sequences_) {
                    const std::size_t off = all_h.size();
                    all_h.resize(off + spatial * units_);
                    std::memcpy(all_h.data() + off, h_buf.data(),
                        spatial * units_ * sizeof(float_type));
                }
            }

            tensors result;
            if (return_sequences_) {
                result.emplace_back(
                    tensor_shape(T, D4_out, H_out, W_out, units_),
                    std::move(all_h));
            } else {
                result.emplace_back(tensor_shape(D4_out, H_out, W_out, units_),
                    float_vec(h_buf));
            }

            if (return_state_) {
                result.emplace_back(tensor_shape(D4_out, H_out, W_out, units_),
                    float_vec(h_buf));
                result.emplace_back(tensor_shape(D4_out, H_out, W_out, units_),
                    std::move(c_buf));
            }

            return result;
        }
    };

}
}
