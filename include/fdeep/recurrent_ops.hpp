// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"
#include "fdeep/tensor.hpp"

#include <cmath>
#include <functional>
#include <string>

namespace fdeep {
namespace internal {

    using Eigen::Dynamic;

    template <int Count>
    using RowVector = Eigen::Matrix<float_type, 1, Count>;

    inline float_type linear_activation(float_type x)
    {
        return x;
    }

    inline float_type tanh_activation(float_type x)
    {
        return std::tanh(x);
    }

    inline float_type sigmoid_activation(float_type x)
    {
        return 1 / (1 + std::exp(-x));
    }

    inline float_type swish_activation(float_type x)
    {
        return x / (1 + std::exp(-x));
    }

    inline float_type relu_activation(float_type x)
    {
        return std::max<float_type>(x, 0);
    }

    inline float_type hard_sigmoid_activation(float_type x)
    {
        // https://github.com/keras-team/keras/blob/f7bc67e6c105c116a2ba7f5412137acf78174b1a/keras/ops/nn.py#L316C6-L316C74
        if (x < -3) {
            return 0;
        }
        if (x > 3) {
            return 1;
        }
        return (x / static_cast<float_type>(6)) + static_cast<float_type>(0.5);
    }

    inline float_type selu_activation(float_type x)
    {
        const float_type alpha = static_cast<float_type>(1.6732632423543772848170429916717);
        const float_type scale = static_cast<float_type>(1.0507009873554804934193349852946);
        return scale * (x >= 0 ? x : alpha * (std::exp(x) - 1));
    }

    inline float_type exponential_activation(float_type x)
    {
        return static_cast<float_type>(std::exp(x));
    }

    inline float_type gelu_activation(float_type x)
    {
        return static_cast<float_type>(0.5) * x * (static_cast<float_type>(1) + static_cast<float_type>(std::erf(x / std::sqrt(static_cast<float_type>(2)))));
    }

    inline float_type softsign_activation(float_type x)
    {
        return x / (std::abs(x) + static_cast<float_type>(1));
    }

    inline float_type elu_activation(float_type x)
    {
        return x >= 0 ? x : std::exp(x) - 1;
    }

    inline std::function<float_type(float_type)> get_activation_func(const std::string& activation_func_name)
    {
        if (activation_func_name == "linear")
            return linear_activation;
        if (activation_func_name == "tanh")
            return tanh_activation;
        if (activation_func_name == "sigmoid")
            return sigmoid_activation;
        if (activation_func_name == "swish" || activation_func_name == "silu")
            return swish_activation;
        if (activation_func_name == "hard_sigmoid")
            return hard_sigmoid_activation;
        if (activation_func_name == "relu")
            return relu_activation;
        if (activation_func_name == "selu")
            return selu_activation;
        if (activation_func_name == "elu")
            return elu_activation;
        if (activation_func_name == "exponential")
            return exponential_activation;
        if (activation_func_name == "gelu")
            return gelu_activation;
        if (activation_func_name == "softsign")
            return softsign_activation;

        raise_error("recurrent activation function '" + activation_func_name + "' not yet implemented");
        return {};
    }

    inline tensors lstm_impl(const tensor& input,
        const std::size_t n_units,
        const bool use_bias,
        const bool return_sequences,
        const bool return_state,
        const float_vec& weights,
        const float_vec& recurrent_weights,
        const float_vec& bias,
        const std::string& activation,
        const std::string& recurrent_activation)
    {
        assertion(n_units > 0, "LSTM units must be > 0.");
        const MappedRowMajorMatrixXf W = eigen_row_major_mat_from_shared_values(
            weights.size() / (n_units * 4), n_units * 4,
            weights.data());
        const MappedRowMajorMatrixXf U = eigen_row_major_mat_from_shared_values(
            n_units, n_units * 4, recurrent_weights.data());

        RowMajorMatrixXf h = RowMajorMatrixXf::Zero(1, static_cast<EigenIndex>(n_units));
        RowMajorMatrixXf c = RowMajorMatrixXf::Zero(1, static_cast<EigenIndex>(n_units));

        const std::size_t n_timesteps = input.shape().width_;
        const std::size_t n_features = input.shape().depth_;

        const MappedRowMajorMatrixXf in = eigen_row_major_mat_from_shared_values(
            n_timesteps, n_features, input.as_vector()->data());

        RowMajorMatrixXf X = in * W;

        if (use_bias) {
            typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic> Vector_Xf;
            const Vector_Xf b = eigen_row_major_mat_from_shared_values(
                1, n_units * 4, bias.data());
            X.rowwise() += b;
        }

        const auto act_func = get_activation_func(activation);
        const auto act_func_recurrent = get_activation_func(recurrent_activation);

        const EigenIndex n = static_cast<EigenIndex>(n_units);

        tensors result;
        if (return_sequences)
            result = { tensor(tensor_shape(n_timesteps, n_units), float_type(0)) };
        else
            result = { tensor(tensor_shape(n_units), float_type(0)) };

        for (EigenIndex k = 0; k < static_cast<EigenIndex>(n_timesteps); ++k) {
            const RowMajorMatrixXf ifco = h * U;

            const RowMajorMatrixXf i = (X.block(k, 0, 1, n) + ifco.block(0, 0, 1, n)).unaryExpr(act_func_recurrent);
            const RowMajorMatrixXf f = (X.block(k, n, 1, n) + ifco.block(0, n, 1, n)).unaryExpr(act_func_recurrent);
            const RowMajorMatrixXf c_pre = (X.block(k, n * 2, 1, n) + ifco.block(0, n * 2, 1, n)).unaryExpr(act_func);
            const RowMajorMatrixXf o = (X.block(k, n * 3, 1, n) + ifco.block(0, n * 3, 1, n)).unaryExpr(act_func_recurrent);

            c = f.array() * c.array() + i.array() * c_pre.array();
            h = o.array() * c.unaryExpr(act_func).array();

            if (return_sequences)
                for (EigenIndex idx = 0; idx < n; ++idx)
                    result.front().set_ignore_rank(tensor_pos(std::size_t(k), std::size_t(idx)), h(idx));
            else if (k == static_cast<EigenIndex>(n_timesteps) - 1)
                for (EigenIndex idx = 0; idx < n; ++idx)
                    result.front().set_ignore_rank(tensor_pos(std::size_t(idx)), h(idx));
        }

        if (return_state) {
            auto state_h = tensor(tensor_shape(n_units), float_type(0));
            auto state_c = tensor(tensor_shape(n_units), float_type(0));
            for (EigenIndex idx = 0; idx < n; ++idx)
                state_h.set_ignore_rank(tensor_pos(std::size_t(idx)), h(idx));
            for (EigenIndex idx = 0; idx < n; ++idx)
                state_c.set_ignore_rank(tensor_pos(std::size_t(idx)), c(idx));
            result.push_back(state_h);
            result.push_back(state_c);
        }

        return result;
    }

    inline tensors gru_impl(const tensor& input,
        const std::size_t n_units,
        const bool use_bias,
        const bool reset_after,
        const bool return_sequences,
        const bool return_state,
        const float_vec& weights,
        const float_vec& recurrent_weights,
        const float_vec& bias,
        const std::string& activation,
        const std::string& recurrent_activation)
    {
        assertion(n_units > 0, "GRU units must be > 0.");
        const std::size_t n_timesteps = input.shape().width_;
        const std::size_t n_features = input.shape().depth_;

        const EigenIndex n = static_cast<EigenIndex>(n_units);
        const MappedRowMajorMatrixXf W = eigen_row_major_mat_from_shared_values(
            n_features, n_units * 3, weights.data());
        const MappedRowMajorMatrixXf U = eigen_row_major_mat_from_shared_values(
            n_units, n_units * 3, recurrent_weights.data());

        // Keras GRU bias layout:
        //   reset_after=False, use_bias=True  -> shape (3*units,)
        //   reset_after=True,  use_bias=True  -> shape (2, 3*units)
        if (use_bias) {
            const std::size_t expected = reset_after ? 2 * n_units * 3 : n_units * 3;
            assertion(bias.size() == expected,
                "GRU bias size does not match reset_after setting.");
        }
        RowVector<Dynamic> b_x(static_cast<EigenIndex>(n_units * 3));
        if (use_bias)
            std::copy_n(bias.cbegin(), n_units * 3, b_x.data());
        else
            b_x.setZero();

        RowVector<Dynamic> b_h(static_cast<EigenIndex>(n_units * 3));
        if (use_bias && reset_after)
            std::copy_n(bias.cbegin() + static_cast<std::ptrdiff_t>(n_units * 3),
                n_units * 3, b_h.data());
        else
            b_h.setZero();

        RowMajorMatrixXf h = RowMajorMatrixXf::Zero(1, n);

        const MappedRowMajorMatrixXf x = eigen_row_major_mat_from_shared_values(
            n_timesteps, n_features, input.as_vector()->data());

        RowMajorMatrixXf Wx = x * W;
        Wx.rowwise() += b_x;

        const auto act_func = get_activation_func(activation);
        const auto act_func_recurrent = get_activation_func(recurrent_activation);

        tensors result;
        if (return_sequences)
            result = { tensor(tensor_shape(n_timesteps, n_units), float_type(0)) };
        else
            result = { tensor(tensor_shape(n_units), float_type(0)) };

        for (EigenIndex k = 0; k < static_cast<EigenIndex>(n_timesteps); ++k) {
            RowVector<Dynamic> r;
            RowVector<Dynamic> z;
            RowVector<Dynamic> m;

            if (reset_after) {
                RowMajorMatrixXf Uh = h * U;
                Uh += b_h;

                z = (Wx.block(k, 0 * n, 1, n) + Uh.block(0, 0 * n, 1, n)).unaryExpr(act_func_recurrent);
                r = (Wx.block(k, 1 * n, 1, n) + Uh.block(0, 1 * n, 1, n)).unaryExpr(act_func_recurrent);
                m = (Wx.block(k, 2 * n, 1, n) + (r.array() * Uh.block(0, 2 * n, 1, n).array()).matrix()).unaryExpr(act_func);
            } else {
                z = (Wx.block(k, 0 * n, 1, n) + h * U.block(0, 0 * n, n, n) + b_h.block(0, 0 * n, 1, n)).unaryExpr(act_func_recurrent);
                r = (Wx.block(k, 1 * n, 1, n) + h * U.block(0, 1 * n, n, n) + b_h.block(0, 1 * n, 1, n)).unaryExpr(act_func_recurrent);
                m = (Wx.block(k, 2 * n, 1, n) + (r.array() * h.array()).matrix() * U.block(0, 2 * n, n, n) + b_h.block(0, 2 * n, 1, n)).unaryExpr(act_func);
            }

            h = ((1 - z.array()) * m.array() + z.array() * h.array()).matrix();

            if (return_sequences)
                for (EigenIndex idx = 0; idx < n; ++idx)
                    result.front().set_ignore_rank(tensor_pos(std::size_t(k), std::size_t(idx)), h(idx));
            else if (k == static_cast<EigenIndex>(n_timesteps) - 1)
                for (EigenIndex idx = 0; idx < n; ++idx)
                    result.front().set_ignore_rank(tensor_pos(std::size_t(idx)), h(idx));
        }

        if (return_state) {
            auto state_h = tensor(tensor_shape(n_units), float_type(0));
            for (EigenIndex idx = 0; idx < n; ++idx)
                state_h.set_ignore_rank(tensor_pos(std::size_t(idx)), h(idx));
            result.push_back(state_h);
        }

        return result;
    }

    inline tensors simple_rnn_impl(const tensor& input,
        const std::size_t n_units,
        const bool use_bias,
        const bool return_sequences,
        const bool return_state,
        const float_vec& weights,
        const float_vec& recurrent_weights,
        const float_vec& bias,
        const std::string& activation)
    {
        assertion(n_units > 0, "SimpleRNN units must be > 0.");
        const std::size_t n_timesteps = input.shape().width_;
        const std::size_t n_features = input.shape().depth_;

        const MappedRowMajorMatrixXf W = eigen_row_major_mat_from_shared_values(
            n_features, n_units, weights.data());
        const MappedRowMajorMatrixXf U = eigen_row_major_mat_from_shared_values(
            n_units, n_units, recurrent_weights.data());

        const MappedRowMajorMatrixXf in = eigen_row_major_mat_from_shared_values(
            n_timesteps, n_features, input.as_vector()->data());

        RowMajorMatrixXf X = in * W;
        if (use_bias) {
            typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic> Vector_Xf;
            const Vector_Xf b = eigen_row_major_mat_from_shared_values(
                1, n_units, bias.data());
            X.rowwise() += b;
        }

        const auto act_func = get_activation_func(activation);

        const EigenIndex n = static_cast<EigenIndex>(n_units);
        RowMajorMatrixXf h = RowMajorMatrixXf::Zero(1, n);

        tensors result;
        if (return_sequences)
            result = { tensor(tensor_shape(n_timesteps, n_units), float_type(0)) };
        else
            result = { tensor(tensor_shape(n_units), float_type(0)) };

        for (EigenIndex k = 0; k < static_cast<EigenIndex>(n_timesteps); ++k) {
            h = (X.block(k, 0, 1, n) + h * U).unaryExpr(act_func);

            if (return_sequences)
                for (EigenIndex idx = 0; idx < n; ++idx)
                    result.front().set_ignore_rank(tensor_pos(std::size_t(k), std::size_t(idx)), h(idx));
            else if (k == static_cast<EigenIndex>(n_timesteps) - 1)
                for (EigenIndex idx = 0; idx < n; ++idx)
                    result.front().set_ignore_rank(tensor_pos(std::size_t(idx)), h(idx));
        }

        if (return_state) {
            auto state_h = tensor(tensor_shape(n_units), float_type(0));
            for (EigenIndex idx = 0; idx < n; ++idx)
                state_h.set_ignore_rank(tensor_pos(std::size_t(idx)), h(idx));
            result.push_back(state_h);
        }

        return result;
    }

    inline tensor reverse_time_series_in_tensor(const tensor& ts)
    {
        tensor reversed = tensor(ts.shape(), float_type(0.0));
        std::size_t n = 0;
        for (std::size_t x = ts.shape().width_; x-- > 0;) {
            for (std::size_t z = 0; z < ts.shape().depth_; ++z)
                reversed.set_ignore_rank(tensor_pos(n, z),
                    ts.get_ignore_rank(tensor_pos(x, z)));
            n++;
        }
        return reversed;
    }

}
}
