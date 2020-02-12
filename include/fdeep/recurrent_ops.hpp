// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <string>
#include <functional>

namespace fdeep { namespace internal
{

using Eigen::Dynamic;

template<int Rows, int Cols>
using ColMajorMatrix = Eigen::Matrix<float_type, Rows, Cols, Eigen::ColMajor>;

template<int Rows, int Cols>
using RowMajorMatrix = Eigen::Matrix<float_type, Rows, Cols, Eigen::RowMajor>;

template<int Count>
using ColVector = Eigen::Matrix<float_type, Count, 1>;

template<int Count>
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

inline float_type hard_sigmoid_activation(float_type x)
{
    return static_cast<float_type>(std::min(1.0, std::max(0.0, (0.2 * x) + 0.5)));
}

inline float_type relu_activation(float_type x)
{
    return std::max<float_type>(x, 0);
}

inline float_type selu_activation(float_type x)
{
    const float_type alpha =
    static_cast<float_type>(1.6732632423543772848170429916717);
    const float_type scale =
    static_cast<float_type>(1.0507009873554804934193349852946);
    return scale * (x >= 0 ? x : alpha * (std::exp(x) - 1));
}

inline float_type elu_activation(float_type x)
{
    return x >= 0 ? x : std::exp(x) - 1;
}

inline std::function<float_type(float_type)> get_activation_func(const std::string& activation_func_name)
{
    if (activation_func_name == "linear")
        return linear_activation;
    else if (activation_func_name == "tanh")
        return tanh_activation;
    else if (activation_func_name == "sigmoid")
        return sigmoid_activation;
    else if (activation_func_name == "hard_sigmoid")
        return hard_sigmoid_activation;
    else if (activation_func_name == "relu")
        return relu_activation;
    else if (activation_func_name == "selu")
        return selu_activation;
    else if (activation_func_name == "elu")
        return elu_activation;

    raise_error("activation function '" + activation_func_name + "' not yet implemented");
    return {}; // Is never called
}

inline tensors lstm_impl(const tensor& input,
                          tensor& initial_state_h,
                          tensor& initial_state_c,
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
    const RowMajorMatrixXf W = eigen_row_major_mat_from_values(weights.size() / (n_units * 4), n_units * 4, weights);
    const RowMajorMatrixXf U = eigen_row_major_mat_from_values(n_units, n_units * 4, recurrent_weights);

    // initialize cell output states h, and cell memory states c for t-1 with initial state values
    RowMajorMatrixXf h(1, n_units);
    RowMajorMatrixXf c(1, n_units);
    h = eigen_row_major_mat_from_values(1, n_units, *initial_state_h.as_vector());
    c = eigen_row_major_mat_from_values(1, n_units, *initial_state_c.as_vector());

    std::size_t n_timesteps = input.shape().width_;
    std::size_t n_features = input.shape().depth_;

    RowMajorMatrixXf in(n_timesteps, n_features);

    // write input to eigen matrix

    for (std::size_t a_t = 0; a_t < n_timesteps; ++a_t)
        for (std::size_t a_f = 0; a_f < n_features; ++a_f)
            in(EigenIndex(a_t), EigenIndex(a_f)) = input.get(tensor_pos(a_t, a_f));

    RowMajorMatrixXf X = in * W;

    if (use_bias)
    {
        // define eigen vector type to be able to use broadcasting
        typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic> Vector_Xf;
        Vector_Xf b = eigen_row_major_mat_from_values(1, n_units * 4, bias);

        X.rowwise() += b;
    }

    // get activation functions
    auto act_func = get_activation_func(activation);
    auto act_func_recurrent = get_activation_func(recurrent_activation);

    // computing LSTM output
    const EigenIndex n = EigenIndex(n_units);

    tensors lstm_result;

    if (return_sequences)
        lstm_result = {tensor(tensor_shape(n_timesteps, n_units), float_type(0))};
    else
        lstm_result = {tensor(tensor_shape(n_units), float_type(0))};

    for (EigenIndex k = 0; k < EigenIndex(n_timesteps); ++k)
    {
        const RowMajorMatrixXf ifco = h * U;

        // Use of Matrix.block(): Block of size (p,q), starting at (i,j) matrix.block(i,j,p,q);  matrix.block<p,q>(i,j);
        const RowMajorMatrixXf i = (X.block(k, 0, 1, n) + ifco.block(0, 0, 1, n)).unaryExpr(act_func_recurrent);
        const RowMajorMatrixXf f = (X.block(k, n, 1, n) + ifco.block(0, n, 1, n)).unaryExpr(act_func_recurrent);
        const RowMajorMatrixXf c_pre = (X.block(k, n * 2, 1, n) + ifco.block(0, n * 2, 1, n)).unaryExpr(act_func);
        const RowMajorMatrixXf o = (X.block(k, n * 3, 1, n) + ifco.block(0, n * 3, 1, n)).unaryExpr(act_func_recurrent);

        c = f.array() * c.array() + i.array() * c_pre.array();
        h = o.array() * c.unaryExpr(act_func).array();

        if (return_sequences)
            for (EigenIndex idx = 0; idx < n; ++idx)
                lstm_result.front().set(tensor_pos(std::size_t(k), std::size_t(idx)), h(idx));
        else if (k == EigenIndex(n_timesteps) - 1)
            for (EigenIndex idx = 0; idx < n; ++idx)
                lstm_result.front().set(tensor_pos(std::size_t(idx)), h(idx));
        }

    if (return_state) {
        auto state_h = tensor(tensor_shape(n_units), float_type(0));
        auto state_c = tensor(tensor_shape(n_units), float_type(0));
        for (EigenIndex idx = 0; idx < n; ++idx)
            state_h.set(tensor_pos(std::size_t(idx)), h(idx));
        for (EigenIndex idx = 0; idx < n; ++idx)
            state_c.set(tensor_pos(std::size_t(idx)), c(idx));
        lstm_result.push_back(state_h);
        lstm_result.push_back(state_c);
    }
    // Copy the final state back into the initial state in the event of a stateful LSTM call
    initial_state_h = tensor(tensor_shape(n_units), eigen_row_major_mat_to_values(h));
    initial_state_c = tensor(tensor_shape(n_units), eigen_row_major_mat_to_values(c));
    return lstm_result;
}

inline tensors gru_impl(const tensor& input,
    tensor& initial_state_h,
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
    const std::size_t n_timesteps = input.shape().width_;
    const std::size_t n_features = input.shape().depth_;

    // weight matrices
    const EigenIndex n = EigenIndex(n_units);
    const RowMajorMatrix<Dynamic, Dynamic> W = eigen_row_major_mat_from_values(n_features, n_units * 3, weights);
    const RowMajorMatrix<Dynamic, Dynamic> U = eigen_row_major_mat_from_values(n_units, n_units * 3, recurrent_weights);

    // kernel bias
    RowVector<Dynamic> b_x(n_units * 3);
    if (use_bias && bias.size() >= 1 * n_units * 3)
        std::copy_n(bias.cbegin(), n_units * 3, b_x.data());
    else
        b_x.setZero();

    // recurrent kernel bias
    RowVector<Dynamic> b_h(n_units * 3);
    if (use_bias && bias.size() >= 2 * n_units * 3)
        std::copy_n(bias.cbegin() + static_cast<float_vec::const_iterator::difference_type>(n_units * 3), n_units * 3, b_h.data());
    else
        b_h.setZero();

    // initialize cell output states h
    // RowVector<Dynamic> h(1, n_units);
    RowMajorMatrixXf h(1, n_units);
    h = eigen_row_major_mat_from_values(1, n_units, *initial_state_h.as_vector());

    // write input to eigen matrix of shape (timesteps, n_features)
    RowMajorMatrix<Dynamic, Dynamic> x(n_timesteps, n_features);

    for (std::size_t a_t = 0; a_t < n_timesteps; ++a_t)
        for (std::size_t a_f = 0; a_f < n_features; ++a_f)
            x(EigenIndex(a_t), EigenIndex(a_f)) = input.get(tensor_pos(a_t, a_f));

    // kernel applied to inputs (with bias), produces shape (timesteps, n_units * 3)
    RowMajorMatrix<Dynamic, Dynamic> Wx = x * W;
    Wx.rowwise() += b_x;

    // get activation functions
    auto act_func = get_activation_func(activation);
    auto act_func_recurrent = get_activation_func(recurrent_activation);

    // computing GRU output
    tensors gru_result;

    if (return_sequences)
        gru_result = { tensor(tensor_shape(n_timesteps, n_units), float_type(0)) };
    else
        gru_result = { tensor(tensor_shape(n_units), float_type(0)) };

    for (EigenIndex k = 0; k < EigenIndex(n_timesteps); ++k)
    {
        RowVector<Dynamic> r;
        RowVector<Dynamic> z;
        RowVector<Dynamic> m;

        // in the formulae below, the following notations are used:
        // A b       matrix product
        // a o b     Hadamard (element-wise) product
        // x         input vector
        // h         state vector
        // W_{x,a}   block of the kernel weight matrix corresponding to "a"
        // W_{h,a}   block of the recurrent kernel weight matrix corresponding to "a"
        // b_{x,a}   part of the kernel bias vector corresponding to "a"
        // b_{h,a}   part of the recurrent kernel bias corresponding to "a"
        // z         update gate vector
        // r         reset gate vector

        if (reset_after)
        {
            // recurrent kernel applied to timestep (with bias), produces shape (1, n_units * 3)
            RowMajorMatrix<1, Dynamic> Uh = h * U;
            Uh += b_h;

            // z = sigmoid(W_{x,z} x + b_{i,z} + W_{h,z} h + b_{h,z})
            z = (Wx.block(k, 0 * n, 1, n) + Uh.block(0, 0 * n, 1, n)).unaryExpr(act_func_recurrent);
            // r = sigmoid(W_{x,r} x + b_{i,r} + W_{h,r} h + b_{h,r})
            r = (Wx.block(k, 1 * n, 1, n) + Uh.block(0, 1 * n, 1, n)).unaryExpr(act_func_recurrent);
            // m = tanh(W_{x,m} x + b_{i,m} + r * (W_{h,m} h + b_{h,m}))
            m = (Wx.block(k, 2 * n, 1, n) + (r.array() * Uh.block(0, 2 * n, 1, n).array()).matrix()).unaryExpr(act_func);
        }
        else
        {
            // z = sigmoid(W_{x,z} x + b_{x,z} + W_{h,z} h + b_{h,z})
            z = (Wx.block(k, 0 * n, 1, n) + h * U.block(0, 0 * n, n, n) + b_h.block(0, 0 * n, 1, n)).unaryExpr(act_func_recurrent);
            // r = sigmoid(W_{x,r} x + b_{x,r} + W_{h,r} h + b_{h,r})
            r = (Wx.block(k, 1 * n, 1, n) + h * U.block(0, 1 * n, n, n) + b_h.block(0, 1 * n, 1, n)).unaryExpr(act_func_recurrent);
            // m = tanh(W_{x,m} x + b_{x,m} + W_{h,m} (r o h) + b_{h,m}))
            m = (Wx.block(k, 2 * n, 1, n) + (r.array() * h.array()).matrix() * U.block(0, 2 * n, n, n) + b_h.block(0, 2 * n, 1, n)).unaryExpr(act_func);
        }

        // output vector: h' = (1 - z) o m + z o h
        h = ((1 - z.array()) * m.array() + z.array() * h.array()).matrix();

        if (return_sequences)
            for (EigenIndex idx = 0; idx < n; ++idx)
                gru_result.front().set(tensor_pos(std::size_t(k), std::size_t(idx)), h(idx));
        else if (k == EigenIndex(n_timesteps) - 1)
            for (EigenIndex idx = 0; idx < n; ++idx)
                gru_result.front().set(tensor_pos(std::size_t(idx)), h(idx));


        if (return_state) {
            auto state_h = tensor(tensor_shape(n_units), float_type(0));
            for (EigenIndex idx = 0; idx < n; ++idx)
                state_h.set(tensor_pos(std::size_t(idx)), h(idx));
            gru_result.push_back(state_h);
        }
        // Copy the final state back into the initial state in the event of a stateful LSTM call
        initial_state_h = tensor(tensor_shape(n_units), eigen_row_major_mat_to_values(h));
    }

    return gru_result;
}

inline tensor reverse_time_series_in_tensor(const tensor& ts)
{
    tensor reversed = tensor(ts.shape(), float_type(0.0));
    std::size_t n = 0;
    for (std::size_t x = ts.shape().width_; x--> 0;)
    {
        for (std::size_t z = 0; z < ts.shape().depth_; ++z)
            reversed.set_ignore_rank(tensor_pos(n, z),
                ts.get_ignore_rank(tensor_pos(x, z)));
        n++;
    }
    return reversed;
}

} } // namespace fdeep, namespace internal
