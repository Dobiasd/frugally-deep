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

float_type linear_activation(float_type x)
{
    return x;
}

float_type tanh_activation(float_type x)
{
    return std::tanh(x);
}

float_type sigmoid_activation(float_type x)
{
    return 1 / (1 + std::exp(-x));
}

float_type hard_sigmoid_activation(float_type x)
{
    return static_cast<float_type>(std::min(1.0, std::max(0.0, (0.2 * x) + 0.5)));
}

float_type relu_activation(float_type x)
{
    return std::max<float_type>(x, 0);
}

float_type selu_activation(float_type x)
{
    const float_type alpha =
    static_cast<float_type>(1.6732632423543772848170429916717);
    const float_type scale =
    static_cast<float_type>(1.0507009873554804934193349852946);
    return scale * (x >= 0 ? x : alpha * (std::exp(x) - 1));
}

float_type elu_activation(float_type x)
{
    return x >= 0 ? x : std::exp(x) - 1;
}

std::function<float_type(float_type)> get_activation_func(const std::string& activation_func_name)
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
    return {}; //should never be called
}

inline tensor5s lstm_impl(const tensor5& input,
                          const std::size_t n_units,
                          const bool use_bias,
                          const bool return_sequences,
                          const float_vec& weights,
                          const float_vec& recurrent_weights,
                          const float_vec& bias,
                          const std::string& activation,
                          const std::string& recurrent_activation)
{
    const RowMajorMatrixXf W = eigen_row_major_mat_from_values(weights.size() / (n_units * 4), n_units * 4, weights);
    const RowMajorMatrixXf U = eigen_row_major_mat_from_values(n_units, n_units * 4, recurrent_weights);
    
    // initialize cell output states h, and cell memory states c for t-1 with zeros
    RowMajorMatrixXf h(1, n_units);
    RowMajorMatrixXf c(1, n_units);
    h.setZero();
    c.setZero();

    std::size_t n_timesteps = input.shape().width_;
    std::size_t n_features = input.shape().depth_;

    RowMajorMatrixXf in(n_timesteps, n_features);

    // write input to eigen matrix

    for (std::size_t a_t = 0; a_t < n_timesteps; ++a_t)
        for (std::size_t a_f = 0; a_f < n_features; ++a_f)
            in(EigenIndex(a_t), EigenIndex(a_f)) = input.get(0, 0, 0, a_t, a_f);

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

    tensor5s lstm_result;

    if (return_sequences)
        lstm_result = {tensor5(shape5(1, 1, 1, n_timesteps, n_units), float_type(0))};
    else
        lstm_result = {tensor5(shape5(1, 1, 1, 1, n_units), float_type(0))};

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
                lstm_result.front().set(0, 0, 0, std::size_t(k), std::size_t(idx), h(idx));
        else if (k == EigenIndex(n_timesteps) - 1)
            for (EigenIndex idx = 0; idx < n; ++idx)
                lstm_result.front().set(0, 0, 0, 0, std::size_t(idx), h(idx));
        }
    

    return lstm_result;
}
    
inline tensor5 reverse_time_series_in_tensor5(const tensor5& ts)
    {
            tensor5 reversed = tensor5(ts.shape(), float_type(0.0));
            std::size_t n = 0;
            for (std::size_t x =  ts.shape().width_; x-- > 0; )
            {
                for (std::size_t z = 0; z < ts.shape().depth_; ++z )
                    reversed.set(0, 0, 0, n, z ,ts.get(0, 0, 0, x, z));
                
                n++;
            }
        
        return reversed;
    }
    
} } // namespace fdeep, namespace internal
