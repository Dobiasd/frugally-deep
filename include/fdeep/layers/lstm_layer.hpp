// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

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
        const auto input_shapes = fplus::transform(fplus_c_mem_fn_t(tensor3, shape, shape3), inputs);

        // ensure that tensor3 shape is (n, 1, 1) when using multiple tensor3s elements as seq_len
        if (inputs.size() > 1)
        {
            assertion(inputs.front().shape().height_ == 1 && inputs.front().shape().width_ == 1,
                      "height and width dimension must be 1, but shape is '" + show_shape3s(input_shapes) + "'");
        }
        else
        {
            assertion(inputs.front().shape().height_ == 1,
                      "height dimension must be 1, but shape is '" + show_shape3s(input_shapes) + "'");
        }

        return lstm_impl(inputs, W_, U_, bias_, activation_, recurrent_activation_);
    }

  private:
    static float_type linear_activation(float_type x)
    {
        return x;
    }

    static float_type tanh_activation(float_type x)
    {
        return std::tanh(x);
    }

    static float_type sigmoid_activation(float_type x)
    {
        return 1 / (1 + std::exp(-x));
    }

    static float_type hard_sigmoid_activation(float_type x)
    {
        return static_cast<float_type>(std::min(1.0, std::max(0.0, (0.2 * x) + 0.5)));
    }

    static float_type relu_activation(float_type x)
    {
        return std::max<float_type>(x, 0);
    }

    static float_type selu_activation(float_type x)
    {
        const float_type alpha =
            static_cast<float_type>(1.6732632423543772848170429916717);
        const float_type scale =
            static_cast<float_type>(1.0507009873554804934193349852946);
        return scale * (x >= 0 ? x : alpha * (std::exp(x) - 1));
    }

    static float_type elu_activation(float_type x)
    {
        return x >= 0 ? x : std::exp(x) - 1;
    }

    std::function<float_type(float_type)> get_activation_func(const std::string &activation_func_name) const
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

    tensor3s lstm_impl(const tensor3s &inputs,
                       const RowMajorMatrixXf &W,
                       const RowMajorMatrixXf &U,
                       const RowMajorMatrixXf &bias,
                       const std::string &activation,
                       const std::string &recurrent_activation) const
    {
        // initialize cell output states h, and cell memory states c for t-1 with zeros
        RowMajorMatrixXf h(1, n_units_);
        RowMajorMatrixXf c(1, n_units_);
        h.setZero();
        c.setZero();

        std::size_t n_timesteps;
        std::size_t n_features;

        const bool multi_inputs = inputs.size() > 1;

        if (multi_inputs)
        {
            n_timesteps = inputs.size();
            n_features = inputs.front().shape().depth_;
        }
        else
        {
            n_timesteps = inputs.front().shape().width_;
            n_features = inputs.front().shape().depth_;
        }

        RowMajorMatrixXf in(n_timesteps, n_features);

        // write input to eigen matrix
        if (multi_inputs)
        {
            for (std::size_t a_t = 0; a_t < n_timesteps; ++a_t)
                for (std::size_t a_f = 0; a_f < n_features; ++a_f)
                    in(EigenIndex(a_t), EigenIndex(a_f)) = inputs[a_t].get(a_f, 0, 0);
        }
        else
        {
            for (std::size_t a_t = 0; a_t < n_timesteps; ++a_t)
                for (std::size_t a_f = 0; a_f < n_features; ++a_f)
                    in(EigenIndex(a_t), EigenIndex(a_f)) = inputs.front().get(a_f, 0, a_t);
        }

        RowMajorMatrixXf X = in * W;

        if (use_bias_)
        {
            // define eigen vector type to be able to use broadcasting
            typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic> Vector_Xf;
            Vector_Xf b = bias;

            X.rowwise() += b;
        }

        // get activation functions
        auto act_func = get_activation_func(activation);
        auto act_func_recurrent = get_activation_func(recurrent_activation);

        // computing LSTM output
        const EigenIndex n = EigenIndex(n_units_);

        tensor3s lstm_result;

        if (multi_inputs)
            lstm_result = {};
        else if (return_sequences_)
            lstm_result = {tensor3(shape3(n_units_, 1, n_timesteps), float_type(0))};
        else
            lstm_result = {tensor3(shape3(n_units_, 1, 1), float_type(0))};

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

            if (multi_inputs)
            {
                if (return_sequences_)
                    lstm_result.push_back(tensor3(shape3(n_units_, 1, 1), eigen_mat_to_values(h)));
                else if (k == EigenIndex(n_timesteps) - 1)
                    lstm_result = {tensor3(shape3(n_units_, 1, 1), eigen_mat_to_values(h))};
            }
            else
            {
                if (return_sequences_)
                    for (EigenIndex idx = 0; idx < n; ++idx)
                        lstm_result.front().set(std::size_t(idx), 0, std::size_t(k), h(idx));
                else if (k == EigenIndex(n_timesteps) - 1)
                    for (EigenIndex idx = 0; idx < n; ++idx)
                        lstm_result.front().set(std::size_t(idx), 0, 0, h(idx));
            }
        }

        return lstm_result;
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
