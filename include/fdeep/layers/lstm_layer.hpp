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
    explicit lstm_layer(const std::string& name,
                        std::size_t n_units,
                        std::string& activation,
                        std::string& recurrent_activation,
                        bool use_bias,
                        bool return_sequences,
                        const RowMajorMatrixXf& W,
                        const RowMajorMatrixXf& U,
                        const RowMajorMatrixXf& bias)
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
        assertion(bias_.size() == EigenIndex(n_units_) * 4, "invalid bias size");
    }
    
protected:
    tensor3s apply_impl(const tensor3s& inputs) const override final
    {
        assertion(inputs.size() == 1, "invalid number of input tensors");
        const auto& input = inputs.front();
        return {lstm_impl(input, W_, U_, bias_, activation_, recurrent_activation_)};
    }
    
private:
    static float_type relu_activation(float_type x)
    {
        return std::max<float_type>(x, 0);
    }
    
    static float_type sigmoid_activation(float_type x)
    {
        return 1 / (1 + std::exp(-x));
    }
    
    static float_type hard_sigmoid_activation(float_type x)
    {
        return static_cast<float_type>(std::min(1.0, std::max(0.0, (0.2 * x) + 0.5)));
    }
    
    static float_type tanh_activation(float_type x)
    {
        return std::tanh(x);
    }
    
    std::function<float_type(float_type)> get_activation_func(const std::string& activation_func_name) const
    {
        if (activation_func_name == "relu")
            return relu_activation;
        else if (activation_func_name == "sigmoid")
            return sigmoid_activation;
        else if (activation_func_name == "hard_sigmoid")
            return hard_sigmoid_activation;
        else if (activation_func_name == "tanh")
            return tanh_activation;
        
        raise_error("activation function '" + activation_func_name + "' not yet implemented");
        return {};
    }
    
    tensor3 lstm_impl(const tensor3& input,
                      const RowMajorMatrixXf& W,
                      const RowMajorMatrixXf& U,
                      const RowMajorMatrixXf& bias,
                      const std::string& activation,
                      const std::string& recurrent_activation) const
    {
        // initialize cell output states h, and cell memory states c for t-1 with zeros
        RowMajorMatrixXf h_tm1(1, n_units_);
        RowMajorMatrixXf c_tm1(1, n_units_);
        h_tm1.setZero();
        c_tm1.setZero();
        
        const std::size_t x_width = input.shape().width_;
        const std::size_t y_height = input.shape().height_;
        
        std::size_t n_output_timesteps;
        
        // allocation of output matrix based on return_sequences
        if (return_sequences_ == true)
            n_output_timesteps = x_width;
        else
            n_output_timesteps = 1;

        RowMajorMatrixXf result(n_output_timesteps, n_units_);
      
        // write input to eigen matrix
        RowMajorMatrixXf in(x_width, y_height);
        
        for (std::size_t a_y = 0; a_y < y_height; ++a_y)
            for (std::size_t a_x = 0; a_x < x_width; ++a_x)
                in(EigenIndex(a_y), EigenIndex(a_x)) = input.get(0, a_y, a_x);
      
        // initialize X
        RowMajorMatrixXf X(x_width, n_units_ * 4);
        
        X = in * W;
        
        if (use_bias_)
        {
            // define eigen vector type to be able to use broadcasting
            typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic> Vector_Xf;
            Vector_Xf b(1, n_units_ * 4);
            b = bias;
            
            X.rowwise() += b;
        }
        
        // get activation functions
        const auto act_func = get_activation_func(activation);
        const auto act_func_recurrent = get_activation_func(recurrent_activation);
        
        // computing LSTM output
        const EigenIndex n = EigenIndex(n_units_);
        
        for (EigenIndex k = 0; k < EigenIndex(x_width); ++k)
        {
            const RowMajorMatrixXf ifco = h_tm1 * U;
           
            // Use of Matrix.block(): Block of size (p,q), starting at (i,j) matrix.block(i,j,p,q);  matrix.block<p,q>(i,j);
            const RowMajorMatrixXf i = (X.block(k, 0, 1, n) + ifco.block(0, 0, 1, n)).unaryExpr(act_func_recurrent);
            const RowMajorMatrixXf f = (X.block(k, n, 1, n) + ifco.block(0, n, 1, n)).unaryExpr(act_func_recurrent);
            const RowMajorMatrixXf c_pre = (X.block(k, n * 2, 1, n) + ifco.block(0, n * 2, 1, n)).unaryExpr(act_func);
            const RowMajorMatrixXf o = (X.block(k, n * 3, 1, n) + ifco.block(0, n * 3, 1, n)).unaryExpr(act_func_recurrent);
            
            c_tm1 = f.cwiseProduct(c_tm1) + i.cwiseProduct(c_pre);
            h_tm1 = o.cwiseProduct(c_tm1.unaryExpr(act_func));
           
            // save every h sequence or just last
            if (return_sequences_ == true)
                for (EigenIndex idx = 0; idx < n; ++idx)
                    result(k, idx) = h_tm1(idx);
            else if (k == EigenIndex(x_width) - 1)
                for (EigenIndex idx = 0; idx < n; ++idx)
                    result(0, idx) = h_tm1(idx);
        }
        
        return tensor3(shape3(1, n_output_timesteps, n_units_), eigen_mat_to_values(result));
    }

    const std::size_t n_units_;
    const std::string activation_;
    const std::string recurrent_activation_;
    const bool use_bias_;
    const bool return_sequences_;
    const RowMajorMatrixXf& W_;
    const RowMajorMatrixXf& U_;
    const RowMajorMatrixXf& bias_;
};

} // namespace internal
} // namespace fdeep
