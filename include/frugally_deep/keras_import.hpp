// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/frugally_deep.hpp"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#endif
// source: https://github.com/nlohmann/json
#include "frugally_deep/json.hpp"
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

#include <fplus/fplus.hpp>
#include <iostream>
#include <chrono>

namespace fdeep { namespace internal
{

inline shape3 create_shape3(const nlohmann::json& data)
{
    assertion(data.is_array(), "shape3 needs to be an array");
    assertion(data.size() > 0, "need at least one dimension");
    const std::size_t offset = data.front().is_null() ? 1 : 0;
    if (data.size() == 1 + offset)
        return shape3(0, 0, data[0 + offset]);
    if (data.size() == 2 + offset)
        return shape3(0, data[0 + offset], data[1 + offset]);
    if (data.size() == 3 + offset)
        return shape3(data[0 + offset], data[1 + offset], data[2 + offset]);
    raise_error("shape3 needs 1, 2 or 3 dimensions");
    return shape3(0, 0, 0);
}

inline shape2 create_shape2(const nlohmann::json& data)
{
    assertion(data.is_array(), "shape2 needs to be an array");
    if (data.size() == 1)
        return shape2(0, data.front());
    if (data.size() == 2)
        return shape2(data.front(), data[1]);
    raise_error("shape2 needs 1 or 2 dimensions");
    return shape2(0, 0);
}

inline tensor3 create_tensor3(const nlohmann::json& data)
{
    const shape3 shape = create_shape3(data["shape"]);
    const float_vec values = data["values"];
    return tensor3(shape, values);
}

template <typename T, typename F>
std::vector<T> create_vector(F f, const nlohmann::json& data)
{
    assertion(data.is_array(), "data needs to be an array");
    return fplus::transform_convert<std::vector<T>>(f, data);
}

inline float_t create_singleton_vec(const nlohmann::json& data)
{
    float_vec values = data;
    assertion(values.size() == 1, "need exactly one value");
    return values.front();
}

inline node_connection create_node_connection(const nlohmann::json& data)
{
    assertion(data.is_array(), "invalid format for inbound node");
    const std::string layer_id = data.front();
    const std::size_t node_idx = data[1];
    const std::size_t tensor_idx = data[2];
    return node_connection(layer_id, node_idx, tensor_idx);
}

using get_param_f =
    std::function<float_vec(const std::string&, const std::string&)>;
using get_global_param_f = std::function<nlohmann::json(const std::string&)>;

layer_ptr create_layer(const get_param_f&, const get_global_param_f&,
    const nlohmann::json&);

inline layer_ptr create_model_layer(const get_param_f& get_param,
    const get_global_param_f& get_global_param, const nlohmann::json& data)
{
    //output_nodes
    //input_nodes
    const std::string name = data["config"]["name"];

    assertion(data["config"]["layers"].is_array(), "missing layers array");

    const auto layers = create_vector<layer_ptr>(
        fplus::bind_1st_and_2nd_of_3(create_layer, get_param, get_global_param),
        data["config"]["layers"]);

    assertion(data["config"]["input_layers"].is_array(), "no input layers");

    const auto inputs = create_vector<node_connection>(
        create_node_connection, data["config"]["input_layers"]);

    const auto outputs = create_vector<node_connection>(
        create_node_connection, data["config"]["output_layers"]);

    return std::make_shared<model_layer>(name, layers, inputs, outputs);
}

inline void fill_with_zeros(float_vec& xs)
{
    std::fill(std::begin(xs), std::end(xs), 0);
}

inline layer_ptr create_conv2d_layer(const get_param_f& get_param,
    const get_global_param_f& get_global_param, const nlohmann::json& data)
{
    const std::string name = data["name"];
    assertion(data["config"]["data_format"] == "channels_last",
        "only channels_last data format supported");

    const std::string padding_str = data["config"]["padding"];
    const auto pad_type = fplus::throw_on_nothing(error("no padding"),
        fplus::choose<std::string, padding>({
        { std::string("valid"), padding::valid },
        { std::string("same"), padding::same },
    }, padding_str));

    const shape2 strides = swap_shape2_dims(
        create_shape2(data["config"]["strides"]));

    const std::size_t filter_count = data["config"]["filters"];
    float_vec bias(filter_count, 0);
    const bool use_bias = data["config"]["use_bias"];
    if (use_bias)
        bias = get_param(name, "bias");
    assertion(bias.size() == filter_count, "size of bias does not match");

    const float_vec weights = get_param(name, "weights");
    const shape2 kernel_size = swap_shape2_dims(
        create_shape2(data["config"]["kernel_size"]));
    assertion(weights.size() % kernel_size.area() == 0,
        "invalid number of weights");
    const std::size_t filter_depths =
        weights.size() / (kernel_size.area() * filter_count);
    const shape3 filter_size(
        filter_depths, kernel_size.height_, kernel_size.width_);

    const bool padding_valid_uses_offset =
        get_global_param("conv2d_padding_valid_uses_offset");
    const bool padding_same_uses_offset =
        get_global_param("conv2d_padding_same_uses_offset");
    return std::make_shared<conv2d_layer>(name,
        filter_size, filter_count, strides, pad_type,
        padding_valid_uses_offset, padding_same_uses_offset,
        weights, bias);
}

inline layer_ptr create_separable_conv2D_layer(const get_param_f& get_param,
    const get_global_param_f& get_global_param, const nlohmann::json& data)
{
    const std::string name = data["name"];
    assertion(data["config"]["data_format"] == "channels_last",
        "only channels_last data format supported");

    const std::string padding_str = data["config"]["padding"];
    const auto pad_type = fplus::throw_on_nothing(error("no padding"),
        fplus::choose<std::string, padding>({
        { std::string("valid"), padding::valid },
        { std::string("same"), padding::same },
    }, padding_str));
    const std::size_t depth_multiplier = data["config"]["depth_multiplier"];
    const shape2 strides = swap_shape2_dims(
        create_shape2(data["config"]["strides"]));

    const std::size_t filter_count = data["config"]["filters"];
    float_vec bias(filter_count, 0);
    const bool use_bias = data["config"]["use_bias"];
    if (use_bias)
        bias = get_param(name, "bias");
    assertion(bias.size() == filter_count, "size of bias does not match");

    const float_vec slice_weights = get_param(name, "slice_weights");
    const float_vec stack_weights = get_param(name, "stack_weights");
    const shape2 kernel_size = swap_shape2_dims(
        create_shape2(data["config"]["kernel_size"]));
    assertion(slice_weights.size() % kernel_size.area() == 0,
        "invalid number of weights");
    assertion(stack_weights.size() % filter_count == 0,
        "invalid number of weights");
    const std::size_t input_depth = slice_weights.size() /
        (depth_multiplier * kernel_size.area());
    const std::size_t stack_output_depths_1 =
        stack_weights.size() / (depth_multiplier * input_depth);
    assertion(stack_output_depths_1 == filter_count, "invalid weights sizes");
    const shape3 filter_size(1,
        kernel_size.height_, kernel_size.width_);
    float_vec bias_0(depth_multiplier * input_depth, 0);
    const bool padding_valid_uses_offset =
        get_global_param("separable_conv2d_padding_valid_uses_offset");
    const bool padding_same_uses_offset =
        get_global_param("separable_conv2d_padding_same_uses_offset");
    return std::make_shared<separable_conv2d_layer>(name, input_depth,
        filter_size, filter_count, strides, pad_type,
        padding_valid_uses_offset, padding_same_uses_offset,
        depth_multiplier,
        slice_weights, stack_weights, bias_0, bias);
}

inline layer_ptr create_input_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data)
{
    assertion(data["inbound_nodes"].empty(),
        "input layer is not allowed to have inbound nodes");
    const std::string name = data["name"];
    const auto input_shape = create_shape3(data["config"]["batch_input_shape"]);
    return std::make_shared<input_layer>(name, input_shape);
}

inline layer_ptr create_batch_normalization_layer(const get_param_f& get_param,
    const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    float_t epsilon = data["config"]["epsilon"];
    const bool center = data["config"]["center"];
    const bool scale = data["config"]["scale"];
    float_vec gamma;
    float_vec beta;
    if (scale) gamma = get_param(name, "gamma");
    if (center) beta = get_param(name, "beta");
    return std::make_shared<batch_normalization_layer>(
        name, epsilon, beta, gamma);
}

inline layer_ptr create_dropout_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    // dropout rate equals zero in forward pass
    return std::make_shared<linear_layer>(name);
}

inline layer_ptr create_leaky_relu_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    float_t alpha = data["config"]["alpha"];
    return std::make_shared<leaky_relu_layer>(name, alpha);
}

inline layer_ptr create_elu_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    float_t alpha = data["config"]["alpha"];
    return std::make_shared<elu_layer>(name, alpha);
}

inline layer_ptr create_max_pooling2d_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    assertion(data["config"]["data_format"] == "channels_last",
        "only channels_last data format supported");
    const auto pool_size = create_shape2(data["config"]["pool_size"]);
    const auto strides = create_shape2(data["config"]["strides"]);
    // todo: support pool_size != strides
    assertion(pool_size == strides, "pool_size and strides not equal");
    // todo: support non-proportional sizes
    assertion(pool_size.width_ == pool_size.height_,
        "pooling not proportional");
    return std::make_shared<max_pooling_2d_layer>(name, pool_size.width_);
}

inline layer_ptr create_average_pooling2d_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    assertion(data["config"]["data_format"] == "channels_last",
        "only channels_last data format supported");
    const auto pool_size = create_shape2(data["config"]["pool_size"]);
    const auto strides = create_shape2(data["config"]["strides"]);
    // todo: support pool_size != strides
    assertion(pool_size == strides, "pool_size and strides not equal");
    // todo: support non-proportional sizes
    assertion(pool_size.width_ == pool_size.height_,
        "pooling not proportional");
    return std::make_shared<average_pooling_2d_layer>(name, pool_size.width_);
}

inline layer_ptr create_upsampling2d_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    assertion(data["config"]["data_format"] == "channels_last",
        "only channels_last data format supported");
    const auto size = create_shape2(data["config"]["size"]);
    assertion(size.width_ == size.height_, "invalid scale factor");
    return std::make_shared<upsampling2d_layer>(name, size.width_);
}

inline layer_ptr create_dense_layer(const get_param_f& get_param,
    const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    const float_vec weights = get_param(name, "weights");

    std::size_t units = data["config"]["units"];
    float_vec bias(units, 0);
    const bool use_bias = data["config"]["use_bias"];
    if (use_bias)
        bias = get_param(name, "bias");
    assertion(bias.size() == units, "size of bias does not match");

    return std::make_shared<dense_layer>(
        name, units, weights, bias);
}

inline layer_ptr create_concatename_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    return std::make_shared<concatenate_layer>(name);
}

inline layer_ptr create_flatten_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    return std::make_shared<flatten_layer>(name);
}

inline layer_ptr create_zero_padding2d_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    assertion(data["config"]["data_format"] == "channels_last",
        "only channels_last data format supported");
    const auto create_size_t = [](const nlohmann::json& int_data) -> std::size_t
    {
        const int val = int_data;
        assertion(val >= 0, "invalid size_t value");
        return static_cast<std::size_t>(val);
    };
    const auto padding =
        create_vector<std::vector<std::size_t>>(fplus::bind_1st_of_2(
            create_vector<std::size_t, decltype(create_size_t)>, create_size_t),
            data["config"]["padding"]);
    assertion(padding.size() == 2 &&
        padding[0].size() == 2 && padding[1].size() == 2,
        "invalid padding format");
    const std::size_t top_pad = padding[1][0];
    const std::size_t bottom_pad = padding[1][1];
    const std::size_t left_pad = padding[0][0];
    const std::size_t right_pad = padding[0][1];
    return std::make_shared<zero_padding2d_layer>(name,
        top_pad, bottom_pad, left_pad, right_pad);
}

inline activation_layer_ptr create_linear_layer(const std::string& name)
{
    return std::make_shared<linear_layer>(name);
}

inline activation_layer_ptr create_softmax_layer(const std::string& name)
{
    return std::make_shared<softmax_layer>(name);
}

inline activation_layer_ptr create_softplus_layer(const std::string& name)
{
    return std::make_shared<softplus_layer>(name);
}

inline activation_layer_ptr create_tanh_layer(const std::string& name)
{
    return std::make_shared<tanh_layer>(name);
}

inline activation_layer_ptr create_sigmoid_layer(const std::string& name)
{
    return std::make_shared<sigmoid_layer>(name);
}

inline activation_layer_ptr create_hard_sigmoid_layer(
    const std::string& name)
{
    return std::make_shared<hard_sigmoid_layer>(name);
}

inline activation_layer_ptr create_relu_layer(const std::string& name)
{
    return std::make_shared<relu_layer>(name);
}

inline activation_layer_ptr create_selu_layer(const std::string& name)
{
    return std::make_shared<selu_layer>(name);
}

inline activation_layer_ptr create_activation_layer(
    const std::string& type, const std::string& name)
{
    const std::unordered_map<std::string,
        std::function<activation_layer_ptr(const std::string&)>>
    creators = {
        {"linear", create_linear_layer},
        {"softmax", create_softmax_layer},
        {"softplus", create_softplus_layer},
        {"tanh", create_tanh_layer},
        {"sigmoid", create_sigmoid_layer},
        {"hard_sigmoid", create_hard_sigmoid_layer},
        {"relu", create_relu_layer},
        {"selu", create_selu_layer}
    };

    return fplus::throw_on_nothing(
        error("unknown activation type: " + type),
        fplus::get_from_map(creators, type))(name);
}

inline layer_ptr create_activation_layer_as_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data)
{
    const std::string name = data["name"];
    const std::string type = data["config"]["activation"];
    return create_activation_layer(type, name);
}

inline bool json_obj_has_member(const nlohmann::json& data,
    const std::string& member_name)
{
    return data.is_object() && data.find(member_name) != data.end();
}

inline node create_node(const nlohmann::json& inbound_nodes_data)
{
    assertion(inbound_nodes_data.is_array(), "nodes need to be an array");
    return node(create_vector<node_connection>(create_node_connection,
            inbound_nodes_data));
}

inline nodes create_nodes(const nlohmann::json& data)
{
    assertion(data["inbound_nodes"].is_array(), "no inbound nodes");
    const std::vector<nlohmann::json> inbound_nodes_data =
        data["inbound_nodes"];
    return fplus::transform(create_node, inbound_nodes_data);
}

inline layer_ptr create_layer(const get_param_f& get_param,
    const get_global_param_f& get_global_param, const nlohmann::json& data)
{
    const std::string name = data["name"];

    const std::unordered_map<std::string,
            std::function<layer_ptr(const get_param_f&,
                const get_global_param_f&, const nlohmann::json&)>>
        creators = {
            {"Model", create_model_layer},
            {"Conv2D", create_conv2d_layer},
            {"SeparableConv2D", create_separable_conv2D_layer},
            {"InputLayer", create_input_layer},
            {"BatchNormalization", create_batch_normalization_layer},
            {"Dropout", create_dropout_layer},
            {"LeakyReLU", create_leaky_relu_layer},
            {"ELU", create_elu_layer},
            {"MaxPooling2D", create_max_pooling2d_layer},
            {"AveragePooling2D", create_average_pooling2d_layer},
            {"UpSampling2D", create_upsampling2d_layer},
            {"Dense", create_dense_layer},
            {"Concatenate", create_concatename_layer},
            {"Flatten", create_flatten_layer},
            {"ZeroPadding2D", create_zero_padding2d_layer},
            {"Activation", create_activation_layer_as_layer}
        };

    const std::string type = data["class_name"];

    auto result = fplus::throw_on_nothing(
        error("unknown layer type: " + type),
        fplus::get_from_map(creators, type))(get_param, get_global_param, data);

    if (type != "Activation" &&
        json_obj_has_member(data["config"], "activation"))
    {
        result->set_activation(
            create_activation_layer(data["config"]["activation"], ""));
    }

    result->set_nodes(create_nodes(data));

    return result;
}

struct test_case
{
    tensor3s input_;
    tensor3s output_;
};

using test_cases = std::vector<test_case>;

inline test_case load_test_case(const nlohmann::json& data)
{
    assertion(data["inputs"].is_array(), "test needs inputs");
    assertion(data["outputs"].is_array(), "test needs outputs");
    return {
        create_vector<tensor3>(create_tensor3, data["inputs"]),
        create_vector<tensor3>(create_tensor3, data["outputs"])
    };
}

inline test_cases load_test_cases(const nlohmann::json& data)
{
    return create_vector<test_case>(load_test_case, data);
}

inline bool is_test_output_ok(const tensor3& output, const tensor3& target)
{
    assertion(output.shape() == target.shape(), "wrong output size");
    for (std::size_t z = 0; z < output.shape().depth_; ++z)
    {
        for (std::size_t y = 0; y < output.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < output.shape().width_; ++x)
            {
                if (!fplus::is_in_closed_interval_around(
                    static_cast<float_t>(0.01),
                    target.get(z, y, x), output.get(z, y, x)))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

inline bool are_test_outputs_ok(const tensor3s& output,
    const tensor3s& target)
{
    assertion(output.size() == target.size(), "invalid output count");
    return fplus::all(fplus::zip_with(is_test_output_ok, output, target));
}

class timer
{
public:
    timer() : beg_(clock::now()) {}
    void reset() { beg_ = clock::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second>
            (clock::now() - beg_).count(); }
private:
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::duration<double,
        std::ratio<1>> second;
    std::chrono::time_point<clock> beg_;
};

} } // namespace fdeep, namespace internal
