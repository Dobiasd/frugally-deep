// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/base64.hpp"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#endif
#if defined _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4706)
#pragma warning(disable : 4996)
#endif
#include <nlohmann/json.hpp>
#if defined _MSC_VER
#pragma warning(pop)
#endif
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
    if (data.is_array())
    {
        assertion(data.size() == 1 || data.size() == 2,
            "invalid number of dimensions in shape2");
        if (data.size() == 1)
            return shape2(1, data[0]);
        else
            return shape2(data[0], data[1]);
    }
    else
    {
        const std::size_t width = data;
        return shape2(1, width);
    }
}

inline std::size_t create_size_t(const nlohmann::json& int_data)
{
    const int val = int_data;
    assertion(val >= 0, "invalid size_t value");
    return static_cast<std::size_t>(val);
}

inline float_vec decode_floats(const nlohmann::json& data)
{
    assertion(data.is_array() || data.is_string(),
        "invalid float array format");

    if (data.is_array() && !data.empty() && data[0].is_number())
    {
        const float_vec result = data;
        return result;
    }

    assertion(std::numeric_limits<float>::is_iec559,
        "The floating-point format of your system is not supported.");

    std::vector<std::string> strs = data;
    const auto res = Base64_decode(fplus::concat(std::move(strs)));
    strs.clear(); // free RAM
    float_vec out;
    assertion(res.size() % 4 == 0, "invalid float vector data");
    out.reserve(res.size() / 4);
    for (std::size_t i = 0; i < res.size(); i+=4)
    {
        float_type val = static_cast<float_type>(
            *(reinterpret_cast<const float*>(&(res[i]))));
        out.push_back(val);
    }
    return out;
}

inline tensor3 create_tensor3(const nlohmann::json& data)
{
    const shape3 shape = create_shape3(data["shape"]);
    return tensor3(shape, decode_floats(data["values"]));
}

template <typename T, typename F>
std::vector<T> create_vector(F f, const nlohmann::json& data)
{
    if (data.is_array())
        return fplus::transform_convert<std::vector<T>>(f, data);
    else
        return fplus::singleton_seq(f(data));
}

inline std::vector<shape3> create_shape3s(const nlohmann::json& data)
{
    return create_vector<shape3>(create_shape3, data);
}

inline node_connection create_node_connection(const nlohmann::json& data)
{
    assertion(data.is_array(), "invalid format for inbound node");
    const std::string layer_id = data.front();
    const auto node_idx = create_size_t(data[1]);
    const auto tensor_idx = create_size_t(data[2]);
    return node_connection(layer_id, node_idx, tensor_idx);
}

using get_param_f =
    std::function<nlohmann::json(const std::string&, const std::string&)>;
using get_global_param_f = std::function<nlohmann::json(const std::string&)>;

layer_ptr create_layer(const get_param_f&, const get_global_param_f&,
    const nlohmann::json&);

inline layer_ptr create_model_layer(const get_param_f& get_param,
    const get_global_param_f& get_global_param, const nlohmann::json& data,
    const std::string& name)
{
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
    std::fill(std::begin(xs), std::end(xs), static_cast<float_type>(0));
}

inline padding create_padding(const std::string& padding_str)
{
    return fplus::throw_on_nothing(error("no padding"),
        fplus::choose<std::string, padding>({
        { std::string("valid"), padding::valid },
        { std::string("same"), padding::same },
    }, padding_str));
}

inline layer_ptr create_conv_2d_layer(const get_param_f& get_param,
    const get_global_param_f& get_global_param, const nlohmann::json& data,
    const std::string& name)
{
    const std::string padding_str = data["config"]["padding"];
    const auto pad_type = create_padding(padding_str);

    const shape2 strides = create_shape2(data["config"]["strides"]);
    const shape2 dilation_rate = create_shape2(data["config"]["dilation_rate"]);

    const auto filter_count = create_size_t(data["config"]["filters"]);
    float_vec bias(filter_count, 0);
    const bool use_bias = data["config"]["use_bias"];
    if (use_bias)
        bias = decode_floats(get_param(name, "bias"));
    assertion(bias.size() == filter_count, "size of bias does not match");

    const float_vec weights = decode_floats(get_param(name, "weights"));
    const shape2 kernel_size = create_shape2(data["config"]["kernel_size"]);
    assertion(weights.size() % kernel_size.area() == 0,
        "invalid number of weights");
    const std::size_t filter_depths =
        weights.size() / (kernel_size.area() * filter_count);
    const shape3 filter_shape(
        filter_depths, kernel_size.height_, kernel_size.width_);

    const bool padding_valid_uses_offset_depth_1 =
        get_global_param("conv2d_valid_offset_depth_1");
    const bool padding_same_uses_offset_depth_1 =
        get_global_param("conv2d_same_offset_depth_1");
    const bool padding_valid_uses_offset_depth_2 =
        get_global_param("conv2d_valid_offset_depth_2");
    const bool padding_same_uses_offset_depth_2 =
        get_global_param("conv2d_same_offset_depth_2");
    return std::make_shared<conv_2d_layer>(name,
        filter_shape, filter_count, strides, pad_type,
        padding_valid_uses_offset_depth_1, padding_same_uses_offset_depth_1,
        padding_valid_uses_offset_depth_2, padding_same_uses_offset_depth_2,
        dilation_rate, weights, bias);
}

inline layer_ptr create_conv_2d_transpose_layer(const get_param_f& get_param,
    const get_global_param_f& get_global_param, const nlohmann::json& data,
    const std::string& name)
{
    const std::string padding_str = data["config"]["padding"];
    const auto pad_type = create_padding(padding_str);

    const shape2 strides = create_shape2(data["config"]["strides"]);

    const auto filter_count = create_size_t(data["config"]["filters"]);
    float_vec bias(filter_count, 0);
    const bool use_bias = data["config"]["use_bias"];
    if (use_bias)
        bias = decode_floats(get_param(name, "bias"));
    assertion(bias.size() == filter_count, "size of bias does not match");

    const float_vec weights = decode_floats(get_param(name, "weights"));
    const shape2 kernel_size = create_shape2(data["config"]["kernel_size"]);
    assertion(weights.size() % kernel_size.area() == 0,
        "invalid number of weights");
    const std::size_t filter_depths =
        weights.size() / (kernel_size.area() * filter_count);
    const shape3 filter_shape(
        filter_depths, kernel_size.height_, kernel_size.width_);

    const bool padding_valid_uses_offset_depth_1 =
        get_global_param("conv2d_valid_offset_depth_1");
    const bool padding_same_uses_offset_depth_1 =
        get_global_param("conv2d_same_offset_depth_1");
    const bool padding_valid_uses_offset_depth_2 =
        get_global_param("conv2d_valid_offset_depth_2");
    const bool padding_same_uses_offset_depth_2 =
        get_global_param("conv2d_same_offset_depth_2");
    return std::make_shared<conv_2d_transpose_layer>(name,
        filter_shape, filter_count, strides, pad_type,
        padding_valid_uses_offset_depth_1, padding_same_uses_offset_depth_1,
        padding_valid_uses_offset_depth_2, padding_same_uses_offset_depth_2,
        weights, bias);
}

inline layer_ptr create_separable_conv_2D_layer(const get_param_f& get_param,
    const get_global_param_f& get_global_param, const nlohmann::json& data,
    const std::string& name)
{
    const auto depth_multiplier = create_size_t(
        data["config"]["depth_multiplier"]);
    assertion(depth_multiplier == 1, "invalid depth_multiplier");

    const std::string padding_str = data["config"]["padding"];
    const auto pad_type = create_padding(padding_str);

    const shape2 strides = create_shape2(data["config"]["strides"]);
    const shape2 dilation_rate = create_shape2(data["config"]["dilation_rate"]);

    const auto filter_count = create_size_t(data["config"]["filters"]);
    float_vec bias(filter_count, 0);
    const bool use_bias = data["config"]["use_bias"];
    if (use_bias)
        bias = decode_floats(get_param(name, "bias"));
    assertion(bias.size() == filter_count, "size of bias does not match");

    const float_vec slice_weights = decode_floats(
        get_param(name, "slice_weights"));
    const float_vec stack_weights = decode_floats(
        get_param(name, "stack_weights"));
    const shape2 kernel_size = create_shape2(data["config"]["kernel_size"]);
    assertion(slice_weights.size() % kernel_size.area() == 0,
        "invalid number of weights");
    assertion(stack_weights.size() % filter_count == 0,
        "invalid number of weights");
    const std::size_t input_depth = slice_weights.size() / kernel_size.area();
    const std::size_t stack_output_depths_1 =
        stack_weights.size() / input_depth;
    assertion(stack_output_depths_1 == filter_count, "invalid weights sizes");
    const shape3 filter_shape(1,
        kernel_size.height_, kernel_size.width_);
    float_vec bias_0(input_depth, 0);
    const bool padding_valid_uses_offset_depth_1 =
        get_global_param("separable_conv2d_valid_offset_depth_1");
    const bool padding_same_uses_offset_depth_1 =
        get_global_param("separable_conv2d_same_offset_depth_1");
        const bool padding_valid_uses_offset_depth_2 =
        get_global_param("separable_conv2d_valid_offset_depth_2");
    const bool padding_same_uses_offset_depth_2 =
        get_global_param("separable_conv2d_same_offset_depth_2");
    return std::make_shared<separable_conv_2d_layer>(name, input_depth,
        filter_shape, filter_count, strides, pad_type,
        padding_valid_uses_offset_depth_1, padding_same_uses_offset_depth_1,
        padding_valid_uses_offset_depth_2, padding_same_uses_offset_depth_2,
        dilation_rate, slice_weights, stack_weights, bias_0, bias);
}

inline layer_ptr create_input_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    assertion(data["inbound_nodes"].empty(),
        "input layer is not allowed to have inbound nodes");
    const auto input_shape = create_shape3(data["config"]["batch_input_shape"]);
    return std::make_shared<input_layer>(name, input_shape);
}

inline layer_ptr create_batch_normalization_layer(const get_param_f& get_param,
    const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    const float_vec moving_mean = decode_floats(get_param(name, "moving_mean"));
    const float_vec moving_variance =
        decode_floats(get_param(name, "moving_variance"));
    const bool center = data["config"]["center"];
    const bool scale = data["config"]["scale"];
    const float_type epsilon = data["config"]["epsilon"];
    float_vec gamma;
    float_vec beta;
    if (scale) gamma = decode_floats(get_param(name, "gamma"));
    if (center) beta = decode_floats(get_param(name, "beta"));
    return std::make_shared<batch_normalization_layer>(
        name, moving_mean, moving_variance, beta, gamma, epsilon);
}

inline layer_ptr create_dropout_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    // dropout rate equals zero in forward pass
    return std::make_shared<linear_layer>(name);
}

inline layer_ptr create_max_pooling_2d_layer(
    const get_param_f&, const get_global_param_f& get_global_param,
    const nlohmann::json& data, const std::string& name)
{
    const auto pool_size = create_shape2(data["config"]["pool_size"]);
    const auto strides = create_shape2(data["config"]["strides"]);
    const std::string padding_str = data["config"]["padding"];
    const auto pad_type = create_padding(padding_str);
    const bool padding_valid_uses_offset =
        get_global_param("max_pooling_2d_valid_offset");
    const bool padding_same_uses_offset =
        get_global_param("max_pooling_2d_same_offset");
    return std::make_shared<max_pooling_2d_layer>(name,
        pool_size, strides, pad_type,
        padding_valid_uses_offset,
        padding_same_uses_offset);
}

inline layer_ptr create_average_pooling_2d_layer(
    const get_param_f&, const get_global_param_f& get_global_param,
    const nlohmann::json& data, const std::string& name)
{
    const auto pool_size = create_shape2(data["config"]["pool_size"]);
    const auto strides = create_shape2(data["config"]["strides"]);
    const std::string padding_str = data["config"]["padding"];
    const auto pad_type = create_padding(padding_str);
    const bool padding_valid_uses_offset =
        get_global_param("average_pooling_2d_valid_offset");
    const bool padding_same_uses_offset =
        get_global_param("average_pooling_2d_same_offset");
    return std::make_shared<average_pooling_2d_layer>(name,
        pool_size, strides, pad_type,
        padding_valid_uses_offset,
        padding_same_uses_offset);
}

inline layer_ptr create_global_max_pooling_2d_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<global_max_pooling_2d_layer>(name);
}

inline layer_ptr create_global_average_pooling_2d_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<global_average_pooling_2d_layer>(name);
}

inline layer_ptr create_upsampling_2d_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    const auto scale_factor = create_shape2(data["config"]["size"]);
    return std::make_shared<upsampling_2d_layer>(name, scale_factor);
}

inline layer_ptr create_dense_layer(const get_param_f& get_param,
    const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    const float_vec weights = decode_floats(get_param(name, "weights"));

    std::size_t units = data["config"]["units"];
    float_vec bias(units, 0);
    const bool use_bias = data["config"]["use_bias"];
    if (use_bias)
        bias = decode_floats(get_param(name, "bias"));
    assertion(bias.size() == units, "size of bias does not match");

    return std::make_shared<dense_layer>(
        name, units, weights, bias);
}

inline layer_ptr create_concatename_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<concatenate_layer>(name);
}

inline layer_ptr create_add_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<add_layer>(name);
}

inline layer_ptr create_maximum_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<maximum_layer>(name);
}

inline layer_ptr create_flatten_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<flatten_layer>(name);
}

inline layer_ptr create_zero_padding_2d_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    const auto padding =
        create_vector<std::vector<std::size_t>>(fplus::bind_1st_of_2(
            create_vector<std::size_t, decltype(create_size_t)>, create_size_t),
            data["config"]["padding"]);

    assertion(padding.size() == 2 && padding[0].size() == padding[1].size(),
        "invalid padding format");

    if (padding[0].size() == 1)
    {
        const std::size_t top_pad = 0;
        const std::size_t bottom_pad = 0;
        const std::size_t left_pad = padding[0][0];
        const std::size_t right_pad = padding[1][0];
        return std::make_shared<zero_padding_2d_layer>(name,
            top_pad, bottom_pad, left_pad, right_pad);
    }
    else
    {
        const std::size_t top_pad = padding[0][0];
        const std::size_t bottom_pad = padding[0][1];
        const std::size_t left_pad = padding[1][0];
        const std::size_t right_pad = padding[1][1];
        return std::make_shared<zero_padding_2d_layer>(name,
            top_pad, bottom_pad, left_pad, right_pad);
    }
}

inline layer_ptr create_cropping_2d_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    const auto cropping =
        create_vector<std::vector<std::size_t>>(fplus::bind_1st_of_2(
            create_vector<std::size_t, decltype(create_size_t)>, create_size_t),
            data["config"]["cropping"]);

    assertion(cropping.size() == 2 && cropping[0].size() == cropping[1].size(),
        "invalid cropping format");

    if (cropping[0].size() == 1)
    {
        const std::size_t top_crop = 0;
        const std::size_t bottom_crop = 0;
        const std::size_t left_crop = cropping[0][0];
        const std::size_t right_crop = cropping[1][0];
        return std::make_shared<cropping_2d_layer>(name,
            top_crop, bottom_crop, left_crop, right_crop);
    }
    else
    {
        const std::size_t top_crop = cropping[0][0];
        const std::size_t bottom_crop = cropping[0][1];
        const std::size_t left_crop = cropping[1][0];
        const std::size_t right_crop = cropping[1][1];
        return std::make_shared<cropping_2d_layer>(name,
            top_crop, bottom_crop, left_crop, right_crop);
    }
}

inline bool json_obj_has_member(const nlohmann::json& data,
    const std::string& member_name)
{
    return data.is_object() && data.find(member_name) != data.end();
}

inline activation_layer_ptr create_linear_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<linear_layer>(name);
}

inline activation_layer_ptr create_softmax_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<softmax_layer>(name);
}

inline activation_layer_ptr create_softplus_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<softplus_layer>(name);
}

inline activation_layer_ptr create_tanh_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<tanh_layer>(name);
}

inline activation_layer_ptr create_sigmoid_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<sigmoid_layer>(name);
}

inline activation_layer_ptr create_hard_sigmoid_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<hard_sigmoid_layer>(name);
}

inline activation_layer_ptr create_relu_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<relu_layer>(name);
}

inline activation_layer_ptr create_selu_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<selu_layer>(name);
}

inline activation_layer_ptr create_leaky_relu_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    float_type alpha = 1.0f;
    if (json_obj_has_member(data, "config") &&
        json_obj_has_member(data["config"], "alpha"))
    {
        alpha = data["config"]["alpha"];
    }
    return std::make_shared<leaky_relu_layer>(name, alpha);
}

inline layer_ptr create_leaky_relu_layer_isolated(
    const get_param_f& get_param, const get_global_param_f& get_global_param,
    const nlohmann::json& data, const std::string& name)
{
    return create_leaky_relu_layer(get_param, get_global_param, data, name);
}

inline activation_layer_ptr create_elu_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    float_type alpha = 1.0f;
    if (json_obj_has_member(data, "config") &&
        json_obj_has_member(data["config"], "alpha"))
    {
        alpha = data["config"]["alpha"];
    }
    return std::make_shared<elu_layer>(name, alpha);
}

inline layer_ptr create_elu_layer_isolated(
    const get_param_f& get_param, const get_global_param_f& get_global_param,
    const nlohmann::json& data, const std::string& name)
{
    return create_elu_layer(get_param, get_global_param, data, name);
}

inline activation_layer_ptr create_activation_layer_type_name(
    const get_param_f& get_param, const get_global_param_f& get_global_param,
    const nlohmann::json& data,
    const std::string& type, const std::string& name)
{
    const std::unordered_map<std::string,
            std::function<activation_layer_ptr(const get_param_f&,
                const get_global_param_f&, const nlohmann::json&,
                const std::string&)>>
    creators = {
        {"linear", create_linear_layer},
        {"softmax", create_softmax_layer},
        {"softplus", create_softplus_layer},
        {"tanh", create_tanh_layer},
        {"sigmoid", create_sigmoid_layer},
        {"hard_sigmoid", create_hard_sigmoid_layer},
        {"relu", create_relu_layer},
        {"selu", create_selu_layer},
        {"elu", create_elu_layer}
    };

    return fplus::throw_on_nothing(
        error("unknown activation type: " + type),
        fplus::get_from_map(creators, type))(
            get_param, get_global_param, data, name);
}

inline layer_ptr create_activation_layer(
    const get_param_f& get_param, const get_global_param_f& get_global_param,
    const nlohmann::json& data, const std::string& name)
{
    const std::string type = data["config"]["activation"];
    return create_activation_layer_type_name(get_param, get_global_param,
        data, type, name);
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
                const get_global_param_f&, const nlohmann::json&,
                const std::string&)>>
        creators = {
            {"Model", create_model_layer},
            {"Conv1D", create_conv_2d_layer},
            {"Conv2D", create_conv_2d_layer},
            {"Conv2DTranspose", create_conv_2d_transpose_layer},
            {"SeparableConv1D", create_separable_conv_2D_layer},
            {"SeparableConv2D", create_separable_conv_2D_layer},
            {"InputLayer", create_input_layer},
            {"BatchNormalization", create_batch_normalization_layer},
            {"Dropout", create_dropout_layer},
            {"LeakyReLU", create_leaky_relu_layer_isolated},
            {"ELU", create_elu_layer_isolated},
            {"MaxPooling1D", create_max_pooling_2d_layer},
            {"MaxPooling2D", create_max_pooling_2d_layer},
            {"AveragePooling1D", create_average_pooling_2d_layer},
            {"AveragePooling2D", create_average_pooling_2d_layer},
            {"GlobalMaxPooling1D", create_global_max_pooling_2d_layer},
            {"GlobalMaxPooling2D", create_global_max_pooling_2d_layer},
            {"GlobalAveragePooling1D", create_global_average_pooling_2d_layer},
            {"GlobalAveragePooling2D", create_global_average_pooling_2d_layer},
            {"UpSampling1D", create_upsampling_2d_layer},
            {"UpSampling2D", create_upsampling_2d_layer},
            {"Dense", create_dense_layer},
            {"Add", create_add_layer},
            {"Maximum", create_maximum_layer},
            {"Concatenate", create_concatename_layer},
            {"Flatten", create_flatten_layer},
            {"ZeroPadding1D", create_zero_padding_2d_layer},
            {"ZeroPadding2D", create_zero_padding_2d_layer},
            {"Cropping1D", create_cropping_2d_layer},
            {"Cropping2D", create_cropping_2d_layer},
            {"Activation", create_activation_layer}
        };

    const std::string type = data["class_name"];

    auto result = fplus::throw_on_nothing(
        error("unknown layer type: " + type),
        fplus::get_from_map(creators, type))(
            get_param, get_global_param, data, name);

    if (type != "Activation" &&
        json_obj_has_member(data["config"], "activation"))
    {
        result->set_activation(
            create_activation_layer_type_name(get_param, get_global_param, data,
                data["config"]["activation"], ""));
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

inline void check_test_outputs(float_type epsilon,
    const tensor3s& outputs, const tensor3s& targets)
{
    assertion(outputs.size() == targets.size(), "invalid output count");
    for (std::size_t i = 0; i < outputs.size(); ++i)
    {
        const auto& output = outputs[i];
        const auto& target = targets[i];
        assertion(output.shape() == target.shape(), "wrong output size");
        for (std::size_t z = 0; z < output.shape().depth_; ++z)
        {
            for (std::size_t y = 0; y < output.shape().height_; ++y)
            {
                for (std::size_t x = 0; x < output.shape().width_; ++x)
                {
                    if (!fplus::is_in_closed_interval_around(epsilon,
                        target.get(z, y, x), output.get(z, y, x)))
                    {
                        const std::string msg =
                            std::string("test failed: ") +
                            "output=" + fplus::show(i) + " " +
                            "pos=" +
                            fplus::show(z) + "," +
                            fplus::show(y) + "," +
                            fplus::show(x) + " " +
                            "value=" + fplus::show(output.get(z, y, x)) + " "
                            "target=" + fplus::show(target.get(z, y, x));
                        internal::assertion(false, msg);
                    }
                }
            }
        }
    }
}

} } // namespace fdeep, namespace internal
