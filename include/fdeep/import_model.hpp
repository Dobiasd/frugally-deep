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

#include "fdeep/common.hpp"
#include "fdeep/layers/add_layer.hpp"
#include "fdeep/layers/average_layer.hpp"
#include "fdeep/layers/average_pooling_2d_layer.hpp"
#include "fdeep/layers/batch_normalization_layer.hpp"
#include "fdeep/layers/bidirectional_layer.hpp"
#include "fdeep/layers/concatenate_layer.hpp"
#include "fdeep/layers/conv_2d_layer.hpp"
#include "fdeep/layers/cropping_2d_layer.hpp"
#include "fdeep/layers/dense_layer.hpp"
#include "fdeep/layers/depthwise_conv_2d_layer.hpp"
#include "fdeep/layers/elu_layer.hpp"
#include "fdeep/layers/flatten_layer.hpp"
#include "fdeep/layers/global_average_pooling_2d_layer.hpp"
#include "fdeep/layers/global_max_pooling_2d_layer.hpp"
#include "fdeep/layers/hard_sigmoid_layer.hpp"
#include "fdeep/layers/input_layer.hpp"
#include "fdeep/layers/layer.hpp"
#include "fdeep/layers/leaky_relu_layer.hpp"
#include "fdeep/layers/embedding_layer.hpp"
#include "fdeep/layers/lstm_layer.hpp"
#include "fdeep/layers/gru_layer.hpp"
#include "fdeep/layers/prelu_layer.hpp"
#include "fdeep/layers/linear_layer.hpp"
#include "fdeep/layers/max_pooling_2d_layer.hpp"
#include "fdeep/layers/maximum_layer.hpp"
#include "fdeep/layers/model_layer.hpp"
#include "fdeep/layers/multiply_layer.hpp"
#include "fdeep/layers/pooling_2d_layer.hpp"
#include "fdeep/layers/relu_layer.hpp"
#include "fdeep/layers/reshape_layer.hpp"
#include "fdeep/layers/separable_conv_2d_layer.hpp"
#include "fdeep/layers/selu_layer.hpp"
#include "fdeep/layers/sigmoid_layer.hpp"
#include "fdeep/layers/softmax_layer.hpp"
#include "fdeep/layers/softplus_layer.hpp"
#include "fdeep/layers/subtract_layer.hpp"
#include "fdeep/layers/tanh_layer.hpp"
#include "fdeep/layers/time_distributed_layer.hpp"
#include "fdeep/layers/upsampling_2d_layer.hpp"
#include "fdeep/layers/zero_padding_2d_layer.hpp"
#include "fdeep/shape5.hpp"
#include "fdeep/shape5_variable.hpp"
#include "fdeep/tensor5.hpp"

#include <fplus/fplus.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


namespace fdeep { namespace internal
{

template<typename KeyT, typename ValueT>
ValueT json_object_get(const nlohmann::json& data, KeyT&& key, ValueT&& default_value)
{
    auto&& it = data.find(key);
    if (it != data.end())
        return *it;
    else
        return default_value;
}

inline fplus::maybe<std::size_t> create_maybe_size_t(const nlohmann::json& data)
{
    if (data.is_null())
    {
        return fplus::nothing<std::size_t>();
    }
    const std::size_t result = data;
    return fplus::just(result);
}

inline shape5_variable create_shape5_variable(const nlohmann::json& data)
{
    assertion(data.is_array(), "shape5_variable needs to be an array");
    assertion(data.size() > 0, "need at least one dimension");
    if (data.size() == 1)
        return shape5_variable(
            fplus::nothing<std::size_t>(),
            fplus::nothing<std::size_t>(),
            fplus::nothing<std::size_t>(),
            fplus::nothing<std::size_t>(),
            create_maybe_size_t(data[0]));
    if (data.size() == 2)
        return shape5_variable(
            fplus::nothing<std::size_t>(),
            fplus::nothing<std::size_t>(),
            fplus::nothing<std::size_t>(),
            create_maybe_size_t(data[0]),
            create_maybe_size_t(data[1]));
    if (data.size() == 3)
        return shape5_variable(
            fplus::nothing<std::size_t>(),
            fplus::nothing<std::size_t>(),
            create_maybe_size_t(data[0]),
            create_maybe_size_t(data[1]),
            create_maybe_size_t(data[2]));
    if (data.size() == 4)
        return shape5_variable(
            fplus::nothing<std::size_t>(),
            create_maybe_size_t(data[0]),
            create_maybe_size_t(data[1]),
            create_maybe_size_t(data[2]),
            create_maybe_size_t(data[3]));
    if (data.size() == 5)
        return shape5_variable(
            create_maybe_size_t(data[0]),
            create_maybe_size_t(data[1]),
            create_maybe_size_t(data[2]),
            create_maybe_size_t(data[3]),
            create_maybe_size_t(data[4]));
     if (data.size() == 6) // todo: is this needed?
        return shape5_variable(
            create_maybe_size_t(data[1]),
            create_maybe_size_t(data[2]),
            create_maybe_size_t(data[3]),
            create_maybe_size_t(data[4]),
            create_maybe_size_t(data[5]));

    raise_error("shape5_variable needs 1, 2, 3, 4 or 5 dimensions");
    return shape5_variable(
        fplus::nothing<std::size_t>(),
        fplus::nothing<std::size_t>(),
        fplus::nothing<std::size_t>(),
        fplus::nothing<std::size_t>(),
        fplus::nothing<std::size_t>()); // Should never be called
}

inline shape5 create_shape5(const nlohmann::json& data)
{
    assertion(data.is_array(), "shape5 needs to be an array");
    assertion(data.size() > 0, "need at least one dimension");
    if (data.size() == 1)
        return shape5(1, 1, 1, 1, data[0]);
    if (data.size() == 2)
        return shape5(1, 1, 1, data[0], data[1]);
    if (data.size() == 3)
        return shape5(1, 1, data[0], data[1], data[2]);
    if (data.size() == 4)
        return shape5(1, data[0], data[1], data[2], data[3]);
    if (data.size() == 5)
        return shape5(data[0], data[1], data[2], data[3], data[4]);
    raise_error("shape5 needs 1, 2, 3, 4 or 5 dimensions");
    return shape5(0, 0, 0, 0, 0); // Should never be called
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

inline int create_int(const nlohmann::json& int_data)
{
    const int val = int_data;
    return val;
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

    const auto res = Base64_decode(json_data_strs_char_prodiver(data, '='));
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

inline tensor5 create_tensor5(const nlohmann::json& data)
{
    const shape5 shape = create_shape5(data["shape"]);
    return tensor5(shape, decode_floats(data["values"]));
}

template <typename T, typename F>
std::vector<T> create_vector(F f, const nlohmann::json& data)
{
    if (data.is_array())
        return fplus::transform_convert<std::vector<T>>(f, data);
    else
        return fplus::singleton_seq(f(data));
}

inline std::vector<shape5_variable> create_shape5s_variable(const nlohmann::json& data)
{
    return create_vector<shape5_variable>(create_shape5_variable, data);
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
    const shape5 filter_shape(1, 1,
        kernel_size.height_, kernel_size.width_, filter_depths);

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

inline layer_ptr create_separable_conv_2D_layer(const get_param_f& get_param,
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
    const shape5 filter_shape(1, 1, kernel_size.height_, kernel_size.width_, 1);
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

inline layer_ptr create_depthwise_conv_2D_layer(const get_param_f& get_param,
    const get_global_param_f& get_global_param, const nlohmann::json& data,
    const std::string& name)
{
    const std::string padding_str = data["config"]["padding"];
    const auto pad_type = create_padding(padding_str);

    const shape2 strides = create_shape2(data["config"]["strides"]);
    const shape2 dilation_rate = create_shape2(data["config"]["dilation_rate"]);

    const float_vec slice_weights = decode_floats(
        get_param(name, "slice_weights"));
    const shape2 kernel_size = create_shape2(data["config"]["kernel_size"]);
    assertion(slice_weights.size() % kernel_size.area() == 0,
        "invalid number of weights");
    const std::size_t input_depth = slice_weights.size() / kernel_size.area();
    const shape5 filter_shape(1, 1, kernel_size.height_, kernel_size.width_, 1);
    const std::size_t filter_count = input_depth;
    float_vec bias(filter_count, 0);
    const bool use_bias = data["config"]["use_bias"];
    if (use_bias)
        bias = decode_floats(get_param(name, "bias"));
    assertion(bias.size() == filter_count, "size of bias does not match");
    const bool padding_valid_uses_offset_depth_1 =
        get_global_param("separable_conv2d_valid_offset_depth_1");
    const bool padding_same_uses_offset_depth_1 =
        get_global_param("separable_conv2d_same_offset_depth_1");
        const bool padding_valid_uses_offset_depth_2 =
        get_global_param("separable_conv2d_valid_offset_depth_2");
    const bool padding_same_uses_offset_depth_2 =
        get_global_param("separable_conv2d_same_offset_depth_2");
    return std::make_shared<depthwise_conv_2d_layer>(name, input_depth,
        filter_shape, filter_count, strides, pad_type,
        padding_valid_uses_offset_depth_1, padding_same_uses_offset_depth_1,
        padding_valid_uses_offset_depth_2, padding_same_uses_offset_depth_2,
        dilation_rate, slice_weights, bias);
}

inline layer_ptr create_input_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    assertion(data["inbound_nodes"].empty(),
        "input layer is not allowed to have inbound nodes");
    const auto input_shape = create_shape5_variable(data["config"]["batch_input_shape"]);
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

inline layer_ptr create_concatenate_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    const std::int32_t keras_axis = data["config"]["axis"];
    return std::make_shared<concatenate_layer>(name, keras_axis);
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

inline layer_ptr create_multiply_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<multiply_layer>(name);
}

inline layer_ptr create_average_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<average_layer>(name);
}

inline layer_ptr create_subtract_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json&,
    const std::string& name)
{
    return std::make_shared<subtract_layer>(name);
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

inline layer_ptr create_reshape_layer(
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    const auto target_shape =
        create_vector<int>(create_int, data["config"]["target_shape"]);

    const auto filled_shape =
        fplus::fill_left(1, 3, target_shape);

    return std::make_shared<reshape_layer>(name, filled_shape);
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
    const get_param_f&, const get_global_param_f&, const nlohmann::json& data,
    const std::string& name)
{
    float_type max_value = std::numeric_limits<float_type>::max();
    if (json_obj_has_member(data, "config") &&
        json_obj_has_member(data["config"], "max_value"))
    {
        max_value = data["config"]["max_value"];
    }
    return std::make_shared<relu_layer>(name, max_value);
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

inline layer_ptr create_prelu_layer(
    const get_param_f& get_param, const get_global_param_f&,
    const nlohmann::json& data, const std::string& name)
{
    std::vector<std::size_t> shared_axes;
    if (json_obj_has_member(data, "config") &&
        json_obj_has_member(data["config"], "shared_axes") &&
        !data["config"]["shared_axes"].empty())
    {
        shared_axes = create_vector<std::size_t>(create_size_t,
            data["config"]["shared_axes"]);
    }
    const float_vec alpha = decode_floats(get_param(name, "alpha"));
    return std::make_shared<prelu_layer>(name, alpha, shared_axes);
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

inline layer_ptr create_relu_layer_isolated(
    const get_param_f& get_param, const get_global_param_f& get_global_param,
    const nlohmann::json& data, const std::string& name)
{
    return create_relu_layer(get_param, get_global_param, data, name);
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

inline layer_ptr create_embedding_layer(const get_param_f &get_param,
                                        const get_global_param_f &,
                                        const nlohmann::json &data,
                                        const std::string &name)
{
    const std::size_t input_dim = data["config"]["input_dim"];
    const std::size_t output_dim = data["config"]["output_dim"];
    const float_vec weights = decode_floats(get_param(name, "weights"));

    return std::make_shared<embedding_layer>(name, input_dim, output_dim, weights);
}

inline layer_ptr create_lstm_layer(const get_param_f &get_param,
                                   const get_global_param_f &,
                                   const nlohmann::json &data,
                                   const std::string &name)
{
    const std::size_t units = data["config"]["units"];
    const std::string unit_activation = data["config"]["activation"];
    const std::string recurrent_activation = data["config"]["recurrent_activation"];
    const bool use_bias = data["config"]["use_bias"];

    float_vec bias;
    if (use_bias)
        bias = decode_floats(get_param(name, "bias"));

    const float_vec weights = decode_floats(get_param(name, "weights"));
    const float_vec recurrent_weights = decode_floats(get_param(name, "recurrent_weights"));
    const bool return_sequences = data["config"]["return_sequences"];

    return std::make_shared<lstm_layer>(name, units, unit_activation,
                                        recurrent_activation, use_bias, return_sequences,
                                        weights, recurrent_weights, bias);
}

inline layer_ptr create_gru_layer(const get_param_f &get_param,
                                  const get_global_param_f &,
                                  const nlohmann::json &data,
                                  const std::string &name)
{
    const std::size_t units = data["config"]["units"];
    const std::string unit_activation = data["config"]["activation"];
    const std::string recurrent_activation = data["config"]["recurrent_activation"];
    const bool use_bias = data["config"]["use_bias"];

    float_vec bias;
    if (use_bias)
        bias = decode_floats(get_param(name, "bias"));

    const float_vec weights = decode_floats(get_param(name, "weights"));
    const float_vec recurrent_weights = decode_floats(get_param(name, "recurrent_weights"));

    auto&& config = data["config"];
    const bool reset_after = json_object_get(config, "reset_after", false);
    const bool return_sequences = json_object_get(config, "return_sequences", false);
    
    return std::make_shared<gru_layer>(name, units, unit_activation,
                                       recurrent_activation, use_bias, reset_after, return_sequences,
                                       weights, recurrent_weights, bias);
}

inline layer_ptr create_bidirectional_layer(const get_param_f& get_param,
                                            const get_global_param_f&,
                                            const nlohmann::json& data,
                                            const std::string& name)
{
    const std::string merge_mode = data["config"]["merge_mode"];
    const std::size_t units = data["config"]["layer"]["config"]["units"];
    const std::string unit_activation = data["config"]["layer"]["config"]["activation"];
    const std::string recurrent_activation = data["config"]["layer"]["config"]["recurrent_activation"];
    const bool use_bias = data["config"]["layer"]["config"]["use_bias"];
    const std::string wrapped_layer_type = data["config"]["layer"]["class_name"];
    
    float_vec forward_bias;
    float_vec backward_bias;
    
    if (use_bias)
    {
        forward_bias = decode_floats(get_param(name, "forward_bias"));
        backward_bias = decode_floats(get_param(name, "backward_bias"));
    }
    
    const float_vec forward_weights = decode_floats(get_param(name, "forward_weights"));
    const float_vec backward_weights = decode_floats(get_param(name, "backward_weights"));
    
    const float_vec forward_recurrent_weights = decode_floats(get_param(name, "forward_recurrent_weights"));
    const float_vec backward_recurrent_weights = decode_floats(get_param(name, "backward_recurrent_weights"));
    
    auto&& layer_config = data["config"]["layer"]["config"];
    const bool reset_after = json_object_get(layer_config, "reset_after", false);
    const bool return_sequences = json_object_get(layer_config, "return_sequences", false);
    
    return std::make_shared<bidirectional_layer>(name, merge_mode, units, unit_activation,
                                                 recurrent_activation, wrapped_layer_type,
                                                 use_bias, reset_after, return_sequences,
                                                 forward_weights, forward_recurrent_weights, forward_bias,
                                                 backward_weights, backward_recurrent_weights, backward_bias);
}
    
inline layer_ptr create_time_distributed_layer(const get_param_f& get_param,
                                   const get_global_param_f& get_global_param,
                                   const nlohmann::json& data,
                                   const std::string& name)
{
    const std::string wrapped_layer_type = data["config"]["layer"]["class_name"];
    nlohmann::json data_inner_layer = data["config"]["layer"];
    data_inner_layer["name"] = data["name"];
    data_inner_layer["inbound_nodes"] = data["inbound_nodes"];
    const std::size_t td_input_len = std::size_t(decode_floats(get_param(name, "td_input_len")).front());
    const std::size_t td_output_len = std::size_t(decode_floats(get_param(name, "td_output_len")).front());

    layer_ptr inner_layer = create_layer(get_param, get_global_param, data_inner_layer);
    
    return std::make_shared<time_distributed_layer>(name, inner_layer, td_input_len, td_output_len);
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
            {"SeparableConv1D", create_separable_conv_2D_layer},
            {"SeparableConv2D", create_separable_conv_2D_layer},
            {"DepthwiseConv2D", create_depthwise_conv_2D_layer},
            {"InputLayer", create_input_layer},
            {"BatchNormalization", create_batch_normalization_layer},
            {"Dropout", create_dropout_layer},
            {"LeakyReLU", create_leaky_relu_layer_isolated},
            {"PReLU", create_prelu_layer },
            {"ELU", create_elu_layer_isolated},
            {"ReLU", create_relu_layer_isolated},
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
            {"Concatenate", create_concatenate_layer},
            {"Multiply", create_multiply_layer},
            {"Average", create_average_layer},
            {"Subtract", create_subtract_layer},
            {"Flatten", create_flatten_layer},
            {"ZeroPadding1D", create_zero_padding_2d_layer},
            {"ZeroPadding2D", create_zero_padding_2d_layer},
            {"Cropping1D", create_cropping_2d_layer},
            {"Cropping2D", create_cropping_2d_layer},
            {"Activation", create_activation_layer},
            {"Reshape", create_reshape_layer},
            {"Embedding", create_embedding_layer},
            {"LSTM", create_lstm_layer},
            {"GRU", create_gru_layer},
            {"Bidirectional", create_bidirectional_layer},
            {"TimeDistributed", create_time_distributed_layer},
        };

    const std::string type = data["class_name"];

    auto result = fplus::throw_on_nothing(
        error("unknown layer type: " + type),
        fplus::get_from_map(creators, type))(
            get_param, get_global_param, data, name);

    if (type != "Activation" &&
        json_obj_has_member(data["config"], "activation")
        && type != "GRU"
        && type != "LSTM"
        && type != "Bidirectional")
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
    tensor5s input_;
    tensor5s output_;
};

using test_cases = std::vector<test_case>;

inline test_case load_test_case(const nlohmann::json& data)
{
    assertion(data["inputs"].is_array(), "test needs inputs");
    assertion(data["outputs"].is_array(), "test needs outputs");
    return {
        create_vector<tensor5>(create_tensor5, data["inputs"]),
        create_vector<tensor5>(create_tensor5, data["outputs"])
    };
}

inline test_cases load_test_cases(const nlohmann::json& data)
{
    return create_vector<test_case>(load_test_case, data);
}

inline void check_test_outputs(float_type epsilon,
    const tensor5s& outputs, const tensor5s& targets)
{
    assertion(outputs.size() == targets.size(), "invalid output count");
    for (std::size_t i = 0; i < outputs.size(); ++i)
    {
        const auto& output = outputs[i];
        const auto& target = targets[i];
        assertion(output.shape() == target.shape(), "wrong output size");
        for (std::size_t y = 0; y < output.shape().height_; ++y)
        {
            for (std::size_t x = 0; x < output.shape().width_; ++x)
            {
                for (std::size_t z = 0; z < output.shape().depth_; ++z)
                {
                    if (!fplus::is_in_closed_interval_around(epsilon,
                        target.get(0, 0, y, x, z), output.get(0, 0, y, x, z)))
                    {
                        const std::string msg =
                            std::string("test failed: ") +
                            "output=" + fplus::show(i) + " " +
                            "pos=" +
                            fplus::show(y) + "," +
                            fplus::show(x) + "," +
                            fplus::show(z) + " " +
                            "value=" + fplus::show(output.get(0, 0, y, x, z)) + " "
                            "target=" + fplus::show(target.get(0, 0, y, x, z));
                        internal::raise_error(msg);
                    }
                }
            }
        }
    }
}

} } // namespace fdeep, namespace internal
