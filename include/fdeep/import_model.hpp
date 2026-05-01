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

#include "fdeep/layers/adaptive_pooling_3d_layer.hpp"
#include "fdeep/layers/add_layer.hpp"
#include "fdeep/layers/additive_attention_layer.hpp"
#include "fdeep/layers/attention_layer.hpp"
#include "fdeep/layers/average_layer.hpp"
#include "fdeep/layers/average_pooling_3d_layer.hpp"
#include "fdeep/layers/batch_normalization_layer.hpp"
#include "fdeep/layers/bidirectional_layer.hpp"
#include "fdeep/layers/category_encoding_layer.hpp"
#include "fdeep/layers/celu_layer.hpp"
#include "fdeep/layers/centercrop_layer.hpp"
#include "fdeep/layers/concatenate_layer.hpp"
#include "fdeep/layers/conv_2d_layer.hpp"
#include "fdeep/layers/conv_2d_transpose_layer.hpp"
#include "fdeep/layers/conv_3d_layer.hpp"
#include "fdeep/layers/conv_3d_transpose_layer.hpp"
#include "fdeep/layers/conv_lstm_2d_layer.hpp"
#include "fdeep/layers/conv_lstm_3d_layer.hpp"
#include "fdeep/layers/cropping_3d_layer.hpp"
#include "fdeep/layers/dense_layer.hpp"
#include "fdeep/layers/depthwise_conv_2d_layer.hpp"
#include "fdeep/layers/discretization_layer.hpp"
#include "fdeep/layers/dot_layer.hpp"
#include "fdeep/layers/einsum_dense_layer.hpp"
#include "fdeep/layers/elu_layer.hpp"
#include "fdeep/layers/embedding_layer.hpp"
#include "fdeep/layers/exponential_layer.hpp"
#include "fdeep/layers/flatten_layer.hpp"
#include "fdeep/layers/gelu_layer.hpp"
#include "fdeep/layers/global_average_pooling_3d_layer.hpp"
#include "fdeep/layers/global_max_pooling_3d_layer.hpp"
#include "fdeep/layers/group_normalization_layer.hpp"
#include "fdeep/layers/group_query_attention_layer.hpp"
#include "fdeep/layers/gru_layer.hpp"
#include "fdeep/layers/hard_shrink_layer.hpp"
#include "fdeep/layers/hard_sigmoid_layer.hpp"
#include "fdeep/layers/hard_tanh_layer.hpp"
#include "fdeep/layers/input_layer.hpp"
#include "fdeep/layers/integer_lookup_layer.hpp"
#include "fdeep/layers/layer.hpp"
#include "fdeep/layers/layer_normalization_layer.hpp"
#include "fdeep/layers/leaky_relu_layer.hpp"
#include "fdeep/layers/linear_layer.hpp"
#include "fdeep/layers/log_sigmoid_layer.hpp"
#include "fdeep/layers/log_softmax_layer.hpp"
#include "fdeep/layers/lstm_layer.hpp"
#include "fdeep/layers/max_pooling_3d_layer.hpp"
#include "fdeep/layers/maximum_layer.hpp"
#include "fdeep/layers/minimum_layer.hpp"
#include "fdeep/layers/model_layer.hpp"
#include "fdeep/layers/multi_head_attention_layer.hpp"
#include "fdeep/layers/multiply_layer.hpp"
#include "fdeep/layers/normalization_layer.hpp"
#include "fdeep/layers/permute_layer.hpp"
#include "fdeep/layers/pooling_3d_layer.hpp"
#include "fdeep/layers/prelu_layer.hpp"
#include "fdeep/layers/relu_layer.hpp"
#include "fdeep/layers/repeat_vector_layer.hpp"
#include "fdeep/layers/rescaling_layer.hpp"
#include "fdeep/layers/reshape_layer.hpp"
#include "fdeep/layers/resizing_layer.hpp"
#include "fdeep/layers/rms_normalization_layer.hpp"
#include "fdeep/layers/selu_layer.hpp"
#include "fdeep/layers/separable_conv_2d_layer.hpp"
#include "fdeep/layers/sigmoid_layer.hpp"
#include "fdeep/layers/simple_rnn_layer.hpp"
#include "fdeep/layers/soft_shrink_layer.hpp"
#include "fdeep/layers/softmax_layer.hpp"
#include "fdeep/layers/softplus_layer.hpp"
#include "fdeep/layers/stacked_rnn_layer.hpp"
#include "fdeep/layers/softsign_layer.hpp"
#include "fdeep/layers/sparse_plus_layer.hpp"
#include "fdeep/layers/square_plus_layer.hpp"
#include "fdeep/layers/subtract_layer.hpp"
#include "fdeep/layers/swish_layer.hpp"
#include "fdeep/layers/tanh_layer.hpp"
#include "fdeep/layers/tanh_shrink_layer.hpp"
#include "fdeep/layers/threshold_layer.hpp"
#include "fdeep/layers/time_distributed_layer.hpp"
#include "fdeep/layers/unit_normalization_layer.hpp"
#include "fdeep/layers/upsampling_1d_layer.hpp"
#include "fdeep/layers/upsampling_2d_layer.hpp"
#include "fdeep/layers/upsampling_3d_layer.hpp"
#include "fdeep/layers/zero_padding_3d_layer.hpp"
#include "fdeep/tensor.hpp"
#include "fdeep/tensor_shape.hpp"
#include "fdeep/tensor_shape_variable.hpp"

#include <fplus/fplus.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fdeep {
namespace internal {

    template <typename KeyT, typename ValueT>
    ValueT json_object_get(const nlohmann::json& data, KeyT&& key, ValueT&& default_value)
    {
        auto&& it = data.find(key);
        if (it != data.end())
            return *it;
        else
            return std::forward<ValueT>(default_value);
    }

    inline bool json_obj_has_member(const nlohmann::json& data,
        const std::string& member_name)
    {
        return data.is_object() && data.find(member_name) != data.end();
    }

    inline fplus::maybe<std::size_t> create_maybe_size_t(const nlohmann::json& data)
    {
        if (data.is_null()) {
            return fplus::nothing<std::size_t>();
        }
        const int signed_result = data;
        if (signed_result < 0) {
            return fplus::nothing<std::size_t>();
        }
        const std::size_t result = data;
        return fplus::just(result);
    }

    inline tensor_shape_variable create_tensor_shape_variable_offset(
        const nlohmann::json& data, std::size_t offset)
    {
        assertion(data.is_array(), "tensor_shape_variable needs to be an array");
        assertion(data.size() > 0, "need at least one dimension");
        if (data.size() == 1 + offset)
            return tensor_shape_variable(
                create_maybe_size_t(data[0 + offset]));
        if (data.size() == 2 + offset)
            return tensor_shape_variable(
                create_maybe_size_t(data[0 + offset]),
                create_maybe_size_t(data[1 + offset]));
        if (data.size() == 3 + offset)
            return tensor_shape_variable(
                create_maybe_size_t(data[0 + offset]),
                create_maybe_size_t(data[1 + offset]),
                create_maybe_size_t(data[2 + offset]));
        if (data.size() == 4 + offset)
            return tensor_shape_variable(
                create_maybe_size_t(data[0 + offset]),
                create_maybe_size_t(data[1 + offset]),
                create_maybe_size_t(data[2 + offset]),
                create_maybe_size_t(data[3 + offset]));
        if (data.size() == 5 + offset)
            return tensor_shape_variable(
                create_maybe_size_t(data[0 + offset]),
                create_maybe_size_t(data[1 + offset]),
                create_maybe_size_t(data[2 + offset]),
                create_maybe_size_t(data[3 + offset]),
                create_maybe_size_t(data[4 + offset]));

        raise_error("tensor_shape_variable needs 1, 2, 3, 4 or 5 dimensions");
        return tensor_shape_variable(
            fplus::nothing<std::size_t>(),
            fplus::nothing<std::size_t>(),
            fplus::nothing<std::size_t>(),
            fplus::nothing<std::size_t>(),
            fplus::nothing<std::size_t>()); // Is never called
    }

    inline tensor_shape_variable create_tensor_shape_variable(const nlohmann::json& data)
    {
        return create_tensor_shape_variable_offset(data, 0);
    }

    inline tensor_shape_variable create_tensor_shape_variable_leading_null(const nlohmann::json& data)
    {
        return create_tensor_shape_variable_offset(data, 1);
    }

    inline tensor_shape create_tensor_shape(const nlohmann::json& data)
    {
        assertion(data.is_array(), "tensor_shape needs to be an array");
        assertion(data.size() > 0, "need at least one dimension");
        if (data.size() == 1)
            return tensor_shape(static_cast<std::size_t>(data[0]));
        if (data.size() == 2)
            return tensor_shape(data[0], data[1]);
        if (data.size() == 3)
            return tensor_shape(data[0], data[1], data[2]);
        if (data.size() == 4)
            return tensor_shape(data[0], data[1], data[2], data[3]);
        if (data.size() == 5)
            return tensor_shape(data[0], data[1], data[2], data[3], data[4]);
        raise_error("tensor_shape needs 1, 2, 3, 4 or 5 dimensions");
        return tensor_shape(static_cast<std::size_t>(0)); // Is never be called
    }

    inline shape2 create_shape2(const nlohmann::json& data)
    {
        if (data.is_array()) {
            assertion(data.size() == 1 || data.size() == 2,
                "invalid number of dimensions in shape2");
            if (data.size() == 1)
                return shape2(1, data[0]);
            else
                return shape2(data[0], data[1]);
        } else {
            const std::size_t width = data;
            return shape2(1, width);
        }
    }

    inline shape3 create_shape3(const nlohmann::json& data)
    {
        if (data.is_array()) {
            assertion(data.size() == 1 || data.size() == 2 || data.size() == 3,
                "invalid number of dimensions in shape2");
            if (data.size() == 1)
                return shape3(1, 1, data[0]);
            if (data.size() == 2)
                return shape3(1, data[0], data[1]);
            else
                return shape3(data[0], data[1], data[2]);
        } else {
            const std::size_t width = data;
            return shape3(1, 1, width);
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

        if (data.is_array() && !data.empty() && data[0].is_number()) {
            const float_vec result = data;
            return result;
        }

        assertion(std::numeric_limits<float>::is_iec559,
            "The floating-point format of your system is not supported.");

        const auto res = Base64_decode(json_data_strs_char_prodiver(data, '='));
        float_vec out;
        assertion(res.size() % 4 == 0, "invalid float vector data");
        out.reserve(res.size() / 4);
        for (std::size_t i = 0; i < res.size(); i += 4) {
            float_type val = static_cast<float_type>(
                *(reinterpret_cast<const float*>(&(res[i]))));
            out.push_back(val);
        }
        return out;
    }

    inline std::string get_activation_type(const nlohmann::json& data)
    {
        assertion(data.is_string(), "Layer activation must be a string.");
        return data;
    }

    inline std::string json_object_get_activation_with_default(const nlohmann::json& config,
        const std::string& default_activation)
    {
        if (json_obj_has_member(config, "activation")) {
            return get_activation_type(config["activation"]);
        }
        return default_activation;
    }

    inline std::string json_object_get_named_activation_with_default(const nlohmann::json& config,
        const std::string& key, const std::string& default_activation)
    {
        if (json_obj_has_member(config, key)) {
            return get_activation_type(config[key]);
        }
        return default_activation;
    }

    inline tensor create_tensor(const nlohmann::json& data)
    {
        const tensor_shape shape = create_tensor_shape(data["shape"]);
        return tensor(shape, decode_floats(data["values"]));
    }

    template <typename T, typename F>
    std::vector<T> create_vector(F f, const nlohmann::json& data)
    {
        if (data.is_array())
            return fplus::transform_convert<std::vector<T>>(f, data);
        else
            return fplus::singleton_seq(f(data));
    }

    inline std::vector<tensor_shape_variable> create_tensor_shapes_variable(const nlohmann::json& data)
    {
        return create_vector<tensor_shape_variable>(create_tensor_shape_variable, data);
    }

    inline node_connection create_node_connection_model_layer(const nlohmann::json& data)
    {
        assertion(data.is_array(), "invalid format for inbound node");
        const std::string layer_id = data.front();
        const auto node_idx = create_size_t(data[1]);
        const auto tensor_idx = create_size_t(data[2]);
        return node_connection(layer_id, node_idx, tensor_idx);
    }

    inline std::vector<node_connection> create_node_connections_model_layer(const nlohmann::json& data)
    {
        assertion(data.is_array(), "input_layers/output_layers must be an array");
        // Keras serializes a single connection as a flat triple ["layer", node_idx, tensor_idx],
        // and multiple connections as a list of such triples.
        if (!data.empty() && !data.front().is_array())
            return { create_node_connection_model_layer(data) };
        return create_vector<node_connection>(create_node_connection_model_layer, data);
    }

    inline node_connection create_node_connection(const nlohmann::json& args)
    {
        assertion(json_obj_has_member(args["config"], "keras_history"),
            "No keras_history on node connection. Constant-value tensors are not supported.");
        const std::vector<nlohmann::json> keras_history = args["config"]["keras_history"];
        assertion(keras_history.size() >= 3, "invalid number of items in keras_history");
        const std::string layer_id = keras_history[0];
        const auto node_idx = create_size_t(keras_history[1]);
        const auto tensor_idx = create_size_t(keras_history[2]);
        return node_connection(layer_id, node_idx, tensor_idx);
    }

    using get_param_f = std::function<nlohmann::json(const std::string&, const std::string&)>;

    using layer_creators = std::map<
        std::string,
        std::function<layer_ptr(
            const get_param_f&,
            const nlohmann::json&,
            const std::string&)>>;

    using wrapper_layer_creators = std::map<
        std::string,
        std::function<layer_ptr(
            const get_param_f&,
            const nlohmann::json&,
            const std::string&,
            const layer_creators&,
            const std::string)>>;

    layer_ptr create_layer(const get_param_f&,
        const nlohmann::json&,
        const layer_creators& custom_layer_creators,
        const std::string&);

    inline layer_ptr create_model_layer(const get_param_f& get_param,
        const nlohmann::json& data,
        const std::string& name, const layer_creators& custom_layer_creators,
        const std::string& prefix)
    {
        assertion(data["config"]["layers"].is_array(), "missing layers array");

        const std::function<nlohmann::json(
            const std::string&, const std::string&)>
            get_prefixed_param = [&](const std::string& layer_name, const std::string& param_name)
            -> nlohmann::json {
            return get_param(prefix + layer_name, param_name);
        };

        const auto make_layer = [&](const nlohmann::json& json) {
            return create_layer(get_prefixed_param, json,
                custom_layer_creators, prefix);
        };
        const auto layers = create_vector<layer_ptr>(make_layer,
            data["config"]["layers"]);

        assertion(data["config"]["input_layers"].is_array(), "no input layers");

        const auto inputs = create_node_connections_model_layer(data["config"]["input_layers"]);

        const auto outputs = create_node_connections_model_layer(data["config"]["output_layers"]);

        return std::make_shared<model_layer>(name, layers, inputs, outputs);
    }

    inline padding create_padding(const std::string& padding_str)
    {
        return fplus::throw_on_nothing(error("no padding"),
            fplus::choose<std::string, padding>({
                                                    { std::string("valid"), padding::valid },
                                                    { std::string("same"), padding::same },
                                                    { std::string("causal"), padding::causal },
                                                },
                padding_str));
    }

    inline layer_ptr create_conv_2d_layer(const get_param_f& get_param,
        const nlohmann::json& data,
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
        const std::size_t filter_depths = weights.size() / (kernel_size.area() * filter_count);
        const tensor_shape filter_shape(
            kernel_size.height_, kernel_size.width_, filter_depths);

        return std::make_shared<conv_2d_layer>(name,
            filter_shape, filter_count, strides, pad_type,
            dilation_rate, weights, bias);
    }

    inline layer_ptr create_conv_3d_layer(const get_param_f& get_param,
        const nlohmann::json& data,
        const std::string& name)
    {
        const std::string padding_str = data["config"]["padding"];
        const auto pad_type = create_padding(padding_str);

        const shape3 strides = create_shape3(data["config"]["strides"]);
        const shape3 dilation_rate = create_shape3(data["config"]["dilation_rate"]);

        const auto filter_count = create_size_t(data["config"]["filters"]);
        float_vec bias(filter_count, 0);
        const bool use_bias = data["config"]["use_bias"];
        if (use_bias)
            bias = decode_floats(get_param(name, "bias"));
        assertion(bias.size() == filter_count, "size of bias does not match");

        const float_vec weights = decode_floats(get_param(name, "weights"));
        const shape3 kernel_size = create_shape3(data["config"]["kernel_size"]);
        assertion(weights.size() % kernel_size.volume() == 0,
            "invalid number of weights");
        const std::size_t filter_depths = weights.size() / (kernel_size.volume() * filter_count);
        const tensor_shape filter_shape(
            kernel_size.size_dim_4_, kernel_size.height_, kernel_size.width_, filter_depths);

        return std::make_shared<conv_3d_layer>(name,
            filter_shape, filter_count, strides, pad_type,
            dilation_rate, weights, bias);
    }

    inline layer_ptr create_conv_3d_transpose_layer(const get_param_f& get_param,
        const nlohmann::json& data,
        const std::string& name)
    {
        const std::string padding_str = data["config"]["padding"];
        const auto pad_type = create_padding(padding_str);

        const shape3 strides = create_shape3(data["config"]["strides"]);
        const shape3 dilation_rate = create_shape3(data["config"]["dilation_rate"]);

        const auto filter_count = create_size_t(data["config"]["filters"]);
        float_vec bias(filter_count, 0);
        const bool use_bias = data["config"]["use_bias"];
        if (use_bias)
            bias = decode_floats(get_param(name, "bias"));
        assertion(bias.size() == filter_count, "size of bias does not match");

        const float_vec weights = decode_floats(get_param(name, "weights"));
        const shape3 kernel_size = create_shape3(data["config"]["kernel_size"]);
        assertion(weights.size() % kernel_size.volume() == 0,
            "invalid number of weights");
        const std::size_t filter_depths = weights.size() / (kernel_size.volume() * filter_count);
        const tensor_shape filter_shape(
            kernel_size.size_dim_4_, kernel_size.height_, kernel_size.width_, filter_depths);

        return std::make_shared<conv_3d_transpose_layer>(name,
            filter_shape, filter_count, strides, pad_type,
            dilation_rate, weights, bias);
    }

    inline layer_ptr create_conv_2d_transpose_layer(const get_param_f& get_param,
        const nlohmann::json& data,
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
        const std::size_t filter_depths = weights.size() / (kernel_size.area() * filter_count);
        const tensor_shape filter_shape(
            kernel_size.height_, kernel_size.width_, filter_depths);

        return std::make_shared<conv_2d_transpose_layer>(name,
            filter_shape, filter_count, strides, pad_type,
            dilation_rate, weights, bias);
    }

    inline layer_ptr create_separable_conv_2D_layer(const get_param_f& get_param,
        const nlohmann::json& data,
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
        const std::size_t stack_output_depths_1 = stack_weights.size() / input_depth;
        assertion(stack_output_depths_1 == filter_count, "invalid weights sizes");
        const tensor_shape filter_shape(kernel_size.height_, kernel_size.width_, 1);
        float_vec bias_0(input_depth, 0);
        return std::make_shared<separable_conv_2d_layer>(name, input_depth,
            filter_shape, filter_count, strides, pad_type,
            dilation_rate, slice_weights, stack_weights, bias_0, bias);
    }

    inline layer_ptr create_depthwise_conv_2D_layer(const get_param_f& get_param,
        const nlohmann::json& data,
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
        const tensor_shape filter_shape(kernel_size.height_, kernel_size.width_, 1);
        float_vec bias(input_depth, 0);
        const bool use_bias = data["config"]["use_bias"];
        if (use_bias)
            bias = decode_floats(get_param(name, "bias"));
        assertion(bias.size() == input_depth, "size of bias does not match");
        return std::make_shared<depthwise_conv_2d_layer>(name, input_depth,
            filter_shape, strides, pad_type,
            dilation_rate, slice_weights, bias);
    }

    inline layer_ptr create_input_layer(
        const get_param_f&, const nlohmann::json& data, const std::string& name)
    {
        assertion(data["inbound_nodes"].empty(),
            "input layer is not allowed to have inbound nodes");
        const auto input_shape = create_tensor_shape_variable_leading_null(data["config"]["batch_shape"]);
        return std::make_shared<input_layer>(name, input_shape);
    }

    inline layer_ptr create_batch_normalization_layer(const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        const float_vec moving_mean = decode_floats(get_param(name, "moving_mean"));
        const float_vec moving_variance = decode_floats(get_param(name, "moving_variance"));
        const bool center = data["config"]["center"];
        const bool scale = data["config"]["scale"];
        const auto axis_vec = create_vector<int>(create_int, data["config"]["axis"]);
        assertion(axis_vec.size() == 1, "invalid axis configuration");
        const int axis = axis_vec.front();
        const float_type epsilon = data["config"]["epsilon"];
        float_vec gamma;
        float_vec beta;
        if (scale)
            gamma = decode_floats(get_param(name, "gamma"));
        if (center)
            beta = decode_floats(get_param(name, "beta"));
        return std::make_shared<batch_normalization_layer>(
            name, axis, moving_mean, moving_variance, beta, gamma, epsilon);
    }

    inline layer_ptr create_layer_normalization_layer(const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        const bool center = data["config"]["center"];
        const bool scale = data["config"]["scale"];
        const auto axes = create_vector<int>(create_int, data["config"]["axis"]);
        const float_type epsilon = data["config"]["epsilon"];
        float_vec gamma;
        float_vec beta;
        if (scale)
            gamma = decode_floats(get_param(name, "gamma"));
        if (center)
            beta = decode_floats(get_param(name, "beta"));
        return std::make_shared<layer_normalization_layer>(
            name, axes, beta, gamma, epsilon);
    }

    inline layer_ptr create_rms_normalization_layer(const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        const auto axes = create_vector<int>(create_int, data["config"]["axis"]);
        const float_type epsilon = data["config"]["epsilon"];
        const float_vec scale = decode_floats(get_param(name, "scale"));
        return std::make_shared<rms_normalization_layer>(
            name, axes, scale, epsilon);
    }

    inline layer_ptr create_adaptive_avg_pooling_layer(const get_param_f&,
        const nlohmann::json& data, const std::string& name)
    {
        const auto& sz = data["config"]["output_size"];
        std::vector<std::size_t> dims;
        if (sz.is_array())
            for (const auto& v : sz)
                dims.push_back(static_cast<std::size_t>(v));
        else
            dims.push_back(static_cast<std::size_t>(sz));
        const std::size_t d4 = dims.size() >= 3 ? dims[dims.size() - 3] : 1;
        const std::size_t h = dims.size() >= 2 ? dims[dims.size() - 2] : 1;
        const std::size_t w = dims.back();
        return std::make_shared<adaptive_pooling_3d_layer>(name, d4, h, w,
            adaptive_pooling_kind::avg);
    }

    inline layer_ptr create_adaptive_max_pooling_layer(const get_param_f&,
        const nlohmann::json& data, const std::string& name)
    {
        const auto& sz = data["config"]["output_size"];
        std::vector<std::size_t> dims;
        if (sz.is_array())
            for (const auto& v : sz)
                dims.push_back(static_cast<std::size_t>(v));
        else
            dims.push_back(static_cast<std::size_t>(sz));
        const std::size_t d4 = dims.size() >= 3 ? dims[dims.size() - 3] : 1;
        const std::size_t h = dims.size() >= 2 ? dims[dims.size() - 2] : 1;
        const std::size_t w = dims.back();
        return std::make_shared<adaptive_pooling_3d_layer>(name, d4, h, w,
            adaptive_pooling_kind::max);
    }

    inline layer_ptr create_conv_lstm_3d_layer(const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        auto&& cfg = data["config"];
        const std::size_t units = cfg["filters"];
        const shape3 strides = create_shape3(cfg["strides"]);
        const shape3 dilation_rate = create_shape3(cfg["dilation_rate"]);
        const std::string padding_str = cfg["padding"];
        const auto pad_type = create_padding(padding_str);
        const shape3 kernel_size = create_shape3(cfg["kernel_size"]);
        const std::string activation = json_object_get_activation_with_default(cfg, "tanh");
        const std::string recurrent_activation = json_object_get_named_activation_with_default(
            cfg, "recurrent_activation", "sigmoid");
        const bool use_bias = json_object_get(cfg, "use_bias", true);
        const bool return_sequences = json_object_get(cfg, "return_sequences", false);
        const bool return_state = json_object_get(cfg, "return_state", false);

        const float_vec weights = decode_floats(get_param(name, "weights"));
        const float_vec recurrent_weights = decode_floats(get_param(name, "recurrent_weights"));
        float_vec bias(units * 4, 0);
        if (use_bias)
            bias = decode_floats(get_param(name, "bias"));

        const std::size_t kernel_volume = kernel_size.size_dim_4_
            * kernel_size.height_ * kernel_size.width_;
        const std::size_t in_c = weights.size() / (kernel_volume * units * 4);
        const tensor_shape filter_shape(kernel_size.size_dim_4_,
            kernel_size.height_, kernel_size.width_, in_c);

        return std::make_shared<conv_lstm_3d_layer>(name, units,
            filter_shape, strides, pad_type, dilation_rate,
            weights, recurrent_weights, bias,
            activation, recurrent_activation,
            return_sequences, return_state);
    }

    inline layer_ptr create_conv_lstm_2d_layer(const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        auto&& cfg = data["config"];
        const std::string class_name = data["class_name"];
        const std::size_t rank = (class_name == "ConvLSTM1D") ? 1 : 2;
        assertion(class_name == "ConvLSTM1D" || class_name == "ConvLSTM2D",
            "create_conv_lstm_2d_layer: unsupported layer class.");
        const std::size_t units = cfg["filters"];
        const shape2 strides = create_shape2(cfg["strides"]);
        const shape2 dilation_rate = create_shape2(cfg["dilation_rate"]);
        const std::string padding_str = cfg["padding"];
        const auto pad_type = create_padding(padding_str);
        const shape2 kernel_size = create_shape2(cfg["kernel_size"]);
        const std::string activation = json_object_get_activation_with_default(cfg, "tanh");
        const std::string recurrent_activation = json_object_get_named_activation_with_default(
            cfg, "recurrent_activation", "sigmoid");
        const bool use_bias = json_object_get(cfg, "use_bias", true);
        const bool return_sequences = json_object_get(cfg, "return_sequences", false);
        const bool return_state = json_object_get(cfg, "return_state", false);

        const float_vec weights = decode_floats(get_param(name, "weights"));
        const float_vec recurrent_weights = decode_floats(get_param(name, "recurrent_weights"));
        float_vec bias(units * 4, 0);
        if (use_bias)
            bias = decode_floats(get_param(name, "bias"));

        // Determine in_channels from weight count.
        // Input weights: (k_h, k_w, in_c, units*4).
        const std::size_t in_c = weights.size() / (kernel_size.area() * units * 4);
        const tensor_shape filter_shape(kernel_size.height_, kernel_size.width_, in_c);

        return std::make_shared<conv_lstm_2d_layer>(name, units, rank,
            filter_shape, strides, pad_type, dilation_rate,
            weights, recurrent_weights, bias,
            activation, recurrent_activation,
            return_sequences, return_state);
    }

    inline layer_ptr create_discretization_layer(const get_param_f&,
        const nlohmann::json& data, const std::string& name)
    {
        const std::string output_mode = data["config"]["output_mode"];
        assertion(output_mode == "int",
            "Discretization only supports output_mode='int'.");
        std::vector<float_type> boundaries;
        for (const auto& v : data["config"]["bin_boundaries"])
            boundaries.push_back(static_cast<float_type>(v));
        return std::make_shared<discretization_layer>(name, boundaries);
    }

    inline layer_ptr create_integer_lookup_layer(const get_param_f&,
        const nlohmann::json& data, const std::string& name)
    {
        auto&& cfg = data["config"];
        const std::string output_mode = cfg["output_mode"];
        assertion(output_mode == "int",
            "IntegerLookup only supports output_mode='int'.");
        assertion(!cfg.value("invert", false),
            "IntegerLookup with invert=True is not supported.");
        const std::size_t num_oov_indices = cfg["num_oov_indices"];
        const bool has_mask_token = !cfg["mask_token"].is_null();
        const std::int64_t mask_token = has_mask_token
            ? static_cast<std::int64_t>(cfg["mask_token"]) : 0;
        std::vector<std::int64_t> vocabulary;
        for (const auto& v : cfg["vocabulary"])
            vocabulary.push_back(static_cast<std::int64_t>(v));
        return std::make_shared<integer_lookup_layer>(name, vocabulary,
            num_oov_indices, has_mask_token, mask_token);
    }

    inline layer_ptr create_einsum_dense_layer(const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        auto&& config = data["config"];
        const std::string equation = config["equation"];
        const std::string bias_axes = config["bias_axes"].is_null()
            ? std::string("")
            : std::string(config["bias_axes"]);
        std::vector<int> output_shape;
        for (const auto& dim : config["output_shape"])
            output_shape.push_back(dim.is_null() ? -1 : static_cast<int>(dim));

        const auto kernel_shape = create_vector<std::size_t>(create_size_t, get_param(name, "kernel_shape"));
        auto kernel_values = decode_floats(get_param(name, "kernel"));
        const tensor kernel_tensor(create_tensor_shape_from_dims(kernel_shape), std::move(kernel_values));

        tensor bias_tensor(tensor_shape(static_cast<std::size_t>(1)), float_type(0));
        if (!bias_axes.empty()) {
            const auto bias_shape = create_vector<std::size_t>(create_size_t, get_param(name, "bias_shape"));
            auto bias_values = decode_floats(get_param(name, "bias"));
            bias_tensor = tensor(create_tensor_shape_from_dims(bias_shape), std::move(bias_values));
        }

        return std::make_shared<einsum_dense_layer>(name, equation,
            output_shape, bias_axes, kernel_tensor, bias_tensor);
    }

    inline layer_ptr create_group_normalization_layer(const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        const std::size_t groups = data["config"]["groups"];
        const int axis = data["config"]["axis"];
        const float_type epsilon = data["config"]["epsilon"];
        const bool center = data["config"]["center"];
        const bool scale = data["config"]["scale"];
        float_vec gamma;
        float_vec beta;
        if (scale)
            gamma = decode_floats(get_param(name, "gamma"));
        if (center)
            beta = decode_floats(get_param(name, "beta"));
        return std::make_shared<group_normalization_layer>(
            name, groups, axis, epsilon, beta, gamma);
    }

    inline layer_ptr create_unit_normalization_layer(const get_param_f&,
        const nlohmann::json& data, const std::string& name)
    {
        const auto axes = create_vector<int>(create_int, data["config"]["axis"]);
        return std::make_shared<unit_normalization_layer>(name, axes);
    }

    inline layer_ptr create_identity_layer(
        const get_param_f&, const nlohmann::json&, const std::string& name)
    {
        // Dropout and noise layers are identity functions during prediction.
        return std::make_shared<linear_layer>(name);
    }

    inline layer_ptr create_max_pooling_3d_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const auto pool_size = create_shape3(data["config"]["pool_size"]);
        const auto strides = create_shape3(data["config"]["strides"]);
        const std::string padding_str = data["config"]["padding"];
        const auto pad_type = create_padding(padding_str);
        return std::make_shared<max_pooling_3d_layer>(name,
            pool_size, strides, pad_type);
    }

    inline layer_ptr create_average_pooling_3d_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const auto pool_size = create_shape3(data["config"]["pool_size"]);
        const auto strides = create_shape3(data["config"]["strides"]);
        const std::string padding_str = data["config"]["padding"];

        const auto pad_type = create_padding(padding_str);
        return std::make_shared<average_pooling_3d_layer>(name,
            pool_size, strides, pad_type);
    }

    inline layer_ptr create_global_max_pooling_3d_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const bool keepdims = data["config"]["keepdims"];
        return std::make_shared<global_max_pooling_3d_layer>(name, keepdims);
    }

    inline layer_ptr create_global_average_pooling_3d_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const bool keepdims = data["config"]["keepdims"];
        return std::make_shared<global_average_pooling_3d_layer>(name, keepdims);
    }

    inline layer_ptr create_upsampling_1d_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const std::size_t size = data["config"]["size"];
        return std::make_shared<upsampling_1d_layer>(name, size);
    }

    inline layer_ptr create_upsampling_2d_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const auto scale_factor = create_shape2(data["config"]["size"]);
        const std::string interpolation = data["config"]["interpolation"];
        return std::make_shared<upsampling_2d_layer>(
            name, scale_factor, interpolation);
    }

    inline layer_ptr create_upsampling_3d_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const auto scale_factor = create_shape3(data["config"]["size"]);
        return std::make_shared<upsampling_3d_layer>(
            name, scale_factor);
    }

    inline layer_ptr create_dense_layer(const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
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
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const int keras_axis = data["config"]["axis"];
        return std::make_shared<concatenate_layer>(name, keras_axis);
    }

    inline layer_ptr create_add_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<add_layer>(name);
    }

    inline layer_ptr create_maximum_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<maximum_layer>(name);
    }

    inline layer_ptr create_minimum_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<minimum_layer>(name);
    }

    inline layer_ptr create_dot_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const auto axes = create_vector<int>(create_int, data["config"]["axes"]);
        const bool normalize = data["config"]["normalize"];
        return std::make_shared<dot_layer>(name, axes, normalize);
    }

    inline layer_ptr create_multiply_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<multiply_layer>(name);
    }

    inline layer_ptr create_average_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<average_layer>(name);
    }

    inline layer_ptr create_subtract_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<subtract_layer>(name);
    }

    inline layer_ptr create_flatten_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<flatten_layer>(name);
    }

    inline layer_ptr create_zero_padding_3d_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const auto padding = create_vector<std::vector<std::size_t>>(fplus::bind_1st_of_2(
                                                                         create_vector<std::size_t, decltype(create_size_t)>, create_size_t),
            data["config"]["padding"]);

        assertion(
            (padding.size() == 2 && padding[0].size() == padding[1].size()) || (padding.size() == 3 && padding[0].size() == padding[1].size() && padding[1].size() == padding[2].size()),
            "invalid padding format");

        if (padding[0].size() == 1) {
            const std::size_t front_pad = 0;
            const std::size_t back_pad = 0;
            const std::size_t top_pad = 0;
            const std::size_t bottom_pad = 0;
            const std::size_t left_pad = padding[0][0];
            const std::size_t right_pad = padding[1][0];
            return std::make_shared<zero_padding_3d_layer>(name,
                front_pad, back_pad, top_pad, bottom_pad, left_pad, right_pad);
        }
        if (padding.size() == 2) {
            const std::size_t front_pad = 0;
            const std::size_t back_pad = 0;
            const std::size_t top_pad = padding[0][0];
            const std::size_t bottom_pad = padding[0][1];
            const std::size_t left_pad = padding[1][0];
            const std::size_t right_pad = padding[1][1];
            return std::make_shared<zero_padding_3d_layer>(name,
                front_pad, back_pad, top_pad, bottom_pad, left_pad, right_pad);
        } else {
            const std::size_t front_pad = padding[0][0];
            const std::size_t back_pad = padding[0][1];
            const std::size_t top_pad = padding[1][0];
            const std::size_t bottom_pad = padding[1][1];
            const std::size_t left_pad = padding[2][0];
            const std::size_t right_pad = padding[2][1];
            return std::make_shared<zero_padding_3d_layer>(name,
                front_pad, back_pad, top_pad, bottom_pad, left_pad, right_pad);
        }
    }

    inline layer_ptr create_cropping_3d_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const auto cropping = create_vector<std::vector<std::size_t>>(fplus::bind_1st_of_2(
                                                                          create_vector<std::size_t, decltype(create_size_t)>, create_size_t),
            data["config"]["cropping"]);

        assertion(
            (cropping.size() == 2 && cropping[0].size() == cropping[1].size()) || (cropping.size() == 3 && cropping[0].size() == cropping[1].size() && cropping[1].size() == cropping[2].size()),
            "invalid cropping format");

        if (cropping[0].size() == 1) {
            const std::size_t front_crop = 0;
            const std::size_t back_crop = 0;
            const std::size_t top_crop = 0;
            const std::size_t bottom_crop = 0;
            const std::size_t left_crop = cropping[0][0];
            const std::size_t right_crop = cropping[1][0];
            return std::make_shared<cropping_3d_layer>(name,
                front_crop, back_crop, top_crop, bottom_crop, left_crop, right_crop);
        }
        if (cropping.size() == 2) {
            const std::size_t front_crop = 0;
            const std::size_t back_crop = 0;
            const std::size_t top_crop = cropping[0][0];
            const std::size_t bottom_crop = cropping[0][1];
            const std::size_t left_crop = cropping[1][0];
            const std::size_t right_crop = cropping[1][1];
            return std::make_shared<cropping_3d_layer>(name,
                front_crop, back_crop, top_crop, bottom_crop, left_crop, right_crop);
        } else {
            const std::size_t front_crop = cropping[0][0];
            const std::size_t back_crop = cropping[0][1];
            const std::size_t top_crop = cropping[1][0];
            const std::size_t bottom_crop = cropping[1][1];
            const std::size_t left_crop = cropping[2][0];
            const std::size_t right_crop = cropping[2][1];
            return std::make_shared<cropping_3d_layer>(name,
                front_crop, back_crop, top_crop, bottom_crop, left_crop, right_crop);
        }
    }

    inline layer_ptr create_centercrop_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const std::size_t height = data["config"]["height"];
        const std::size_t width = data["config"]["width"];
        return std::make_shared<centercrop_layer>(name, height, width);
    }

    inline layer_ptr create_repeat_vector_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const std::size_t n = data["config"]["n"];
        return std::make_shared<repeat_vector_layer>(name, n);
    }

    inline layer_ptr create_rescaling_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const float_type scale = data["config"]["scale"];
        const float_type offset = data["config"]["offset"];
        return std::make_shared<rescaling_layer>(name, scale, offset);
    }

    inline layer_ptr create_reshape_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const auto target_shape = create_tensor_shape_variable(data["config"]["target_shape"]);
        return std::make_shared<reshape_layer>(name, target_shape);
    }

    inline layer_ptr create_resizing_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        const std::size_t height = data["config"]["height"];
        const std::size_t width = data["config"]["width"];
        const std::string interpolation = data["config"]["interpolation"];
        const bool crop_to_aspect_ratio = data["config"]["crop_to_aspect_ratio"];
        return std::make_shared<resizing_layer>(name, height, width, interpolation, crop_to_aspect_ratio);
    }

    inline activation_layer_ptr create_linear_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<linear_layer>(name);
    }

    inline activation_layer_ptr create_softmax_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<softmax_layer>(name);
    }

    inline activation_layer_ptr create_softplus_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<softplus_layer>(name);
    }

    inline activation_layer_ptr create_tanh_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<tanh_layer>(name);
    }

    inline activation_layer_ptr create_sigmoid_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<sigmoid_layer>(name);
    }

    inline activation_layer_ptr create_swish_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<swish_layer>(name);
    }

    inline activation_layer_ptr create_hard_sigmoid_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<hard_sigmoid_layer>(name);
    }

    inline activation_layer_ptr create_hard_shrink_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        float_type threshold = data["config"]["threshold"];
        return std::make_shared<hard_shrink_layer>(name, threshold);
    }

    inline activation_layer_ptr create_hard_tanh_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<hard_tanh_layer>(name);
    }

    inline activation_layer_ptr create_log_sigmoid_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<log_sigmoid_layer>(name);
    }

    inline activation_layer_ptr create_log_softmax_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<log_softmax_layer>(name);
    }

    inline activation_layer_ptr create_soft_shrink_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        float_type threshold = data["config"]["threshold"];
        return std::make_shared<soft_shrink_layer>(name, threshold);
    }

    inline activation_layer_ptr create_sparse_plus_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<sparse_plus_layer>(name);
    }

    inline activation_layer_ptr create_square_plus_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        float_type b = static_cast<float_type>(4.0);
        if (json_obj_has_member(data, "config") && json_obj_has_member(data["config"], "b") && !data["config"]["b"].is_null()) {
            b = data["config"]["b"];
        }
        return std::make_shared<square_plus_layer>(name, b);
    }

    inline activation_layer_ptr create_tanh_shrink_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<tanh_shrink_layer>(name);
    }

    inline activation_layer_ptr create_threshold_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        // Keras renamed these config keys: threshold_value -> threshold, value -> default_value.
        const auto& config = data["config"];
        float_type threshold = json_obj_has_member(config, "threshold")
            ? static_cast<float_type>(config["threshold"])
            : static_cast<float_type>(config["threshold_value"]);
        float_type default_value = json_obj_has_member(config, "default_value")
            ? static_cast<float_type>(config["default_value"])
            : static_cast<float_type>(config["value"]);
        return std::make_shared<threshold_layer>(name, threshold, default_value);
    }

    inline activation_layer_ptr create_relu_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        float_type max_value = std::numeric_limits<float_type>::max();
        float_type negative_slope = static_cast<float_type>(0);
        float_type threshold = static_cast<float_type>(0);
        if (json_obj_has_member(data, "config") && json_obj_has_member(data["config"], "max_value") && !data["config"]["max_value"].is_null()) {
            max_value = data["config"]["max_value"];
            negative_slope = data["config"]["negative_slope"];
            threshold = data["config"]["threshold"];
        }
        return std::make_shared<relu_layer>(name, max_value, negative_slope, threshold);
    }

    inline activation_layer_ptr create_relu6_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<relu_layer>(name,
            static_cast<float_type>(6),
            static_cast<float_type>(0),
            static_cast<float_type>(0));
    }

    inline activation_layer_ptr create_selu_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<selu_layer>(name);
    }

    inline activation_layer_ptr create_exponential_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<exponential_layer>(name);
    }

    inline activation_layer_ptr create_gelu_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        if (json_obj_has_member(data, "config") && json_obj_has_member(data["config"], "approximate") && !data["config"]["approximate"].is_null()) {
            const bool approximate = data["config"]["approximate"];
            assertion(approximate == false, "Gelu with approximate = True is not supported.");
        }
        return std::make_shared<gelu_layer>(name);
    }

    inline activation_layer_ptr create_softsign_layer(
        const get_param_f&, const nlohmann::json&,
        const std::string& name)
    {
        return std::make_shared<softsign_layer>(name);
    }

    inline activation_layer_ptr create_celu_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        float_type alpha = 1.0f;
        if (json_obj_has_member(data, "config") && json_obj_has_member(data["config"], "alpha")) {
            alpha = data["config"]["alpha"];
        }
        return std::make_shared<celu_layer>(name, alpha);
    }

    inline activation_layer_ptr create_leaky_relu_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        float_type negative_slope = 0.2f;
        if (json_obj_has_member(data, "config") && json_obj_has_member(data["config"], "negative_slope")) {
            negative_slope = data["config"]["negative_slope"];
        }
        return std::make_shared<leaky_relu_layer>(name, negative_slope);
    }

    inline layer_ptr create_prelu_layer(
        const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        std::vector<std::size_t> shared_axes;
        if (json_obj_has_member(data, "config") && json_obj_has_member(data["config"], "shared_axes") && !data["config"]["shared_axes"].empty()) {
            shared_axes = create_vector<std::size_t>(create_size_t,
                data["config"]["shared_axes"]);
        }
        const float_vec alpha = decode_floats(get_param(name, "alpha"));
        return std::make_shared<prelu_layer>(name, alpha, shared_axes);
    }

    inline activation_layer_ptr create_elu_layer(
        const get_param_f&, const nlohmann::json& data,
        const std::string& name)
    {
        float_type alpha = 1.0f;
        if (json_obj_has_member(data, "config") && json_obj_has_member(data["config"], "alpha")) {
            alpha = data["config"]["alpha"];
        }
        return std::make_shared<elu_layer>(name, alpha);
    }

    inline layer_ptr create_normalization_layer(
        const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        const auto axex = create_vector<int>(create_int, data["config"]["axis"]);
        const float_vec mean = decode_floats(get_param(name, "mean"));
        const float_vec variance = decode_floats(get_param(name, "variance"));
        bool invert = false;
        if (json_obj_has_member(data["config"], "invert")) {
            invert = data["config"]["invert"];
        }
        return std::make_shared<normalization_layer>(name, axex, mean, variance, invert);
    }

    inline layer_ptr create_category_encoding_layer(
        const get_param_f&,
        const nlohmann::json& data, const std::string& name)
    {
        const std::size_t num_tokens = data["config"]["num_tokens"];
        const std::string output_mode = data["config"]["output_mode"];
        return std::make_shared<category_encoding_layer>(name, num_tokens, output_mode);
    }

    inline layer_ptr create_attention_layer(
        const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        const bool use_scale = data["config"]["use_scale"];
        const std::string score_mode = data["config"]["score_mode"];
        float_type scale = static_cast<float_type>(1);
        float_type concat_score_weight = static_cast<float_type>(1);
        if (use_scale) {
            scale = get_param(name, "scale");
        }
        if (score_mode == "concat") {
            concat_score_weight = get_param(name, "concat_score_weight");
        }
        return std::make_shared<attention_layer>(name, score_mode, scale, concat_score_weight);
    }

    inline layer_ptr create_additive_attention_layer(
        const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        const bool use_scale = data["config"]["use_scale"];
        float_vec scale(1, static_cast<float_type>(1));
        if (use_scale) {
            scale = decode_floats(get_param(name, "scale"));
        }
        return std::make_shared<additive_attention_layer>(name, scale);
    }

    inline layer_ptr create_group_query_attention_layer(
        const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        const std::size_t head_dim = data["config"]["head_dim"];
        const std::size_t num_query_heads = data["config"]["num_query_heads"];
        const std::size_t num_kv_heads = data["config"]["num_key_value_heads"];
        const bool use_bias = data["config"]["use_bias"];
        const bool use_gate = json_object_get(data["config"], "use_gate", false);

        const auto weight_shapes = create_vector<std::vector<std::size_t>>(fplus::bind_1st_of_2(
                                                                               create_vector<std::size_t, decltype(create_size_t)>, create_size_t),
            get_param(name, "weight_shapes"));
        const auto weight_values = create_vector<float_vec>(decode_floats, get_param(name, "weights"));
        const auto weights = fplus::zip_with(
            [](const std::vector<std::size_t>& shape, const float_vec& values) -> tensor {
                return tensor(create_tensor_shape_from_dims(shape),
                    fplus::convert_container<float_vec>(values));
            },
            weight_shapes, weight_values);
        return std::make_shared<group_query_attention_layer>(name,
            head_dim, num_query_heads, num_kv_heads, use_bias, use_gate, weights);
    }

    inline layer_ptr create_multi_head_attention_layer(
        const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        const std::size_t num_heads = data["config"]["num_heads"];
        const std::size_t key_dim = data["config"]["key_dim"];
        const std::size_t value_dim = data["config"]["value_dim"];
        const bool use_bias = data["config"]["use_bias"];
        const auto weight_shapes = create_vector<std::vector<std::size_t>>(fplus::bind_1st_of_2(
                                                                               create_vector<std::size_t, decltype(create_size_t)>, create_size_t),
            get_param(name, "weight_shapes"));
        const auto weight_values = create_vector<float_vec>(decode_floats, get_param(name, "weights"));
        const auto weights_and_biases = fplus::zip_with(
            [](const std::vector<std::size_t>& shape, const float_vec& values) -> tensor {
                return tensor(
                    create_tensor_shape_from_dims(shape),
                    fplus::convert_container<float_vec>(values));
            },
            weight_shapes, weight_values);
        return std::make_shared<multi_head_attention_layer>(name,
            num_heads, key_dim, value_dim, use_bias, weights_and_biases);
    }


    inline layer_ptr create_lstm_layer(const get_param_f& get_param,
        const nlohmann::json& data,
        const std::string& name)
    {
        auto&& config = data["config"];
        const std::size_t units = config["units"];
        const std::string unit_activation = json_object_get_activation_with_default(config, "tanh");
        const std::string recurrent_activation = json_object_get_named_activation_with_default(config, "recurrent_activation", "sigmoid");
        const bool use_bias = json_object_get(config, "use_bias", true);
        const bool return_sequences = json_object_get(config, "return_sequences", false);
        const bool return_state = json_object_get(config, "return_state", false);

        float_vec bias;
        if (use_bias)
            bias = decode_floats(get_param(name, "bias"));

        const float_vec weights = decode_floats(get_param(name, "weights"));
        const float_vec recurrent_weights = decode_floats(get_param(name, "recurrent_weights"));

        return std::make_shared<lstm_layer>(name, units, unit_activation,
            recurrent_activation, use_bias,
            return_sequences, return_state,
            weights, recurrent_weights, bias);
    }

    inline layer_ptr create_gru_layer(const get_param_f& get_param,
        const nlohmann::json& data,
        const std::string& name)
    {
        auto&& config = data["config"];
        const std::size_t units = config["units"];
        const std::string unit_activation = json_object_get_activation_with_default(config, "tanh");
        const std::string recurrent_activation = json_object_get_named_activation_with_default(config, "recurrent_activation", "sigmoid");
        const bool use_bias = json_object_get(config, "use_bias", true);
        const bool return_sequences = json_object_get(config, "return_sequences", false);
        const bool return_state = json_object_get(config, "return_state", false);
        const bool reset_after = json_object_get(config, "reset_after", true);

        float_vec bias;
        if (use_bias)
            bias = decode_floats(get_param(name, "bias"));

        const float_vec weights = decode_floats(get_param(name, "weights"));
        const float_vec recurrent_weights = decode_floats(get_param(name, "recurrent_weights"));

        return std::make_shared<gru_layer>(name, units, unit_activation,
            recurrent_activation, use_bias, reset_after,
            return_sequences, return_state,
            weights, recurrent_weights, bias);
    }

    inline layer_ptr create_simple_rnn_layer(const get_param_f& get_param,
        const nlohmann::json& data,
        const std::string& name)
    {
        auto&& config = data["config"];
        const std::size_t units = config["units"];
        const std::string unit_activation = json_object_get_activation_with_default(config, "tanh");
        const bool use_bias = json_object_get(config, "use_bias", true);
        const bool return_sequences = json_object_get(config, "return_sequences", false);
        const bool return_state = json_object_get(config, "return_state", false);

        float_vec bias;
        if (use_bias)
            bias = decode_floats(get_param(name, "bias"));

        const float_vec weights = decode_floats(get_param(name, "weights"));
        const float_vec recurrent_weights = decode_floats(get_param(name, "recurrent_weights"));

        return std::make_shared<simple_rnn_layer>(name, units, unit_activation,
            use_bias, return_sequences, return_state,
            weights, recurrent_weights, bias);
    }

    inline layer_ptr create_rnn_from_cell(const get_param_f& get_param,
        const std::string& cell_class, const nlohmann::json& cell_config,
        const nlohmann::json& outer_cfg, const std::string& name,
        bool override_return_sequences)
    {
        nlohmann::json synthetic;
        synthetic["class_name"] = cell_class == "LSTMCell" ? "LSTM"
            : cell_class == "GRUCell" ? "GRU"
            : cell_class == "SimpleRNNCell" ? "SimpleRNN"
            : std::string("");
        synthetic["config"] = cell_config;
        for (const char* key : { "return_sequences", "return_state", "go_backwards", "stateful", "unroll" }) {
            if (json_obj_has_member(outer_cfg, key))
                synthetic["config"][key] = outer_cfg[key];
        }
        if (override_return_sequences)
            synthetic["config"]["return_sequences"] = true;

        if (cell_class == "LSTMCell")
            return create_lstm_layer(get_param, synthetic, name);
        if (cell_class == "GRUCell")
            return create_gru_layer(get_param, synthetic, name);
        if (cell_class == "SimpleRNNCell")
            return create_simple_rnn_layer(get_param, synthetic, name);
        raise_error("RNN cell '" + cell_class + "' is not supported.");
        return {};
    }

    inline layer_ptr create_rnn_layer(const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        auto&& cfg = data["config"];
        const auto& cell = cfg["cell"];
        const std::string cell_class = cell["class_name"];

        if (cell_class == "StackedRNNCells") {
            const auto& cells = cell["config"]["cells"];
            const std::size_t num_cells = cells.size();
            assertion(num_cells > 0, "StackedRNNCells must contain at least one cell.");
            std::vector<layer_ptr> inner_layers;
            inner_layers.reserve(num_cells);
            for (std::size_t i = 0; i < num_cells; ++i) {
                const auto& sub_cell = cells[i];
                const std::string sub_class = sub_cell["class_name"];
                const std::string sub_name = name + "_cell" + std::to_string(i);
                const std::string prefix = "cell" + std::to_string(i) + "_";
                const get_param_f sub_get_param =
                    [&get_param, name, prefix](const std::string& layer_name, const std::string& key) {
                        (void)layer_name;
                        return get_param(name, prefix + key);
                    };
                const bool is_last = (i + 1 == num_cells);
                inner_layers.push_back(create_rnn_from_cell(sub_get_param,
                    sub_class, sub_cell["config"], cfg, sub_name, !is_last));
            }
            return std::make_shared<stacked_rnn_layer>(name, inner_layers);
        }

        return create_rnn_from_cell(get_param, cell_class, cell["config"],
            cfg, name, false);
    }

    inline layer_ptr create_bidirectional_layer(const get_param_f& get_param,
        const nlohmann::json& data,
        const std::string& name)
    {
        const std::string merge_mode = data["config"]["merge_mode"];
        auto&& wrapped = data["config"]["layer"];
        auto&& wrapped_config = wrapped["config"];
        const std::string wrapped_layer_type = wrapped["class_name"];
        const std::size_t units = wrapped_config["units"];
        const std::string unit_activation = json_object_get_activation_with_default(wrapped_config, "tanh");
        const std::string recurrent_activation = json_object_get_named_activation_with_default(wrapped_config, "recurrent_activation", "sigmoid");
        const bool use_bias = json_object_get(wrapped_config, "use_bias", true);
        const bool return_sequences = json_object_get(wrapped_config, "return_sequences", false);
        const bool reset_after = json_object_get(wrapped_config, "reset_after", true);

        float_vec forward_bias;
        float_vec backward_bias;
        if (use_bias) {
            forward_bias = decode_floats(get_param(name, "forward_bias"));
            backward_bias = decode_floats(get_param(name, "backward_bias"));
        }

        const float_vec forward_weights = decode_floats(get_param(name, "forward_weights"));
        const float_vec backward_weights = decode_floats(get_param(name, "backward_weights"));
        const float_vec forward_recurrent_weights = decode_floats(get_param(name, "forward_recurrent_weights"));
        const float_vec backward_recurrent_weights = decode_floats(get_param(name, "backward_recurrent_weights"));

        return std::make_shared<bidirectional_layer>(name, merge_mode, units,
            unit_activation, recurrent_activation, wrapped_layer_type,
            use_bias, reset_after, return_sequences,
            forward_weights, forward_recurrent_weights, forward_bias,
            backward_weights, backward_recurrent_weights, backward_bias);
    }

    inline activation_layer_ptr create_activation_layer_type_name(
        const get_param_f& get_param,
        const nlohmann::json& data,
        const std::string& type, const std::string& name)
    {
        const std::map<std::string,
            std::function<activation_layer_ptr(const get_param_f&,
                const nlohmann::json&,
                const std::string&)>>
            creators = {
                { "linear", create_linear_layer },
                { "softmax", create_softmax_layer },
                { "softplus", create_softplus_layer },
                { "tanh", create_tanh_layer },
                { "sigmoid", create_sigmoid_layer },
                { "swish", create_swish_layer },
                { "silu", create_swish_layer },
                { "hard_sigmoid", create_hard_sigmoid_layer },
                { "hard_shrink", create_hard_shrink_layer },
                { "hard_tanh", create_hard_tanh_layer },
                { "log_sigmoid", create_log_sigmoid_layer },
                { "log_softmax", create_log_softmax_layer },
                { "leaky_relu", create_leaky_relu_layer },
                { "soft_shrink", create_soft_shrink_layer },
                { "sparse_plus", create_sparse_plus_layer },
                { "squareplus", create_square_plus_layer },
                { "tanh_shrink", create_tanh_shrink_layer },
                { "relu", create_relu_layer },
                { "relu6", create_relu6_layer },
                { "selu", create_selu_layer },
                { "elu", create_elu_layer },
                { "celu", create_celu_layer },
                { "exponential", create_exponential_layer },
                { "gelu", create_gelu_layer },
                { "softsign", create_softsign_layer }
            };

        return fplus::throw_on_nothing(
            error("unknown activation type: " + type),
            fplus::get_from_map(creators, type))(
            get_param, data, name);
    }

    inline layer_ptr create_activation_layer(
        const get_param_f& get_param,
        const nlohmann::json& data, const std::string& name)
    {
        const std::string type = get_activation_type(data["config"]["activation"]);
        return create_activation_layer_type_name(get_param,
            data, type, name);
    }

    inline layer_ptr create_permute_layer(
        const get_param_f&,
        const nlohmann::json& data, const std::string& name)
    {
        const auto dims = create_vector<std::size_t>(create_size_t,
            data["config"]["dims"]);
        return std::make_shared<permute_layer>(name, dims);
    }

    inline node create_node(const nlohmann::json& inbound_nodes_data)
    {
        assertion(inbound_nodes_data["args"].is_array(), "node args need to be an array");
        std::vector<nlohmann::json> args = inbound_nodes_data["args"];
        if (args.front().is_array()) {
            assertion(args.size() == 1, "invalid args format");
            const std::vector<nlohmann::json> inner_args = args.front();
            return node(fplus::transform(create_node_connection, inner_args));
        } else {
            return node(fplus::transform(create_node_connection, args));
        }
    }

    inline nodes create_nodes(const nlohmann::json& data)
    {
        assertion(data["inbound_nodes"].is_array(), "no inbound nodes");
        const std::vector<nlohmann::json> inbound_nodes_data = data["inbound_nodes"];
        return fplus::transform(create_node, inbound_nodes_data);
    }

    inline layer_ptr create_embedding_layer(const get_param_f& get_param,
        const nlohmann::json& data,
        const std::string& name)
    {
        const std::size_t input_dim = data["config"]["input_dim"];
        const std::size_t output_dim = data["config"]["output_dim"];
        const float_vec weights = decode_floats(get_param(name, "weights"));

        return std::make_shared<embedding_layer>(name, input_dim, output_dim, weights);
    }

    inline layer_ptr create_time_distributed_layer(const get_param_f& get_param,
        const nlohmann::json& data,
        const std::string& name,
        const layer_creators& custom_layer_creators,
        const std::string& prefix)
    {
        const std::string wrapped_layer_type = data["config"]["layer"]["class_name"];
        nlohmann::json data_inner_layer = data["config"]["layer"];
        data_inner_layer["name"] = data["name"];
        data_inner_layer["inbound_nodes"] = data["inbound_nodes"];
        const std::size_t td_input_len = std::size_t(decode_floats(get_param(name, "td_input_len")).front());
        const std::size_t td_output_len = std::size_t(decode_floats(get_param(name, "td_output_len")).front());

        layer_ptr inner_layer = create_layer(get_param, data_inner_layer, custom_layer_creators, prefix);

        return std::make_shared<time_distributed_layer>(name, inner_layer, td_input_len, td_output_len);
    }

    inline layer_ptr create_layer(const get_param_f& get_param,
        const nlohmann::json& data,
        const layer_creators& custom_layer_creators,
        const std::string&)
    {
        const std::string name = data["name"];

        const layer_creators default_creators = {
            { "Identity", create_identity_layer },
            { "Conv1D", create_conv_2d_layer },
            { "Conv2D", create_conv_2d_layer },
            { "Conv3D", create_conv_3d_layer },
            { "Conv1DTranspose", create_conv_2d_transpose_layer },
            { "Conv2DTranspose", create_conv_2d_transpose_layer },
            { "Conv3DTranspose", create_conv_3d_transpose_layer },
            { "SeparableConv1D", create_separable_conv_2D_layer },
            { "SeparableConv2D", create_separable_conv_2D_layer },
            { "DepthwiseConv1D", create_depthwise_conv_2D_layer },
            { "DepthwiseConv2D", create_depthwise_conv_2D_layer },
            { "InputLayer", create_input_layer },
            { "BatchNormalization", create_batch_normalization_layer },
            { "GroupNormalization", create_group_normalization_layer },
            { "LayerNormalization", create_layer_normalization_layer },
            { "RMSNormalization", create_rms_normalization_layer },
            { "UnitNormalization", create_unit_normalization_layer },
            { "Dropout", create_identity_layer },
            { "Masking", create_identity_layer },
            { "ActivityRegularization", create_identity_layer },
            { "AlphaDropout", create_identity_layer },
            { "FixedDropout", create_identity_layer },
            { "GaussianDropout", create_identity_layer },
            { "GaussianNoise", create_identity_layer },
            { "SpatialDropout1D", create_identity_layer },
            { "SpatialDropout2D", create_identity_layer },
            { "SpatialDropout3D", create_identity_layer },
            { "RandomBrightness", create_identity_layer },
            { "RandomColorDegeneration", create_identity_layer },
            { "RandomColorJitter", create_identity_layer },
            { "RandomContrast", create_identity_layer },
            { "RandomCrop", create_identity_layer },
            { "RandomElasticTransform", create_identity_layer },
            { "RandomErasing", create_identity_layer },
            { "RandomFlip", create_identity_layer },
            { "RandomGaussianBlur", create_identity_layer },
            { "RandomGrayscale", create_identity_layer },
            { "RandomHeight", create_identity_layer },
            { "RandomHue", create_identity_layer },
            { "RandomInvert", create_identity_layer },
            { "RandomPerspective", create_identity_layer },
            { "RandomPosterization", create_identity_layer },
            { "RandomRotation", create_identity_layer },
            { "RandomSaturation", create_identity_layer },
            { "RandomSharpness", create_identity_layer },
            { "RandomShear", create_identity_layer },
            { "RandomTranslation", create_identity_layer },
            { "RandomWidth", create_identity_layer },
            { "RandomZoom", create_identity_layer },
            { "AugMix", create_identity_layer },
            { "AutoContrast", create_identity_layer },
            { "CutMix", create_identity_layer },
            { "Equalization", create_identity_layer },
            { "MaxNumBoundingBoxes", create_identity_layer },
            { "MixUp", create_identity_layer },
            { "Pipeline", create_identity_layer },
            { "RandAugment", create_identity_layer },
            { "Solarization", create_identity_layer },
            { "LeakyReLU", create_leaky_relu_layer },
            { "Permute", create_permute_layer },
            { "PReLU", create_prelu_layer },
            { "ELU", create_elu_layer },
            { "ReLU", create_relu_layer },
            { "Relu6", create_relu6_layer },
            { "Celu", create_celu_layer },
            { "Elu", create_elu_layer },
            { "Exp", create_exponential_layer },
            { "Gelu", create_gelu_layer },
            { "Selu", create_selu_layer },
            { "Silu", create_swish_layer },
            { "Tanh", create_tanh_layer },
            { "TanhShrink", create_tanh_shrink_layer },
            { "Threshold", create_threshold_layer },
            { "Sigmoid", create_sigmoid_layer },
            { "HardShrink", create_hard_shrink_layer },
            { "HardSigmoid", create_hard_sigmoid_layer },
            { "HardTanh", create_hard_tanh_layer },
            { "SoftShrink", create_soft_shrink_layer },
            { "Softplus", create_softplus_layer },
            { "Softsign", create_softsign_layer },
            { "SparsePlus", create_sparse_plus_layer },
            { "Squareplus", create_square_plus_layer },
            { "LeakyRelu", create_leaky_relu_layer },
            { "LogSigmoid", create_log_sigmoid_layer },
            { "LogSoftmax", create_log_softmax_layer },
            { "MaxPooling1D", create_max_pooling_3d_layer },
            { "MaxPooling2D", create_max_pooling_3d_layer },
            { "MaxPooling3D", create_max_pooling_3d_layer },
            { "AveragePooling1D", create_average_pooling_3d_layer },
            { "AveragePooling2D", create_average_pooling_3d_layer },
            { "AveragePooling3D", create_average_pooling_3d_layer },
            { "AdaptiveAveragePooling1D", create_adaptive_avg_pooling_layer },
            { "AdaptiveAveragePooling2D", create_adaptive_avg_pooling_layer },
            { "AdaptiveAveragePooling3D", create_adaptive_avg_pooling_layer },
            { "AdaptiveMaxPooling1D", create_adaptive_max_pooling_layer },
            { "AdaptiveMaxPooling2D", create_adaptive_max_pooling_layer },
            { "AdaptiveMaxPooling3D", create_adaptive_max_pooling_layer },
            { "GlobalMaxPooling1D", create_global_max_pooling_3d_layer },
            { "GlobalMaxPooling2D", create_global_max_pooling_3d_layer },
            { "GlobalMaxPooling3D", create_global_max_pooling_3d_layer },
            { "GlobalAveragePooling1D", create_global_average_pooling_3d_layer },
            { "GlobalAveragePooling2D", create_global_average_pooling_3d_layer },
            { "GlobalAveragePooling3D", create_global_average_pooling_3d_layer },
            { "UpSampling1D", create_upsampling_1d_layer },
            { "UpSampling2D", create_upsampling_2d_layer },
            { "UpSampling3D", create_upsampling_3d_layer },
            { "Dense", create_dense_layer },
            { "Add", create_add_layer },
            { "Maximum", create_maximum_layer },
            { "Minimum", create_minimum_layer },
            { "Dot", create_dot_layer },
            { "Concatenate", create_concatenate_layer },
            { "Multiply", create_multiply_layer },
            { "Average", create_average_layer },
            { "Subtract", create_subtract_layer },
            { "Flatten", create_flatten_layer },
            { "ZeroPadding1D", create_zero_padding_3d_layer },
            { "ZeroPadding2D", create_zero_padding_3d_layer },
            { "ZeroPadding3D", create_zero_padding_3d_layer },
            { "Cropping1D", create_cropping_3d_layer },
            { "Cropping2D", create_cropping_3d_layer },
            { "Cropping3D", create_cropping_3d_layer },
            { "CenterCrop", create_centercrop_layer },
            { "Activation", create_activation_layer },
            { "RepeatVector", create_repeat_vector_layer },
            { "Rescaling", create_rescaling_layer },
            { "Reshape", create_reshape_layer },
            { "Resizing", create_resizing_layer },
            { "ConvLSTM1D", create_conv_lstm_2d_layer },
            { "ConvLSTM2D", create_conv_lstm_2d_layer },
            { "ConvLSTM3D", create_conv_lstm_3d_layer },
            { "Discretization", create_discretization_layer },
            { "IntegerLookup", create_integer_lookup_layer },
            { "EinsumDense", create_einsum_dense_layer },
            { "Embedding", create_embedding_layer },
            { "Softmax", create_softmax_layer },
            { "Normalization", create_normalization_layer },
            { "CategoryEncoding", create_category_encoding_layer },
            { "Attention", create_attention_layer },
            { "AdditiveAttention", create_additive_attention_layer },
            { "MultiHeadAttention", create_multi_head_attention_layer },
            { "GroupQueryAttention", create_group_query_attention_layer },
            { "GroupedQueryAttention", create_group_query_attention_layer },
            { "LSTM", create_lstm_layer },
            { "GRU", create_gru_layer },
            { "SimpleRNN", create_simple_rnn_layer },
            { "RNN", create_rnn_layer },
            { "Bidirectional", create_bidirectional_layer },
        };

        const wrapper_layer_creators wrapper_creators = {
            { "Model", create_model_layer },
            { "Functional", create_model_layer },
            { "TimeDistributed", create_time_distributed_layer }
        };

        const std::string type = data["class_name"];

        if (fplus::map_contains(wrapper_creators, type)) {
            auto result = fplus::get_from_map_unsafe(wrapper_creators, type)(
                get_param, data, name, custom_layer_creators, name + "_");
            result->set_nodes(create_nodes(data));
            return result;
        } else {
            const layer_creators creators = fplus::map_union(custom_layer_creators,
                default_creators);

            auto result = fplus::throw_on_nothing(
                error("unknown layer type: " + type),
                fplus::get_from_map(creators, type))(
                get_param, data, name);

            const bool layer_consumes_activation_internally = type == "Activation"
                || type == "LSTM" || type == "GRU" || type == "SimpleRNN"
                || type == "Bidirectional"
                || type == "ConvLSTM1D" || type == "ConvLSTM2D" || type == "ConvLSTM3D";
            if (!layer_consumes_activation_internally && json_obj_has_member(data["config"], "activation")) {
                const std::string activation = get_activation_type(data["config"]["activation"]);
                result->set_activation(
                    create_activation_layer_type_name(get_param, data,
                        activation, ""));
            }
            result->set_nodes(create_nodes(data));
            return result;
        }
    }

    struct test_case {
        tensors input_;
        tensors output_;
    };

    using test_cases = std::vector<test_case>;

    inline test_case load_test_case(const nlohmann::json& data)
    {
        assertion(data["inputs"].is_array(), "test needs inputs");
        assertion(data["outputs"].is_array(), "test needs outputs");
        return {
            create_vector<tensor>(create_tensor, data["inputs"]),
            create_vector<tensor>(create_tensor, data["outputs"])
        };
    }

    inline test_cases load_test_cases(const nlohmann::json& data)
    {
        return create_vector<test_case>(load_test_case, data);
    }

    inline void check_test_outputs(float_type epsilon,
        const tensors& outputs, const tensors& targets)
    {
        assertion(outputs.size() == targets.size(), "invalid output count");
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            const auto& output = outputs[i];
            const auto& target = targets[i];
            assertion(output.shape() == target.shape(),
                std::string("test failed: ") + "output=" + fplus::show(i) + " " + "Wrong output size. Is " + show_tensor_shape(output.shape()) + ", should be " + show_tensor_shape(target.shape()) + ".");
            for (std::size_t pos_dim_5 = 0; pos_dim_5 < output.shape().size_dim_5_; ++pos_dim_5) {
                for (std::size_t pos_dim_4 = 0; pos_dim_4 < output.shape().size_dim_4_; ++pos_dim_4) {
                    for (std::size_t y = 0; y < output.shape().height_; ++y) {
                        for (std::size_t x = 0; x < output.shape().width_; ++x) {
                            for (std::size_t z = 0; z < output.shape().depth_; ++z) {
                                const tensor_pos pos(pos_dim_5, pos_dim_4, y, x, z);
                                const auto target_val = target.get_ignore_rank(pos);
                                const auto output_val = output.get_ignore_rank(pos);
                                if (!fplus::is_in_closed_interval_around(epsilon,
                                        target_val, output_val)
                                    && !(std::isnan(target_val) && std::isnan(output_val))) {
                                    const std::string msg = std::string("test failed: ") + "output=" + fplus::show(i) + " " + "pos=" + fplus::show(y) + "," + fplus::show(x) + "," + fplus::show(z) + " " + "value=" + fplus::show(output_val) + " "
                                                                                                                                                                                                                                                 "target="
                                        + fplus::show(target_val);
                                    internal::raise_error(msg);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}
}
