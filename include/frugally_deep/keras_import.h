// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <iostream>

#include "frugally_deep/frugally_deep.h"

#include "frugally_deep/json.hpp"

#include <fplus/fplus.hpp>

using json = nlohmann::json;

inline fd::size3d create_size3d(const json& data)
{
    if (!data.is_array())
    {
        throw std::runtime_error(std::string("size3d needs to be an array"));
    }
    if (data.size() == 1)
        return fd::size3d(0, 0, data[0]);
    if (data.size() == 2)
        return fd::size3d(0, data[0], data[1]);
    if (data.size() == 3)
        return fd::size3d(data[0], data[1], data[2]);
    throw std::runtime_error(std::string("size3d needs 1, 2 or 3 dimensions"));
}

inline fd::size2d create_size2d(const json& data)
{
    if (!data.is_array())
    {
        throw std::runtime_error(std::string("size2d needs to be an array"));
    }
    if (data.size() == 1)
        return fd::size2d(0, data[0]);
    if (data.size() == 2)
        return fd::size2d(data[0], data[1]);
    throw std::runtime_error(std::string("size2d needs 1 or 2 dimensions"));
}

inline fd::matrix3d create_matrix3d(const json& data)
{
    const fd::size3d shape = create_size3d(data["shape"]);
    const fd::float_vec values = data["values"];
    return fd::matrix3d(shape, values);
}

template <typename T, typename F>
std::vector<T> create_vector(F f, const json& data)
{
    if (!data.is_array())
    {
        throw std::runtime_error(std::string("array needs to be an array"));
    }
    return fplus::transform_convert<std::vector<T>>(f, data);
}

fd::model create_model(const json& data);

inline fd::layer_ptr create_model_layer(const json& data)
{
    return std::make_shared<fd::model>(create_model(data));
}

inline fd::layer_ptr create_conv2d_layer(const json& data)
{
    const std::string name = data["name"];
    const std::size_t filter_count = data["config"]["filters"];
    //const fd::float_vec weights = data["weights"];
    //const fd::float_vec biases = data["biases"];
    const fd::size3d filter_size = create_size3d(data["config"]["kernel_size"]);
    const bool use_bias = data["config"]["use_bias"];
    if (!use_bias)
    {
        // todo: if not use_bias set all biases to 0
    }
    const std::string padding_str = data["config"]["padding"];
    const auto maybe_padding =
        fplus::choose<std::string, fd::convolutional_layer::padding>({
        { std::string("valid"), fd::convolutional_layer::padding::valid },
        { std::string("same"), fd::convolutional_layer::padding::same },
    }, padding_str);
    fd::assertion(fplus::is_just(maybe_padding), "no padding");
    //fd::assertion(biases.size() == filter_count, "number of biases does not match");
    //fd::assertion(weights.size() == filter_size.volume() * filter_count, "number of weights does not match");
    const auto padding = maybe_padding.unsafe_get_just();
    const fd::size2d strides = create_size2d(data["config"]["strides"]);
    return std::make_shared<fd::convolutional_layer>(name,
        filter_size, filter_count, strides, padding); // todo
}

inline fd::layer_ptr create_input_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::model>(name); // todo
}

inline fd::layer_ptr create_batch_normalization_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::batch_normalization_layer>(name, 0.1); // todo
}

inline fd::layer_ptr create_dropout_layer(const json& data)
{
    const std::string name = data["name"];
    // dropout rate equals zero in forward pass
    return std::make_shared<fd::identity_layer>(name);
}

inline fd::layer_ptr create_leaky_relu_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::leaky_relu_layer>(name, 0.1); // todo
}

inline fd::layer_ptr create_elu_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::elu_layer>(name, 0.1); // todo
}

inline fd::layer_ptr create_max_pooling2d_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::max_pool_layer>(name, 2); // todo
}

inline fd::layer_ptr create_average_pooling2d_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::avg_pool_layer>(name, 2); // todo
}

inline fd::layer_ptr create_upsampling2d_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::unpool_layer>(name, 2); // todo
}

inline fd::layer_ptr create_dense_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::fully_connected_layer>(name, 2, 3); // todo
}

inline fd::layer_ptr create_concatename_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::model>(name); // todo
}

inline fd::layer_ptr create_flatten_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::flatten_layer>(name);
}

inline fd::activation_layer_ptr create_identity_layer(const std::string& name)
{
    return std::make_shared<fd::identity_layer>(name);
}

inline fd::activation_layer_ptr create_softmax_layer(const std::string& name)
{
    return std::make_shared<fd::softmax_layer>(name);
}

inline fd::activation_layer_ptr create_softplus_layer(const std::string& name)
{
    return std::make_shared<fd::softplus_layer>(name);
}

inline fd::activation_layer_ptr create_tanh_layer(const std::string& name)
{
    return std::make_shared<fd::tanh_layer>(name, false, 0.5); // todo
}

inline fd::activation_layer_ptr create_sigmoid_layer(const std::string& name)
{
    return std::make_shared<fd::sigmoid_layer>(name);
}

inline fd::activation_layer_ptr create_hard_sigmoid_layer(const std::string& name)
{
    return std::make_shared<fd::hard_sigmoid_layer>(name);
}

inline fd::activation_layer_ptr create_relu_layer(const std::string& name)
{
    return std::make_shared<fd::relu_layer>(name);
}

inline fd::activation_layer_ptr create_selu_layer(const std::string& name)
{
    return std::make_shared<fd::selu_layer>(name);
}

inline fd::activation_layer_ptr create_activation_layer(
    const std::string& type, const std::string& name)
{
    const std::unordered_map<std::string,
        std::function<fd::activation_layer_ptr(const std::string&)>>
    creators = {
        {"linear", create_identity_layer},
        {"softmax", create_softmax_layer},
        {"softplus", create_softplus_layer},
        {"tanh", create_tanh_layer},
        {"sigmoid", create_sigmoid_layer},
        {"hard_sigmoid", create_hard_sigmoid_layer},
        {"relu", create_relu_layer},
        {"selu", create_selu_layer}
    };

    const auto creator = fplus::get_from_map(creators, type);
    if (fplus::is_nothing(creator))
    {
        throw std::runtime_error(
            std::string("unknown activation type: ") + type);
    }

    // todo: solve nicer
    auto result = creator.unsafe_get_just()(name);
    return result;
}

inline fd::layer_ptr create_activation_layer_as_layer(const json& data)
{
    const std::string name = data["name"];
    const std::string type = data["config"]["activation"];
    return create_activation_layer(type, name);
}

inline bool json_obj_has_member(const json& data,
    const std::string& member_name)
{
    return data.is_object() && data.find(member_name) != data.end();
}

inline fd::layer_ptr create_layer(const json& data)
{
    const std::string name = data["name"];

    const std::unordered_map<std::string,
            std::function<fd::layer_ptr(const json&)>>
        creators = {
            {"Model", create_model_layer},
            {"Conv2D", create_conv2d_layer},
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
            {"Activation", create_activation_layer_as_layer}
        };

    const std::string type = data["class_name"];
    const auto creator = fplus::get_from_map(creators, type);
    if (fplus::is_nothing(creator))
    {
        throw std::runtime_error(std::string("unknown type: ") + type);
    }

    // todo: solve nicer
    auto result = creator.unsafe_get_just()(data);
    if (json_obj_has_member(data["config"], "activation"))
    {
        result->set_activation(
            create_activation_layer(
                data["config"]["activation"], ""));
    }
    return result;

/*auto result = (data.find("inbound_nodes") == data.end() ||
create_activation_layer(data["activation"])
        !data["inbound_nodes"].is_array())
    {
        const std::string name = data["name"];
        throw std::runtime_error(name + ": inbound_nodes need to be an array");
    }
    */
}

inline fd::model create_model(const json& data)
{
    //output_nodes
    //input_nodes
    const std::string name = data["config"]["name"];

    if (!data["config"]["layers"].is_array())
    {
        // todo: ok for empty inner models?
        throw std::runtime_error("no layers");
    }

    const auto layers = fplus::transform_convert<std::vector<fd::layer_ptr>>(
        create_layer, data["config"]["layers"]);

    // todo: remove
    const auto show_layer = [](const fd::layer_ptr& ptr) -> std::string
        {
            return ptr->name();
        };
    std::cout
        << fplus::show_cont_with("\n", fplus::transform(show_layer, layers))
            << std::endl;

    return fd::model(name);
}

// todo load_model_from_data

// todo load layers into model (also nested models)

// todo: move everything into namespace

struct test_case
{
    fd::matrix3ds input_;
    fd::matrix3ds output_;
};

using test_cases = std::vector<test_case>;

inline test_case load_test_case(const json& data)
{
    if (!data["inputs"].is_array())
    {
        throw std::runtime_error(std::string("test needs inputs"));
    }
    if (!data["outputs"].is_array())
    {
        throw std::runtime_error(std::string("test needs inputs"));
    }
    return {
        create_vector<fd::matrix3d>(create_matrix3d, data["inputs"]),
        create_vector<fd::matrix3d>(create_matrix3d, data["outputs"])
    };
}

inline test_cases load_test_cases(const json& data)
{
    if (!data["tests"].is_array())
    {
        throw std::runtime_error(std::string("no tests"));
    }
    const auto tests = fplus::transform_convert<test_cases>(
        load_test_case, data["tests"]);

    // todo return without temp variable
    return tests;
}

inline bool run_test_cases(const fd::model&, const test_cases&)
{
    // todo
    return true;
}

// thrown an std::runtime_error if a problem occurs.
inline fd::model load_model(const std::string& path, bool verify = true)
{
    const auto maybe_json_str = fplus::read_text_file_maybe(path)();
    if (fplus::is_nothing(maybe_json_str))
    {
        throw std::runtime_error(std::string("Unable to load file: " + path));
    }

    const auto json_str = maybe_json_str.unsafe_get_just();
    const auto json_data = json::parse(json_str);

    const auto model = create_model(json_data["architecture"]);

    const auto tests = load_test_cases(json_data);

    if (verify)
    {
        if (!run_test_cases(model, tests))
        {
            throw std::runtime_error(std::string("Tests failed."));
        }
    }

    return model;
}

inline void keras_import_test()
{
    const auto model = load_model("keras_export/model.json");
}
