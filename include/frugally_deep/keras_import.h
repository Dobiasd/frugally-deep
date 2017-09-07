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
    return fd::size3d(data[0], data[1], data[2]);
}

inline std::vector<fd::size3d> create_size3ds(const json& data)
{
    if (!data.is_array())
    {
        throw std::runtime_error(std::string("size3ds needs to be an array"));
    }
    return fplus::transform_convert<std::vector<fd::size3d>>(
        create_size3d, data);
}

inline std::string create_string(const json& data)
{
    if (!data.is_string())
    {
        throw std::runtime_error(std::string("string needs to be a string"));
    }
    return data;
}

inline std::vector<std::string> create_strings(const json& data)
{
    if (!data.is_array())
    {
        throw std::runtime_error(std::string("strings need to be an array"));
    }
    return fplus::transform_convert<std::vector<std::string>>(
        create_string, data);
}

inline fd::float_t create_float(const json& data)
{
    if (!data.is_number())
    {
        throw std::runtime_error(std::string("float needs to be a number"));
    }
    return data;
}

inline std::vector<fd::float_t> create_floats(const json& data)
{
    if (!data.is_array())
    {
        throw std::runtime_error(std::string("floats need to be an array"));
    }
    return fplus::transform_convert<std::vector<fd::float_t>>(
        create_float, data);
}

fd::matrix3d create_matrix3d(const json& data)
{
    const fd::size3d shape = create_size3d(data["shape"]);
    const fd::float_vec values = create_floats(data["values"]);
    return fd::matrix3d(shape, values);
}

fd::matrix3ds create_matrix3ds(const json& data)
{
    if (!data.is_array())
    {
        throw std::runtime_error(std::string("matrix3ds need to be an array"));
    }
    return fplus::transform_convert<fd::matrix3ds>(create_matrix3d, data);
}

fd::model create_model(const json& data);

inline fd::layer_ptr create_model_layer(const json& data)
{
    return std::make_shared<fd::model>(create_model(data));
}

inline fd::layer_ptr create_conv2d_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::convolutional_layer>(name,
        fd::size2d(3,3), 4, 1); // todo
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

inline fd::layer_ptr create_batch_dropout_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::identity_layer>(name);
}

inline fd::layer_ptr create_leaky_relu_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::leaky_relu_layer>(name, 0.1); // todo
}

inline fd::layer_ptr create_relu_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::relu_layer>(name);
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

inline fd::layer_ptr create_softmax_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::softmax_layer>(name);
}

inline fd::layer_ptr create_softplus_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::softplus_layer>(name);
}

inline fd::layer_ptr create_tanh_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::tanh_layer>(name, false, 0.5); // todo
}

inline fd::layer_ptr create_sigmoid_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::sigmoid_layer>(name);
}

inline fd::layer_ptr create_hard_sigmoid_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::hard_sigmoid_layer>(name);
}

inline fd::layer_ptr create_selu_layer(const json& data)
{
    const std::string name = data["name"];
    return std::make_shared<fd::selu_layer>(name);
}

inline fd::layer_ptr create_layer(const json& data)
{
    if (data.find("name") == data.end() || !data["name"].is_string())
    {
        throw std::runtime_error(std::string("name need to be a string"));
    }

    if (data.find("type") == data.end() || !data["type"].is_string())
    {
        const std::string name = data["name"];
        throw std::runtime_error(name + " has no type");
    }

    const std::unordered_map<std::string, std::function<fd::layer_ptr(const json&)>>
        creators = {
            {"Model", create_model_layer},
            {"Conv2D", create_conv2d_layer},
            {"InputLayer", create_input_layer},
            {"BatchNormalization", create_batch_normalization_layer},
            {"Dropout", create_batch_dropout_layer},
            {"LeakyReLU", create_leaky_relu_layer},
            {"ReLU", create_relu_layer},
            {"ELU", create_elu_layer},
            {"MaxPooling2D", create_max_pooling2d_layer},
            {"AveragePooling2D", create_average_pooling2d_layer},
            {"UpSampling2D", create_upsampling2d_layer},
            {"Dense", create_dense_layer},
            {"Concatenate", create_concatename_layer},
            {"Flatten", create_flatten_layer},
            {"SoftMax", create_softmax_layer},
            {"SoftPlus", create_softplus_layer},
            {"TanH", create_tanh_layer},
            {"Sigmoid", create_sigmoid_layer},
            {"HardSigmoid", create_hard_sigmoid_layer},
            {"SeLU", create_selu_layer}
        };

    const std::string type = data["type"];
    const auto creator = fplus::get_from_map(creators, type);
    if (fplus::is_nothing(creator))
    {
        throw std::runtime_error(std::string("unknown type: ") + type);
    }

    // todo: solve nicer
    return creator.unsafe_get_just()(data);
/*
    if (data.find("inbound_nodes") == data.end() ||
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
    const std::string name = data["name"];

    if (!data["layers"].is_array())
    {
        // todo: ok for empty inner models?
        throw std::runtime_error("no layers");
    }

    const auto layers = fplus::transform_convert<std::vector<fd::layer_ptr>>(
        create_layer, data["layers"]);

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
        create_matrix3ds(data["inputs"]),
        create_matrix3ds(data["outputs"])
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

    const auto model = create_model(json_data);

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
