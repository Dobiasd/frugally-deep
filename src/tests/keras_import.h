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

struct import_layer
{
    std::string name_;
};

std::string show_layer(const import_layer& l)
{
    return fplus::join(std::string(";"),
        std::vector<std::string>{
            l.name_
        });
}

inline fd::size3d create_size3d(const json& data)
{
    if (!data.is_array())
    {
        std::cerr << "size3d needs to be an array" << std::endl;
    }
    std::cout << data << std::endl;
    return fd::size3d(data[0], data[1], data[2]);
}

inline std::vector<fd::size3d> create_size3ds(const json& data)
{
    if (!data.is_array())
    {
        std::cerr << "size3ds need to be an array" << std::endl;
    }
    return fplus::transform_convert<std::vector<fd::size3d>>(
        create_size3d, data);
}

inline std::string create_string(const json& data)
{
    if (!data.is_string())
    {
        std::cerr << "string needs to be a string" << std::endl;
    }
    return data;
}

inline std::vector<std::string> create_strings(const json& data)
{
    if (!data.is_array())
    {
        std::cerr << "strings need to be an array" << std::endl;
    }
    return fplus::transform_convert<std::vector<std::string>>(
        create_string, data);
}

inline import_layer create_layer(const json& data)
{
    if (!data["type"].is_string())
    {
        std::cerr << "name need to be a string" << std::endl;
    }

    if (!data["name"].is_string())
    {
        std::cerr << "name need to be a string" << std::endl;
    }

    if (!data["inbound_nodes"].is_array())
    {
        std::cerr << "inbound_nodes need to be an array" << std::endl;
    }

    return {create_string(data["name"])};
}

inline void keras_import_test()
{
    using namespace fd;

    const auto json_str =
        fplus::read_text_file("keras_export/test_model.json")();
    const json data = json::parse(json_str);

    if (!data["layers"].is_array())
    {
        std::cerr << "no layers" << std::endl;
    }

    const auto layers = fplus::transform_convert<std::vector<import_layer>>(
        create_layer, data["layers"]);

    std::cout
        << fplus::show_cont_with("\n", fplus::transform(show_layer, layers))
        << std::endl;
}
