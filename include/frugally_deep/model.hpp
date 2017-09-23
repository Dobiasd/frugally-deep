// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/common.hpp"
#include "frugally_deep/keras_import.hpp"

namespace fdeep
{

using tensor3 = internal::tensor3;
using tensor3s = internal::tensor3s;

class model
{
public:
    model(const internal::layer_ptr& model_layer) : model_layer_(model_layer) {}
    tensor3s predict(const tensor3s& inputs) const
    {
        return model_layer_->apply(inputs);
    }
private:
    internal::layer_ptr model_layer_;
};

// Throws an exception if a problem occurs.
inline model load_model(const std::string& path, bool verify = true)
{
    const auto maybe_json_str = fplus::read_text_file_maybe(path)();
    internal::assertion(fplus::is_just(maybe_json_str),
        "Unable to load: " + path);

    const auto json_str = maybe_json_str.unsafe_get_just();
    const auto json_data = nlohmann::json::parse(json_str);

    const std::string image_data_format = json_data["image_data_format"];
    internal::assertion(image_data_format == "channels_last",
        "only channels_last data format supported");

    const std::function<internal::float_vec(
            const std::string&, const std::string&)>
        get_param = [&json_data]
        (const std::string& layer_name, const std::string& param_name)
        -> internal::float_vec
    {
        const internal::float_vec result =
            json_data["trainable_params"][layer_name][param_name];
        return result;
    };

    const model full_model(
        internal::create_model_layer(get_param, json_data["architecture"]));

    if (verify)
    {
        const auto tests = internal::load_test_cases(json_data);
        for (const auto& test_case : tests)
        {
            const auto output = full_model.predict(test_case.input_);
            internal::assertion(are_test_outputs_ok(output, test_case.output_),
                "test failed");
        }
    }

    return full_model;
}

} // namespace fdeep