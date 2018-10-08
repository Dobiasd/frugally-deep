// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/import_model.hpp"
#include "fdeep/common.hpp"
#include "fdeep/layers/layer.hpp"
#include "fdeep/tensor3.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace fdeep
{

class model
{
public:
    // A single forward pass.
    tensor3s predict(const tensor3s& inputs) const
    {
        const auto input_shapes = fplus::transform(
            fplus_c_mem_fn_t(tensor3, shape, shape_hwc),
            inputs);
        internal::assertion(input_shapes
            == get_input_shapes(),
            std::string("Invalid inputs shape.\n") +
                "The model takes " + show_shape_hwcs_variable(get_input_shapes()) +
                " but provided was: " + show_shape_hwcs(input_shapes));
        const auto outputs = model_layer_->apply(inputs);
        return outputs;
    }

    // Forward pass multiple data.
    // When parallelly == true, the work is distributed to up to
    // as many CPUs as data entries are provided.
    std::vector<tensor3s> predict_multi(const std::vector<tensor3s>& inputs_vec,
        bool parallelly) const
    {
        const auto f = [this](const tensor3s& inputs) -> tensor3s
        {
            return predict(inputs);
        };
        if (parallelly)
        {
            return fplus::transform_parallelly(f, inputs_vec);
        }
        else
        {
            return fplus::transform(f, inputs_vec);
        }
    }

    // Convenience wrapper around predict for models with
    // single tensor outputs of shape (1, 1, z).
    // Suitable for classification models with more than one output neuron.
    // Returns the index of the output neuron with the maximum activation.
    std::size_t predict_class(const tensor3s& inputs) const
    {
        const tensor3s outputs = predict(inputs);
        internal::assertion(outputs.size() == 1,
            "invalid number of outputs");
        const auto output_shape = outputs.front().shape();
        internal::assertion(output_shape.without_depth().area() == 1,
            "invalid output shape");
        return internal::tensor3_max_pos(outputs.front()).z_;
    }

    // Convenience wrapper around predict for models with
    // single tensor outputs of shape (1, 1, 1),
    // typically used for regression or binary classification.
    // Returns this one activation value.
    float_type predict_single_output(const tensor3s& inputs) const
    {
        const tensor3s outputs = predict(inputs);
        internal::assertion(outputs.size() == 1,
            "invalid number of outputs");
        const auto output_shape = outputs.front().shape();
        internal::assertion(output_shape.volume() == 1,
            "invalid output shape");
        return outputs.front().get_yxz(0, 0, 0);
    }

    const std::vector<shape_hwc_variable>& get_input_shapes() const
    {
        return input_shapes_;
    }

    const std::vector<shape_hwc> get_dummy_input_shapes() const
    {
        return fplus::transform(
            fplus::bind_1st_of_2(internal::make_shape_hwc_with,
                                 shape_hwc(42, 42, 42)),
            get_input_shapes());
    }

    // Returns zero-filled tensors with the models input shapes.
    tensor3s generate_dummy_inputs() const
    {
        return fplus::transform([](const shape_hwc& shape) -> tensor3
        {
            return tensor3(shape, 0);
        }, get_dummy_input_shapes());
    }

    // Measure time of one single forward pass using dummy input data.
    double test_speed() const
    {
        const auto inputs = generate_dummy_inputs();
        fplus::stopwatch stopwatch;
        predict(inputs);
        return stopwatch.elapsed();
    }

private:
    model(const internal::layer_ptr& model_layer,
        const std::vector<shape_hwc_variable>& input_shapes) :
            input_shapes_(input_shapes),
            model_layer_(model_layer) {}

    friend model read_model(std::istream&, bool,
        const std::function<void(std::string)>&, float_type);

    std::vector<shape_hwc_variable> input_shapes_;
    internal::layer_ptr model_layer_;
};

// Write an std::string to std::cout.
inline void cout_logger(const std::string& str)
{
    std::cout << str << std::flush;
}

// Load and construct an fdeep::model from an istream
// providing the exported json content.
// Throws an exception if a problem occurs.
inline model read_model(std::istream& model_file_stream,
    bool verify = true,
    const std::function<void(std::string)>& logger = cout_logger,
    float_type verify_epsilon = static_cast<float_type>(0.0001))
{
    const auto log = [&logger](const std::string& msg)
    {
        if (logger)
        {
            logger(msg + "\n");
        }
    };

    fplus::stopwatch stopwatch;

    const auto log_sol = [&stopwatch, &logger](const std::string& msg)
    {
        stopwatch.reset();
        if (logger)
        {
            logger(msg + " ... ");
        }
    };

    const auto log_duration = [&stopwatch, &logger]()
    {
        if (logger)
        {
            logger("done. elapsed time: " +
                fplus::show_float(0, 6, stopwatch.elapsed()) + " s\n");
        }
        stopwatch.reset();
    };

    log_sol("Loading json");
    nlohmann::json json_data;
    model_file_stream >> json_data;
    log_duration();

    const std::string image_data_format = json_data["image_data_format"];
    internal::assertion(image_data_format == "channels_last",
        "only channels_last data format supported");

    const std::function<nlohmann::json(
            const std::string&, const std::string&)>
        get_param = [&json_data]
        (const std::string& layer_name, const std::string& param_name)
        -> nlohmann::json
    {
        return json_data["trainable_params"][layer_name][param_name];
    };

    const std::function<nlohmann::json(const std::string&)>
        get_global_param =
            [&json_data](const std::string& param_name) -> nlohmann::json
    {
        return json_data[param_name];
    };

    const model full_model(internal::create_model_layer(
        get_param, get_global_param, json_data["architecture"],
        json_data["architecture"]["config"]["name"]),
        internal::create_shape_hwcs_variable(json_data["input_shapes"]));

    if (verify)
    {
        if (!json_data["tests"].is_array())
        {
            log("No test cases available");
        }
        else
        {
            const auto tests = internal::load_test_cases(json_data["tests"]);
            json_data = {}; // free RAM
            for (std::size_t i = 0; i < tests.size(); ++i)
            {
                log_sol("Running test " + fplus::show(i + 1) +
                    " of " + fplus::show(tests.size()));
                const auto output = full_model.predict(tests[i].input_);
                log_duration();
                check_test_outputs(verify_epsilon, output, tests[i].output_);
            }
        }
    }

    return full_model;
}

inline model read_model_from_string(const std::string& content,
    bool verify = true,
    const std::function<void(std::string)>& logger = cout_logger,
    float_type verify_epsilon = static_cast<float_type>(0.0001))
{
    std::istringstream content_stream(content);
    return read_model(content_stream, verify, logger, verify_epsilon);
}

// Load and construct an fdeep::model from file.
// Throws an exception if a problem occurs.
inline model load_model(const std::string& file_path,
    bool verify = true,
    const std::function<void(std::string)>& logger = cout_logger,
    float_type verify_epsilon = static_cast<float_type>(0.0001))
{
    fplus::stopwatch stopwatch;
    std::ifstream in_stream(file_path);
    internal::assertion(in_stream.good(), "Can not open " + file_path);
    const auto model = read_model(in_stream, verify, logger, verify_epsilon);
    if (logger)
    {
        const std::string additional_action = verify ? ", testing" : "";
        logger("Loading, constructing" + additional_action +
            " of " + file_path + " took " +
            fplus::show_float(0, 6, stopwatch.elapsed()) + " s overall.\n");
    }
    return model;
}

} // namespace fdeep