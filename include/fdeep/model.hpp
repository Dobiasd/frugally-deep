// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"
#include "fdeep/import_model.hpp"

namespace fdeep
{

class model
{
public:
    // A single forward pass.
    tensor3s predict(const tensor3s& inputs) const
    {
        const auto outputs = model_layer_->apply(inputs);
        internal::assertion(
            fplus::transform(fplus_c_mem_fn_t(tensor3, shape, shape3), outputs)
            == get_output_shapes(), "invalid outputs shape");
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
    // Returns the index of the output neuron with the maximum actication.
    std::size_t predict_class(const tensor3s& inputs) const
    {
        internal::assertion(get_output_shapes().size() == 1,
            "invalid number of outputs");
        const auto output_shape = get_output_shapes().front();
        internal::assertion(output_shape.without_depth().area() == 1,
            "invalid output shape");
        const tensor3s outputs = predict(inputs);
        return internal::tensor3_max_pos(outputs.front()).z_;
    }

    const std::vector<shape3>& get_input_shapes() const
    {
        return input_shapes_;
    }

    const std::vector<shape3>& get_output_shapes() const
    {
        return output_shapes_;
    }

    // Returns zero-filled tensors with the models input shapes.
    tensor3s generate_dummy_inputs() const
    {
        return fplus::transform([](const shape3& shape) -> tensor3
        {
            return tensor3(shape, 0);
        }, get_input_shapes());
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
        const std::vector<shape3>& input_shapes,
        const std::vector<shape3>& output_shapes) :
            input_shapes_(input_shapes),
            output_shapes_(output_shapes),
            model_layer_(model_layer) {}

    friend model read_model(const std::string&, bool,
        const std::function<void(std::string)>&, float_type);

    std::vector<shape3> input_shapes_;
    std::vector<shape3> output_shapes_;
    internal::layer_ptr model_layer_;
};

// Write an std::string to std::cout.
inline void cout_logger(const std::string& str)
{
    std::cout << str << std::flush;
}

// Load and construct an fdeep::model from an std::string
// holding the exported json content.
// Throws an exception if a problem occurs.
inline model read_model(const std::string& content,
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

    log_sol("Parsing json");
    auto json_data = nlohmann::json::parse(content);
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
        internal::create_shape3s(json_data["input_shapes"]),
        internal::create_shape3s(json_data["output_shapes"]));

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

// Load and construct an fdeep::model from file.
// Throws an exception if a problem occurs.
inline model load_model(const std::string& path,
    bool verify = true,
    const std::function<void(std::string)>& logger = cout_logger,
    float_type verify_epsilon = static_cast<float_type>(0.0001))
{
    auto maybe_json_str = fplus::read_text_file_maybe(path)();
    internal::assertion(fplus::is_just(maybe_json_str),
        "Unable to load: " + path);
    return read_model(maybe_json_str.unsafe_get_just(),
        verify, logger, verify_epsilon);
}

} // namespace fdeep