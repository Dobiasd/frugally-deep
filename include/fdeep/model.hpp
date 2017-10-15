// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/common.hpp"
#include "fdeep/keras_import.hpp"

namespace fdeep
{

class model
{
public:
    model(const internal::layer_ptr& model_layer,
    const std::vector<shape3>& input_shapes,
    const std::vector<shape3>& output_shapes) :
        input_shapes_(input_shapes),
        output_shapes_(output_shapes),
        model_layer_(model_layer) {}

    // im2col is faster on most architectures but uses more RAM.
    tensor3s predict(const tensor3s& inputs, bool use_im2col = true) const
    {
        return model_layer_->apply(use_im2col, inputs);
    }
    std::size_t predict_class(const tensor3s& inputs,
        bool use_im2col = true) const
    {
        const tensor3s outputs = model_layer_->apply(use_im2col, inputs);
        internal::assertion(outputs.size() == 1, "invalid number of outputs");
        const tensor3 output = outputs.front();
        internal::assertion(output.shape().without_depth().area() == 1,
            "invalid output shape");
        return internal::tensor3_max_pos(output).z_;

    }
    const std::vector<shape3>& get_input_shapes() const
    {
        return input_shapes_;
    }
    const std::vector<shape3>& get_output_shapes() const
    {
        return output_shapes_;
    }
private:
    std::vector<shape3> input_shapes_;
    std::vector<shape3> output_shapes_;
    internal::layer_ptr model_layer_;
};

// Throws an exception if a problem occurs.
inline model load_model(const std::string& path,
    bool verify = true,
    bool verbose = true,
    float_type test_epsilon = static_cast<float_type>(0.00001))
{
    const auto log = [verbose](const std::string& msg)
    {
        if (verbose)
        {
            std::cout << msg << std::endl;
        }
    };

    const auto log_sol = [verbose](const std::string& msg)
    {
        if (verbose)
        {
            std::cout << msg << " ... " << std::flush;
        }
    };

    internal::timer stopwatch;
    const auto log_duration = [&stopwatch, verbose]()
    {
        if (verbose)
        {
            std::cout << " done. elapsed time: " <<
                fplus::show_float(0, 6, stopwatch.elapsed()) << " s" <<
                std::endl;
        }
        stopwatch.reset();
    };

    const auto log_ok = [&stopwatch, verbose]()
    {
        if (verbose)
        {
            std::cout << " ok" << std::endl;
        }
        stopwatch.reset();
    };

    log_sol("Reading " + path);
    auto maybe_json_str = fplus::read_text_file_maybe(path)();
    internal::assertion(fplus::is_just(maybe_json_str),
        "Unable to load: " + path);
    log_duration();

    log_sol("Parsing JSON");
    auto json_data =
        nlohmann::json::parse(maybe_json_str.unsafe_get_just());
    maybe_json_str = fplus::nothing<std::string>(); // free RAM
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

    log_sol("Building model");
    const model full_model(internal::create_model_layer(
        get_param, get_global_param, json_data["architecture"]),
        internal::create_shape3s(json_data["input_shapes"]),
        internal::create_shape3s(json_data["output_shapes"]));
    log_duration();

    if (verify)
    {
        if (!json_data["tests"].is_array())
        {
            log_sol("No test cases available");
        }
        else
        {
            log_sol("Loading tests");
            const auto tests = internal::load_test_cases(json_data["tests"]);
            log_duration();
            json_data = {}; // free RAM
            for (std::size_t i = 0; i < tests.size(); ++i)
            {
                log_sol("Running test (im2col)    " + fplus::show(i + 1) +
                    " of " + fplus::show(tests.size()));
                const auto output_im2col =
                    full_model.predict(tests[i].input_, true);
                log_duration();
                log_sol("Checking test output (im2col)   ");
                check_test_outputs(test_epsilon,
                    output_im2col, tests[i].output_);
                log_ok();

                log_sol("Running test (no im2col) " + fplus::show(i + 1) +
                    " of " + fplus::show(tests.size()));
                const auto output_no_im2col =
                full_model.predict(tests[i].input_, false);
                log_duration();
                log_sol("Checking test output (no im2col)");
                check_test_outputs(test_epsilon,
                    output_no_im2col, tests[i].output_);
                log_ok();
            }
            log("All tests OK");
        }
    }

    return full_model;
}

} // namespace fdeep