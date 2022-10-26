// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/import_model.hpp"
#include "fdeep/common.hpp"
#include "fdeep/layers/layer.hpp"
#include "fdeep/tensor.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace fdeep
{

class model
{
public:
    // A single forward pass (no batches).
    // Will raise an exception when used with a stateful model.
    // For those, use predict_stateful instead.
    tensors predict(const tensors& inputs) const
    {
        internal::assertion(!is_stateful(),
            "Prediction on stateful models is not const. Use predict_stateful instead.");
        return predict_impl(inputs);
    }

    // A single forward pass, supporting stateful models.
    tensors predict_stateful(const tensors& inputs)
    {
        return predict_impl(inputs);
    }

    // Forward pass multiple data.
    // When parallelly == true, the work is distributed to up to
    // as many CPUs as data entries are provided.
    std::vector<tensors> predict_multi(const std::vector<tensors>& inputs_vec,
        bool parallelly) const
    {
        internal::assertion(!is_stateful(),
            "Prediction on stateful models is not thread-safe.");
        const auto f = [this](const tensors& inputs) -> tensors
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
    std::size_t predict_class(const tensors& inputs) const
    {
        internal::assertion(!is_stateful(),
            "Prediction on stateful models is not const. Use predict_class_stateful instead.");
        return predict_class_with_confidence_impl(inputs).first;
    }

    std::size_t predict_class_stateful(const tensors& inputs)
    {
        return predict_class_with_confidence_impl(inputs).first;
    }

    // Like predict_class,
    // but also returns the value of the maximally activated output neuron.
    std::pair<std::size_t, float_type>
    predict_class_with_confidence(const tensors& inputs) const
    {
        internal::assertion(!is_stateful(),
            "Prediction on stateful models is not const. Use predict_class_with_confidence_stateful instead.");
        return predict_class_with_confidence_impl(inputs);
    }

    std::pair<std::size_t, float_type>
    predict_class_with_confidence_stateful(const tensors& inputs)
    {
        return predict_class_with_confidence_stateful_impl(inputs);
    }

    // Convenience wrapper around predict for models with
    // single tensor outputs of shape (1, 1, 1),
    // typically used for regression or binary classification.
    // Returns this one activation value.
    float_type predict_single_output(const tensors& inputs) const
    {
        internal::assertion(!is_stateful(),
            "Prediction on stateful models is not const. Use predict_single_output_stateful instead.");
        return predict_single_output_impl(inputs);
    }

    float_type predict_single_output_stateful(const tensors& inputs)
    {
        return predict_single_output_stateful_impl(inputs);
    }

    const std::vector<tensor_shape_variable>& get_input_shapes() const
    {
        return input_shapes_;
    }

    const std::vector<tensor_shape_variable>& get_output_shapes() const
    {
        return output_shapes_;
    }

    const std::vector<tensor_shape> get_dummy_input_shapes() const
    {
        return fplus::transform(
            fplus::bind_1st_of_2(internal::make_tensor_shape_with,
                                 tensor_shape(42, 42, 42)),
            get_input_shapes());
    }

    // Returns zero-filled tensors with the models input shapes.
    tensors generate_dummy_inputs() const
    {
        return fplus::transform([](const tensor_shape& shape) -> tensor
        {
            return tensor(shape, 0);
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

    // Measure time of one single forward pass using dummy input data.
    double test_speed_stateful()
    {
        const auto inputs = generate_dummy_inputs();
        fplus::stopwatch stopwatch;
        predict_stateful(inputs);
        return stopwatch.elapsed();
    }

    const std::string& name() const
    {
        return model_layer_->name_;
    }

    const std::string& hash() const
    {
        return hash_;
    }

    void reset_states()
    {
        model_layer_->reset_states();
    }

    bool is_stateful() const
    {
        return model_layer_->is_stateful();
    }

private:
    model(const internal::layer_ptr& model_layer,
        const std::vector<tensor_shape_variable>& input_shapes,
        const std::vector<tensor_shape_variable>& output_shapes,
        const std::string& hash) :
            input_shapes_(input_shapes),
            output_shapes_(output_shapes),
            model_layer_(model_layer),
            hash_(hash) {}

    friend model read_model(std::istream&, bool,
        const std::function<void(std::string)>&, float_type,
        const internal::layer_creators&);

    tensors predict_impl(const tensors& inputs) const {
        const auto input_shapes = fplus::transform(
            fplus_c_mem_fn_t(tensor, shape, tensor_shape),
            inputs);
        internal::assertion(input_shapes
            == get_input_shapes(),
            std::string("Invalid inputs shape.\n") +
                "The model takes " + show_tensor_shapes_variable(get_input_shapes()) +
                " but provided was: " + show_tensor_shapes(input_shapes));

        const auto outputs = model_layer_->apply(inputs);

        const auto output_shapes = fplus::transform(
            fplus_c_mem_fn_t(tensor, shape, tensor_shape),
            outputs);
        internal::assertion(output_shapes
            == get_output_shapes(),
            std::string("Invalid outputs shape.\n") +
                "The model should return " + show_tensor_shapes_variable(get_output_shapes()) +
                " but actually returned: " + show_tensor_shapes(output_shapes));

        return outputs;
    }

    std::pair<std::size_t, float_type>
    predict_class_with_confidence_impl(const tensors& inputs) const
    {
        const tensors outputs = predict(inputs);
        internal::assertion(outputs.size() == 1,
            std::string("invalid number of outputs.\n") +
            "Use model::predict instead of model::predict_class.");
        const auto output_shape = outputs.front().shape();
        internal::assertion(output_shape.without_depth().area() == 1,
            std::string("invalid output shape.\n") +
            "Use model::predict instead of model::predict_class.");
        const auto pos = internal::tensor_max_pos(outputs.front());
        return std::make_pair(pos.z_, outputs.front().get(pos));
    }

    float_type predict_single_output_impl(const tensors& inputs) const
    {
        const tensors outputs = predict(inputs);
        internal::assertion(outputs.size() == 1,
            "invalid number of outputs");
        const auto output_shape = outputs.front().shape();
        internal::assertion(output_shape.volume() == 1,
            "invalid output shape");
        return to_singleton_value(outputs.front());
    }

    std::pair<std::size_t, float_type>
    predict_class_with_confidence_stateful_impl(const tensors& inputs)
    {
        const tensors outputs = predict_stateful(inputs);
        internal::assertion(outputs.size() == 1,
            std::string("invalid number of outputs.\n") +
            "Use model::predict instead of model::predict_class.");
        const auto output_shape = outputs.front().shape();
        internal::assertion(output_shape.without_depth().area() == 1,
            std::string("invalid output shape.\n") +
            "Use model::predict instead of model::predict_class.");
        const auto pos = internal::tensor_max_pos(outputs.front());
        return std::make_pair(pos.z_, outputs.front().get(pos));
    }

    float_type predict_single_output_stateful_impl(const tensors& inputs)
    {
        const tensors outputs = predict_stateful(inputs);
        internal::assertion(outputs.size() == 1,
            "invalid number of outputs");
        const auto output_shape = outputs.front().shape();
        internal::assertion(output_shape.volume() == 1,
            "invalid output shape");
        return to_singleton_value(outputs.front());
    }

    std::vector<tensor_shape_variable> input_shapes_;
    std::vector<tensor_shape_variable> output_shapes_;
    internal::layer_ptr model_layer_;
    std::string hash_;
};

// Write an std::string to std::cout.
inline void cout_logger(const std::string& str)
{
    std::cout << str << std::flush;
}

// Take an std::string and do nothing.
// Useful for silencing the logging when loading a model.
inline void dev_null_logger(const std::string&)
{
}

// Load and construct an fdeep::model from an istream
// providing the exported json content.
// Throws an exception if a problem occurs.
inline model read_model(std::istream& model_file_stream,
    bool verify = true,
    const std::function<void(std::string)>& logger = cout_logger,
    float_type verify_epsilon = static_cast<float_type>(0.0001),
    const internal::layer_creators& custom_layer_creators = internal::layer_creators())
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

    log_sol("Building model");
    model full_model(internal::create_model_layer(
        get_param, json_data["architecture"],
        json_data["architecture"]["config"]["name"],
        custom_layer_creators,
        ""),
        internal::create_tensor_shapes_variable(json_data["input_shapes"]),
        internal::create_tensor_shapes_variable(json_data["output_shapes"]),
        internal::json_object_get<std::string, std::string>(
            json_data, "hash", ""));
    log_duration();

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
                const auto output = full_model.predict_impl(tests[i].input_);
                log_duration();
                check_test_outputs(verify_epsilon, output, tests[i].output_);
            }
        }
        full_model.reset_states();
    }

    return full_model;
}

inline model read_model_from_string(const std::string& content,
    bool verify = true,
    const std::function<void(std::string)>& logger = cout_logger,
    float_type verify_epsilon = static_cast<float_type>(0.0001),
    const internal::layer_creators& custom_layer_creators =
        internal::layer_creators())
{
    std::istringstream content_stream(content);
    return read_model(content_stream, verify, logger, verify_epsilon,
        custom_layer_creators);
}

// Load and construct an fdeep::model from file.
// Throws an exception if a problem occurs.
inline model load_model(const std::string& file_path,
    bool verify = true,
    const std::function<void(std::string)>& logger = cout_logger,
    float_type verify_epsilon = static_cast<float_type>(0.0001),
    const internal::layer_creators& custom_layer_creators =
        internal::layer_creators())
{
    fplus::stopwatch stopwatch;
    std::ifstream in_stream(file_path);
    internal::assertion(in_stream.good(), "Can not open " + file_path);
    const auto model = read_model(in_stream, verify, logger, verify_epsilon,
    custom_layer_creators);
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
