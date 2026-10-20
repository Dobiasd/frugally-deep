// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <fdeep/fdeep.hpp>

#include <cmath>

TEST_CASE("Keras 3 LSTM initial_state connections are validated")
{
    auto data = nlohmann::json::parse(R"({
        "class_name":"LSTM",
        "inbound_nodes":[{
            "args":[{"config":{"keras_history":["sequence",0,0]}}],
            "kwargs":{"initial_state":[
                {"config":{"keras_history":["encoder",0,1]}},
                {"config":{"keras_history":["encoder",0,2]}}
            ]}
        }]
    })");
    CHECK_NOTHROW(fdeep::internal::create_nodes(data));
    auto& states = data["inbound_nodes"][0]["kwargs"]["initial_state"];
    states.erase(1);
    CHECK_THROWS(fdeep::internal::create_nodes(data));
    states = nullptr;
    CHECK_NOTHROW(fdeep::internal::create_nodes(data));
    states = nlohmann::json::parse(R"([
        {"config":{"keras_history":["encoder",0,1]}},
        {"config":{"keras_history":["encoder",0,2]}}
    ])");
    data["class_name"] = "GRU";
    CHECK_THROWS(fdeep::internal::create_nodes(data));
}

TEST_CASE("LSTM explicit initial states preserve predictions and input tensors")
{
    using namespace fdeep;
    const auto sequence = tensor(tensor_shape(2, 1), float_vec { 0, 0 });
    const auto hidden = tensor(tensor_shape(1), float_vec { 0.7f });
    const auto cell = tensor(tensor_shape(1), float_vec { 0.8f });
    const float_vec weights(4, 0);
    // Only the candidate gate reads h(t-1), so dropping either state changes the result.
    const float_vec recurrent { 0, 0, 1, 0 };
    const auto c1 = static_cast<float_type>(0.4) + static_cast<float_type>(0.5) * std::tanh(hidden.as_vector()->front());
    const auto h1 = static_cast<float_type>(0.5) * std::tanh(c1);
    const auto c2 = static_cast<float_type>(0.5) * c1 + static_cast<float_type>(0.5) * std::tanh(h1);
    const auto h2 = static_cast<float_type>(0.5) * std::tanh(c2);

    for (const bool sequences : { false, true }) {
        for (const bool states : { false, true }) {
            internal::lstm_layer layer("lstm", 1, "tanh", "sigmoid", false,
                sequences, states, weights, recurrent, {});
            const auto output = layer.apply({ sequence, hidden, cell });
            REQUIRE(output.size() == (states ? 3u : 1u));
            CHECK(output.front().shape() == (sequences ? tensor_shape(2, 1) : tensor_shape(1)));
            const auto last_pos = sequences ? tensor_pos(1, 0) : tensor_pos(0);
            CHECK(output.front().get_ignore_rank(last_pos) == doctest::Approx(h2));
            if (sequences)
                CHECK(output.front().get_ignore_rank(tensor_pos(0, 0)) == doctest::Approx(h1));
            if (states) {
                CHECK(output[1].get_ignore_rank(tensor_pos(0)) == doctest::Approx(h2));
                CHECK(output[2].get_ignore_rank(tensor_pos(0)) == doctest::Approx(c2));
            }
            CHECK(hidden.as_vector()->front() == 0.7f);
            CHECK(cell.as_vector()->front() == 0.8f);
            CHECK(layer.apply({ sequence }).front().as_vector()->back() == 0);
            CHECK_THROWS(layer.apply({}));
            CHECK_THROWS(layer.apply({ sequence, hidden }));
            CHECK_THROWS(layer.apply({ sequence, hidden, cell, hidden }));
            const auto empty_sequence = tensor(tensor_shape(0, 1), float_vec {});
            CHECK_THROWS(layer.apply({ empty_sequence, hidden, cell }));
            const auto wrong_state = tensor(tensor_shape(2), float_vec { 0, 0 });
            CHECK_THROWS(layer.apply({ sequence, hidden, wrong_state }));
            CHECK_THROWS(layer.apply({ sequence, wrong_state, cell }));
        }
    }
}
