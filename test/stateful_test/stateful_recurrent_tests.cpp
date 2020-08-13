#include "fdeep/fdeep.hpp"
#include <fstream>

using namespace fdeep;

void vec_append(fdeep::float_vec& results, const fdeep::float_vec& x){
    results.insert(std::end(results), std::begin(x), std::end(x));
    return;
}

int main()
{
    std::ofstream outFile;
    outFile.open("./models/fd_results.bin", std::ios::binary);
    const bool verbose = false;

    const fdeep::float_vec x_inf_0 = {2.1, -1.2, 3.14, 1.2};
    const fdeep::float_vec x_inf_1 = {1, 3, -2, 10};
    const fdeep::float_vec state_0 = {40.1, -25.1};
    const fdeep::float_vec state_1 = {34.7, 56.1};
    const fdeep::float_vec state_2 = {-62.5, 12.0};
    const fdeep::float_vec state_3 = {-33.0, -100.0};



    // const fdeep::float_vec state_0 = {1.1, -2.1};
    // const fdeep::float_vec state_1 = {2.7, 3.1};
    // const fdeep::float_vec state_2 = {-2.5, 3.0};
    // const fdeep::float_vec state_3 = {-2.0, -10.0};
    fdeep::float_vec all_results = {};
    fdeep::float_vec one_result = {};

// [40.1, -25.1, 34.7, 56.1, -62.5, 12.0, -33.0, -100.0]
// [1.1, -2.1, 2.7, 3.1, -2.5, 3.0, -2.0, -10.0]

    const shared_float_vec xt0(fplus::make_shared_ref<float_vec>(x_inf_0));
    const shared_float_vec xt1(fplus::make_shared_ref<float_vec>(x_inf_1));
    const shared_float_vec st0(fplus::make_shared_ref<float_vec>(state_0));
    const shared_float_vec st1(fplus::make_shared_ref<float_vec>(state_1));
    const shared_float_vec st2(fplus::make_shared_ref<float_vec>(state_2));
    const shared_float_vec st3(fplus::make_shared_ref<float_vec>(state_3));

    const tensor test_in_0(tensor_shape(4, 1), xt0);
    const tensor test_in_1(tensor_shape(4, 1), xt1);
    const tensor test_state_0(tensor_shape(static_cast<std::size_t>(2)), st0);
    const tensor test_state_1(tensor_shape(static_cast<std::size_t>(2)), st1);
    const tensor test_state_2(tensor_shape(static_cast<std::size_t>(2)), st2);
    const tensor test_state_3(tensor_shape(static_cast<std::size_t>(2)), st3);

    // *********** TEST 1: "GRU_nonstateful_no_init_state.json" ***********
    auto model = load_model("./models/GRU_nonstateful_no_init_state.json");
    /// state_reset = true
    auto result = model.predict({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict({test_in_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict({test_in_1});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 2: "GRU_nonstateful_init_state.json" ***********
    model = load_model("./models/GRU_nonstateful_init_state.json");
    /// state_reset = true
    result = model.predict({test_in_0, test_state_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict({test_in_1, test_state_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict({test_in_0, test_state_0});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict({test_in_1, test_state_0});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 3: "GRU_stateful_no_init_state.json" ***********
    model = load_model("./models/GRU_stateful_no_init_state.json");
    /// state_reset = true fdr =
    result = model.predict_stateful({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict_stateful({test_in_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict_stateful({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict_stateful({test_in_1});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 4: "GRU_stateful_init_state.json" ***********
    model = load_model("./models/GRU_stateful_init_state.json");
    /// state_reset = true
    result = model.predict_stateful({test_in_0, test_state_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict_stateful({test_in_1, test_state_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict_stateful({test_in_0, test_state_0});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict_stateful({test_in_1, test_state_0});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 5: "LSTM_nonstateful_no_init_state.json" ***********
    model = load_model("./models/LSTM_nonstateful_no_init_state.json");
    /// state_reset = true
    result = model.predict({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict({test_in_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict({test_in_1});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 6: "LSTM_nonstateful_init_state.json" ***********
    model = load_model("./models/LSTM_nonstateful_init_state.json");
    /// state_reset = true
    result = model.predict({test_in_0, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict({test_in_1, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict({test_in_0, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict({test_in_1, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 7: "LSTM_stateful_no_init_state.json" ***********
    model = load_model("./models/LSTM_stateful_no_init_state.json");
    /// state_reset = true
    result = model.predict_stateful({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict_stateful({test_in_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict_stateful({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict_stateful({test_in_1});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 8: "LSTM_stateful_init_state.json" ***********
    model = load_model("./models/LSTM_stateful_init_state.json");
    /// state_reset = true
    result = model.predict_stateful({test_in_0, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict_stateful({test_in_1, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict_stateful({test_in_0, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict_stateful({test_in_1, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());

    // ************************* BIDIRECTIONAL TESTS ************************* //

    // *********** TEST 9: "bidi-GRU_nonstateful_no_init_state.json" ***********
    model = load_model("./models/bidi-GRU_nonstateful_no_init_state.json");
    /// state_reset = true
    result = model.predict({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict({test_in_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict({test_in_1});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 10: "bidi-GRU_nonstateful_init_state.json" ***********
    model = load_model("./models/bidi-GRU_nonstateful_init_state.json");
    /// state_reset = true
    result = model.predict({test_in_0, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict({test_in_1, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict({test_in_0, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict({test_in_1, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 11: "bidi-GRU_stateful_no_init_state.json" ***********
    model = load_model("./models/bidi-GRU_stateful_no_init_state.json");
    /// state_reset = true
    result = model.predict_stateful({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict_stateful({test_in_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict_stateful({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict_stateful({test_in_1});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 12: "bidi-GRU_stateful_init_state.json" ***********
    model = load_model("./models/bidi-GRU_stateful_init_state.json");
    /// state_reset = true
    result = model.predict_stateful({test_in_0, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict_stateful({test_in_1, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict_stateful({test_in_0, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict_stateful({test_in_1, test_state_0, test_state_1});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 13: "bidi-LSTM_nonstateful_no_init_state.json" ***********
    model = load_model("./models/bidi-LSTM_nonstateful_no_init_state.json");
    /// state_reset = true
    result = model.predict({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict({test_in_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict({test_in_1});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 14: "bidi-LSTM_nonstateful_init_state.json" ***********
    model = load_model("./models/bidi-LSTM_nonstateful_init_state.json");
    /// state_reset = true
    result = model.predict({test_in_0, test_state_0, test_state_1, test_state_2, test_state_3});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict({test_in_1, test_state_0, test_state_1, test_state_2, test_state_3});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict({test_in_0, test_state_0, test_state_1, test_state_2, test_state_3});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict({test_in_1, test_state_0, test_state_1, test_state_2, test_state_3});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 15: "bidi-LSTM_stateful_no_init_state.json" ***********
    model = load_model("./models/bidi-LSTM_stateful_no_init_state.json");
    /// state_reset = true
    result = model.predict_stateful({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict_stateful({test_in_1});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict_stateful({test_in_0});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict_stateful({test_in_1});
    vec_append(all_results, *result[0].as_vector());

    // *********** TEST 16: "bidi-LSTM_stateful_init_state.json" ***********
    model = load_model("./models/bidi-LSTM_stateful_init_state.json");
    /// state_reset = true
    result = model.predict_stateful({test_in_0, test_state_0, test_state_1, test_state_2, test_state_3});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    result = model.predict_stateful({test_in_1, test_state_0, test_state_1, test_state_2, test_state_3});
    vec_append(all_results, *result[0].as_vector());
    model.reset_states();
    /// state_reset = false
    result = model.predict_stateful({test_in_0, test_state_0, test_state_1, test_state_2, test_state_3});
    vec_append(all_results, *result[0].as_vector());
    result = model.predict_stateful({test_in_1, test_state_0, test_state_1, test_state_2, test_state_3});
    vec_append(all_results, *result[0].as_vector());

    if(verbose){
        std::cout << "\n\nOUTPUT ***" << std::endl;
            for(size_t idx = 0; idx < all_results.size(); ++ idx){
                std::cout << all_results[idx] << std::endl;
        }
    }

    const size_t sz = all_results.size();
    // outFile.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
    outFile.write(reinterpret_cast<const char*>(&all_results[0]), sz * sizeof(all_results[0]));
    outFile.close();
}
