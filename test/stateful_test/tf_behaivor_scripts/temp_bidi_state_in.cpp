#include "fdeep/fdeep.hpp"
#include <fstream>

using namespace fdeep;

int main()
{

    // x_in = np.random.normal(0,10,sequence_length)
    // x_in = np.asarray([1,0,0])
    // x_in = x_in.reshape( (1, sequence_length, feature_dim) )

    // fwd_initial_h = np.asarray(2.75).reshape(1,1)
    // fwd_initial_c = np.asarray(1.3).reshape(1,1)
    // bwd_initial_h = np.asarray(-2.0).reshape(1,1)
    // bwd_initial_c = np.asarray(-1.2).reshape(1,1)

    const fdeep::float_vec x_inf_0 = {1.0, 0.0, 0.0};
    const fdeep::float_vec state_0 = {2.75};
    const fdeep::float_vec state_1 = {1.3};
    const fdeep::float_vec state_2 = {-2.0};
    const fdeep::float_vec state_3 = {-1.2};

    const shared_float_vec xt0(fplus::make_shared_ref<float_vec>(x_inf_0));
    const shared_float_vec st0(fplus::make_shared_ref<float_vec>(state_0));
    const shared_float_vec st1(fplus::make_shared_ref<float_vec>(state_1));
    const shared_float_vec st2(fplus::make_shared_ref<float_vec>(state_2));
    const shared_float_vec st3(fplus::make_shared_ref<float_vec>(state_3));

    const tensor test_in_0(tensor_shape(3, 1), xt0);
    const tensor test_state_0(tensor_shape(static_cast<std::size_t>(1)), st0);
    const tensor test_state_1(tensor_shape(static_cast<std::size_t>(1)), st1);
    const tensor test_state_2(tensor_shape(static_cast<std::size_t>(1)), st2);
    const tensor test_state_3(tensor_shape(static_cast<std::size_t>(1)), st3);


    std::cout << "loading models" << std::endl;
    auto stateful_model = load_model("temp_stateful.json");
    auto stateless_model = load_model("temp_stateless.json");

    // input for GRU: {test_in_0, test_state_0, test_state_2};
    // input for LSTM: {test_in_0, test_state_0, test_state_1, test_state_2, test_state_3}

    // A
    std::cout << "starting A" << std::endl;
    auto non_stateful_out = stateless_model.predict({test_in_0, test_state_0, test_state_1, test_state_2, test_state_3});
    auto stateful_out = stateful_model.predict_stateful({test_in_0, test_state_0, test_state_1, test_state_2, test_state_3});
    std::cout << "Non-Stateful" << std::endl;
    std::cout << fdeep::show_tensors(non_stateful_out) << std::endl;
    std::cout << "Stateful" << std::endl;
    std::cout << fdeep::show_tensors(stateful_out) << std::endl;

    // B
    std::cout << "starting B" << std::endl;
    non_stateful_out = stateless_model.predict({test_in_0, test_state_0, test_state_1, test_state_2, test_state_3});
    stateful_out = stateful_model.predict_stateful({test_in_0, test_state_0, test_state_1, test_state_2, test_state_3});
    std::cout << "Non-Stateful" << std::endl;
    std::cout << fdeep::show_tensors(non_stateful_out) << std::endl;
    std::cout << "Stateful" << std::endl;
    std::cout << fdeep::show_tensors(stateful_out) << std::endl;

    // C
    std::cout << "starting C" << std::endl;
    // stateful_model.reset_states();
    non_stateful_out = stateless_model.predict({test_in_0, test_state_0, test_state_1, test_state_2, test_state_3});
    stateful_out = stateful_model.predict_stateful({test_in_0, test_state_0, test_state_1, test_state_2, test_state_3});
    std::cout << "Non-Stateful" << std::endl;
    std::cout << fdeep::show_tensors(non_stateful_out) << std::endl;
    std::cout << "Stateful" << std::endl;
    std::cout << fdeep::show_tensors(stateful_out) << std::endl;
}

