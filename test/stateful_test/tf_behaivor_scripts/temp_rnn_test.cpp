#include "fdeep/fdeep.hpp"
#include <fstream>

using namespace fdeep;

int main()
{
    const fdeep::float_vec x_inf_0 = {1.0, 0.0, 0.0};
    const fdeep::float_vec state_0 = {10.0};

    const shared_float_vec xt0(fplus::make_shared_ref<float_vec>(x_inf_0));
    const shared_float_vec st0(fplus::make_shared_ref<float_vec>(state_0));

    std::cout << "convert to tensors" << std::endl;
    const tensor test_in_0(tensor_shape(3, 1), xt0);
    std::cout << "convert to tensors" << std::endl;
    const tensor test_state_0(tensor_shape(static_cast<std::size_t>(1)), st0);

    std::cout << "loading models" << std::endl;
    auto stateful_model = load_model("temp_stateful.json");
    auto stateless_model = load_model("temp_stateless.json");

    // A
    std::cout << "starting A" << std::endl;
    auto non_stateful_out = stateless_model.predict({test_in_0, test_state_0});
    auto stateful_out = stateful_model.predict_stateful({test_in_0, test_state_0});
    std::cout << "Non-Stateful" << std::endl;
    std::cout << fdeep::show_tensors(non_stateful_out) << std::endl;
    std::cout << "Stateful" << std::endl;
    std::cout << fdeep::show_tensors(stateful_out) << std::endl;

    // B
    non_stateful_out = stateless_model.predict({test_in_0, test_state_0});
    stateful_out = stateful_model.predict_stateful({test_in_0, test_state_0});
    std::cout << "Non-Stateful" << std::endl;
    std::cout << fdeep::show_tensors(non_stateful_out) << std::endl;
    std::cout << "Stateful" << std::endl;
    std::cout << fdeep::show_tensors(stateful_out) << std::endl;

    // C
    std::cout << "** RESETING STATES in STATEFUL MODEL **" << std::endl;
    stateful_model.reset_states();
    non_stateful_out = stateless_model.predict({test_in_0, test_state_0});
    stateful_out = stateful_model.predict_stateful({test_in_0, test_state_0});
    std::cout << "Non-Stateful" << std::endl;
    std::cout << fdeep::show_tensors(non_stateful_out) << std::endl;
    std::cout << "Stateful" << std::endl;
    std::cout << fdeep::show_tensors(stateful_out) << std::endl;

    //D
    non_stateful_out = stateless_model.predict({test_in_0, test_state_0});
    stateful_out = stateful_model.predict_stateful({test_in_0, test_state_0});
    std::cout << "Non-Stateful" << std::endl;
    std::cout << fdeep::show_tensors(non_stateful_out) << std::endl;
    std::cout << "Stateful" << std::endl;
    std::cout << fdeep::show_tensors(stateful_out) << std::endl;
}