#include <fdeep/fdeep.hpp>

void write_to_results(const std::vector<float>& x, std::vector<float>& results){
	results.insert(std::end(results), std::begin(x), std::end(x));
	return;
}

int main()
{
	// "GRU_nonstateful_no_init_state.json", 
	// "GRU_nonstateful_init_state.json", 
	// "GRU_stateful_no_init_state.json", 
	// "GRU_stateful_init_state.json", 
	// "LSTM_nonstateful_no_init_state.json", 
	// "LSTM_nonstateful_init_state.json", 
	// "LSTM_stateful_no_init_state.json", 
	// "LSTM_stateful_init_state.json", 
	// "bidi-GRU_nonstateful_no_init_state.json", 
	// "bidi-GRU_nonstateful_init_state.json", 
	// "bidi-GRU_stateful_no_init_state.json", 
	// "bidi-GRU_stateful_init_state.json", 
	// "bidi-LSTM_nonstateful_no_init_state.json", 
	// "bidi-LSTM_nonstateful_init_state.json", 
	// "bidi-LSTM_stateful_no_init_state.json", 
	// "bidi-LSTM_stateful_init_state.json"

    const std::vector<float> x_inf_0 = {2.1, -1.2, 3.14, 1.2};
	const std::vector<float> x_inf_1 = {1, 3, -2, 10};
	const std::vector<float> state_0 = {1.1, -2.1};
	const std::vector<float> state_1 = {2.7, 3.1};
	const std::vector<float> state_2 = {-2.5, 3.0};
	const std::vector<float> state_3 = {-2.0, -10.0};
	std::vector<float> all_results = {};
	std::vector<float> one_result = {};


    const fdeep::shared_float_vec xt0(fplus::make_shared_ref<fdeep::float_vec>(x_inf_0));
	const fdeep::shared_float_vec xt1(fplus::make_shared_ref<fdeep::float_vec>(x_inf_1));
    const fdeep::shared_float_vec st0(fplus::make_shared_ref<fdeep::float_vec>(state_0));
    const fdeep::shared_float_vec st1(fplus::make_shared_ref<fdeep::float_vec>(state_1));
    const fdeep::shared_float_vec st2(fplus::make_shared_ref<fdeep::float_vec>(state_2));
    const fdeep::shared_float_vec st3(fplus::make_shared_ref<fdeep::float_vec>(state_3));
	
	// fdeep::tensor5 test_in_0(fdeep::shape5(1, 1, 1, 4, 1), xt0);
    fdeep::tensor5 test_in_1(fdeep::shape5(1, 1, 1, 4, 1), xt1);
    fdeep::tensor5 test_state_0(fdeep::shape5(1, 1, 1, 2, 1), st0);
    fdeep::tensor5 test_state_1(fdeep::shape5(1, 1, 1, 2, 1), st1);
    fdeep::tensor5 test_state_2(fdeep::shape5(1, 1, 1, 2, 1), st2);
    fdeep::tensor5 test_state_3(fdeep::shape5(1, 1, 1, 2, 1), st3);

	fdeep::tensor5 test_in_0(fdeep::shape5(1, 1, 1, 4, 1),  static_cast<fdeep::float_type>(0));
	for(size_t idx = 0; idx < 4; ++ idx){
	    test_in_0.set(0, 0, 0, idx, 1, x_inf_0[idx]);
	}

	const auto GRU_nonstateful_no_init_state = fdeep::load_model("./models/GRU_nonstateful_no_init_state.json", true, fdeep::cout_logger, static_cast<fdeep::float_type>(0.00001));
	const auto result_0 = GRU_nonstateful_no_init_state.predict({test_in_0});
	const auto result_1 = GRU_nonstateful_no_init_state.predict({test_in_1});
	one_result = *result_0[0].as_vector();
	all_results.insert(std::end(all_results), std::begin(one_result), std::end(one_result));
	// write_to_results(*result_1[0].as_vector(), all_results);

	const auto in_seq = *test_in_0.as_vector();

	std::cout << "\n\nin_seq-vectorized ***" << std::endl;
	for(size_t idx = 0; idx < 4; ++ idx){
	    std::cout << in_seq[idx]  << std::endl;
	}

	std::cout << "\n\nin_seq-tensor ***" << std::endl;
	for(size_t idx = 0; idx < 4; ++ idx){
	    std::cout << test_in_0.get(0, 0, 0, idx, 1) << std::endl;
	}

	std::cout << "\n\nout_seq (vectorized) ***" << std::endl;
	for(size_t idx = 0; idx < one_result.size(); ++ idx){
	    std::cout << one_result[idx] << std::endl;
	}

}