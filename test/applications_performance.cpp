// rnn_performance.cpp
#include <fdeep/fdeep.hpp>
int main()
{
    //for (std::size_t e = 2; e < 13; ++e)
    const std::size_t e = 12;
    {
        const std::size_t num_units = 1 << e;
        const auto filename = std::string("/home/tobias/Documents/coding/CPP/frugally-deep/experiments/rnn_performance/") + std::string("model_") + std::to_string(num_units) + ".h5.json";
        auto model = fdeep::load_model(filename);
        const auto inputs = model.generate_dummy_inputs();
        const std::size_t runs = 100;
        fplus::stopwatch stopwatch;
        for (std::size_t run = 0; run < runs; ++run)
        {
            model.predict_stateful(inputs);
        }
        std::cout << "num_units: " << num_units <<
            "; Average time per prediction: " <<
            stopwatch.elapsed() / static_cast<double>(runs) << " s" << std::endl;
    }
}
