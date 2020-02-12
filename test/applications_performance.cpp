// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#include "fdeep/fdeep.hpp"

int main()
{
    std::vector<std::string> model_paths = {
        "test_model_exhaustive.json"
    };


    for (const auto& model_path : model_paths)
    {
        std::cout << "----" << std::endl;
        std::cout << model_path << std::endl;

        {
            const auto model = fdeep::load_model(model_path, true);
            const std::size_t warm_up_runs = 3;
            const std::size_t test_runs = 5;
            for (std::size_t i = 0; i < warm_up_runs; ++i)
            {
                const double duration = model.test_speed();
                std::cout << "Forward pass took "
                    << duration << " s." << std::endl;
            }
            double duration_sum = 0;
            std::cout << "Starting performance measurements." << std::endl;
            for (std::size_t i = 0; i < test_runs; ++i)
            {
                const double duration = model.test_speed();
                duration_sum += duration;
                std::cout << "Forward pass took "
                    << duration << " s." << std::endl;
            }
            const double duration_avg =
                duration_sum / static_cast<double>(test_runs);
            std::cout << "Forward pass took "
                << duration_avg << " s on average." << std::endl;
        }
    }
}
