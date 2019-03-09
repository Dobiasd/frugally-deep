// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#include "fdeep/fdeep.hpp"

int main()
{
    std::vector<std::string> model_paths = {
        "test_model_small.json",
        "test_model_variable.json",
        "test_model_sequential.json",
        "test_model_recurrent.json",
        "test_model_lstm.json",
        "test_model_gru.json",
        "test_model_full.json",
        "densenet121.json",
        "densenet169.json",
        "densenet201.json",
        //"inceptionresnetv2.json", // lambda
        "inceptionv3.json",
        "mobilenet.json",
        "mobilenetv2.json",
        "nasnetlarge.json",
        "nasnetmobile.json",
        "resnet50.json",
        "vgg16.json",
        "vgg19.json",
        "xception.json"
    };

    bool error = false;

    for (const auto& model_path : model_paths)
    {
        std::cout << "----" << std::endl;
        std::cout << model_path << std::endl;
        #ifdef NDEBUG
        try
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
        catch (const std::exception& e)
        {
            std::cerr << "ERROR: " << e.what() << std::endl;
            error = true;
        }
        #else
            const auto model = fdeep::load_model(model_path, true);
        #endif
    }

    if (error)
    {
        std::cout << "There were errors." << std::endl;
        return 1;
    }
    std::cout << "All imports and test OK." << std::endl;
}
