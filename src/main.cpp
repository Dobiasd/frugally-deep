// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

// todo raus
#include <iostream>

#include "opencv_helpers.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

// http://stackoverflow.com/a/21802936
std::vector<unsigned char> read_file(const std::string& filename,
    std::size_t max_bytes)
{
    // open the file:
    std::ifstream file(filename, std::ios::binary);

    assert(file.good());

    // Stop eating new lines in binary mode!!!
    file.unsetf(std::ios::skipws);

    // get its size:
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    assert(fileSize == static_cast<std::streamoff>(30730000));
    if (max_bytes != 0)
    {
    	fileSize = static_cast<std::streamoff>(max_bytes);
    }
    file.seekg(0, std::ios::beg);

    // reserve memory and fill with zeroes
    std::vector<unsigned char> vec(static_cast<std::size_t>(fileSize), 0);

    // read the data:
    file.read(reinterpret_cast<char*>(&vec[0]), fileSize);

    return vec;
}

fd::input_with_output parse_cifar_10_bin_line(
    const std::vector<unsigned char>& vec)
{
    assert(vec.size() == 3073);
    fd::matrix3d output(fd::size3d(1, 10, 1));
    assert(vec[0] < 10);
    output.set(0, vec[0], 0, 1);
    fd::matrix3d input(fd::size3d(3, 32, 32));
    std::size_t vec_i = 0;
    for (std::size_t z = 0; z < input.size().depth_; ++z)
    {
        for (std::size_t y = 0; y < input.size().height_; ++y)
        {
            for (std::size_t x = 0; x < input.size().width_; ++x)
            {
                input.set(input.size().depth_ - (z + 1), y, x,
                    vec[++vec_i] / static_cast<float_t>(256));
            }
        }
    }
    return {input, output};
}

fd::input_with_output_vec load_cifar_10_bin_file(const std::string& file_path,
		bool mini_version)
{
    std::size_t mini_version_img_count = 1;
	std::size_t max_bytes = mini_version ? 3073 * mini_version_img_count : 0;
    const auto bytes = read_file(file_path, max_bytes);
    const auto lines = fplus::split_every(3073, bytes);
    assert((mini_version && lines.size() == mini_version_img_count) ||
        lines.size() == 10000);
    return fplus::transform(parse_cifar_10_bin_line, lines);
}

fd::input_with_output_vec load_cifar_10_bin_training(
    const std::string& base_directory,
	bool mini_version)
{
    return fplus::concat(std::vector<fd::input_with_output_vec>({
        load_cifar_10_bin_file(base_directory + "/data_batch_1.bin", mini_version),
        load_cifar_10_bin_file(base_directory + "/data_batch_2.bin", mini_version),
        load_cifar_10_bin_file(base_directory + "/data_batch_3.bin", mini_version),
        load_cifar_10_bin_file(base_directory + "/data_batch_4.bin", mini_version),
        load_cifar_10_bin_file(base_directory + "/data_batch_5.bin", mini_version)}));
}

fd::input_with_output_vec load_cifar_10_bin_test(
    const std::string& base_directory,
	bool mini_version)
{
    return load_cifar_10_bin_file(base_directory + "/test_batch.bin", mini_version);
}

fd::classification_dataset load_cifar_10_bin(
    const std::string& base_directory,
	bool mini_version = false)
{
    return {
        load_cifar_10_bin_training(base_directory, mini_version),
        load_cifar_10_bin_test(base_directory, mini_version)};
}

std::string frame_string(const std::string& str)
{
    return
        std::string(str.size(), '-') + "\n" +
        str + "\n" +
        std::string(str.size(), '-');
}

void lenna_filter_test()
{
    std::cout << frame_string("lenna_filter_test") << std::endl;
    cv::Mat img_uchar = cv::imread("test_images/lenna_512x512.png", cv::IMREAD_COLOR);
    cv::Mat img = uchar_img_to_float_img(img_uchar);

    cv::Mat kernel_scharr_x(cv::Size(3,3), CV_32FC1, cv::Scalar(0));
    kernel_scharr_x.at<float>(0,0) =   3.0f / 32.0f;
    kernel_scharr_x.at<float>(1,0) =  10.0f / 32.0f;
    kernel_scharr_x.at<float>(2,0) =   3.0f / 32.0f;
    kernel_scharr_x.at<float>(0,1) =   0.0f / 32.0f;
    kernel_scharr_x.at<float>(1,1) =   0.0f / 32.0f;
    kernel_scharr_x.at<float>(2,1) =   0.0f / 32.0f;
    kernel_scharr_x.at<float>(0,2) =  -3.0f / 32.0f;
    kernel_scharr_x.at<float>(1,2) = -10.0f / 32.0f;
    kernel_scharr_x.at<float>(2,2) =  -3.0f / 32.0f;

    cv::Mat kernel_blur(cv::Size(3,3), CV_32FC1, cv::Scalar(1.0f/9.0f));

    cv::Mat filtered1;
    cv::filter2D(img, filtered1, CV_32FC3, kernel_scharr_x);
    cv::filter2D(filtered1, filtered1, -1, kernel_scharr_x);
    cv::resize(filtered1, filtered1, cv::Size(0,0), 0.5, 0.5, cv::INTER_AREA);
    cv::resize(filtered1, filtered1, cv::Size(0,0), 2, 2, cv::INTER_NEAREST);
    filtered1 = normalize_float_img(filtered1);
    filtered1 = float_img_to_uchar_img(filtered1);
    cv::imwrite("stuff/lenna_512x512_filtered1.png", filtered1);

    cv::Mat filtered2 = filter2Ds_via_net(img, {kernel_scharr_x, kernel_scharr_x});
    filtered2 = shrink_via_net(filtered2, 2);
    filtered2 = grow_via_net(filtered2, 2);
    filtered2 = normalize_float_img(filtered2);
    filtered2 = float_img_to_uchar_img(filtered2);
    cv::imwrite("stuff/lenna_512x512_filtered2.png", filtered2);

    std::cout << frame_string("filtered images written to ./stuff/lenna_512x512_filtered*.png") << std::endl;
}

void xor_as_net_test()
{
    std::cout << frame_string("xor_as_net_test") << std::endl;

    using namespace fd;

    input_with_output_vec xor_table =
    {
       {{size3d(1,2,1), {0, 0}}, {size3d(1,1,1), {0}}},
       {{size3d(1,2,1), {0, 1}}, {size3d(1,1,1), {1}}},
       {{size3d(1,2,1), {1, 0}}, {size3d(1,1,1), {1}}},
       {{size3d(1,2,1), {1, 1}}, {size3d(1,1,1), {0}}},
    };

    classification_dataset classifcation_data =
    {
        xor_table,
        xor_table
    };

    classifcation_data = normalize_classification_dataset(classifcation_data, false);

    pre_layers layers = {
        fc(4),
        tanh(),
        fc(4),
        tanh(),
        fc(1),
        tanh(),
        };

    pre_layers layers_min = {
        fc(2),
        tanh(),
        fc(1),
        tanh(),
        };
    float_vec layers_min_good_params =
    {
         1, 1, -1, -1,
         0.5f, -1.5f,
         1, 1,
         1.5f
    };

    auto xor_net = net(layers_min)(size3d(1, 2, 1));
    std::cout << "net.param_count() " << xor_net->param_count() << std::endl;

    //xor_net->set_params(layers_min_good_params);
    xor_net->set_params(randomly_change_params(xor_net->get_params(), 0.1f));
    train(xor_net, classifcation_data.training_data_, 100000, 0.1f, 0.1f);
    test(xor_net, classifcation_data.test_data_);
}

bool file_exists(const std::string& file_path)
{
    return static_cast<bool>(std::ifstream(file_path));
}

fd::matrix3d load_col_image_as_matrix3d(const std::string& file_path)
{
    assert(file_exists(file_path));
    cv::Mat img_uchar = cv::imread(file_path, cv::IMREAD_COLOR);
    cv::Mat img = uchar_img_to_float_img(img_uchar);
    return cv_bgr_img_float_to_matrix3d(img);
}

fd::matrix3d load_gray_image_as_matrix3d(const std::string& file_path)
{
    assert(file_exists(file_path));
    cv::Mat img_uchar = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img = uchar_img_to_float_img(img_uchar);
    return cv_gray_img_float_to_matrix3d(img);
}

fd::classification_dataset load_gradient_dataset(const std::string& base_dir)
{
    fd::classification_dataset classifcation_data =
    {
        {
            {load_gray_image_as_matrix3d(base_dir + "/training/x/001.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/x/002.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/x/003.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/y/001.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/y/002.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/training/y/003.png"), {fd::size3d(1,2,1), {0,1}}}
        },
        {
            {load_gray_image_as_matrix3d(base_dir + "/test/x/001.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/x/002.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/x/003.png"), {fd::size3d(1,2,1), {1,0}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/y/001.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/y/002.png"), {fd::size3d(1,2,1), {0,1}}},
            {load_gray_image_as_matrix3d(base_dir + "/test/y/003.png"), {fd::size3d(1,2,1), {0,1}}}
        }
    };

    return classifcation_data;
}

void gradients_classification_test()
{
    std::cout << frame_string("gradients_classification_test") << std::endl;
    auto classifcation_data = load_gradient_dataset("test_images/datasets/classification/gradients");
    assert(!classifcation_data.training_data_.empty());
    assert(!classifcation_data.test_data_.empty());

    classifcation_data = normalize_classification_dataset(classifcation_data, false);

    using namespace fd;

    pre_layers layers = {
        conv(size2d(3, 3), 2, 1),
        elu(1),
        max_pool(32),
        flatten(),
        fc(2),
        //sigmoid(),
        softmax()
        };

    auto gradnet = net(layers)(size3d(1, 32, 32));
    std::cout << "net.param_count() " << gradnet->param_count() << std::endl;

    float_vec good_params =
    {
         3,  0,  -3,
        10,  0, -10,
         3,  0,  -3,
        0,
         3,  10,  3,
         0,   0,  0,
        -3, -10, -3,
        0,
        1,0,0,1,0,0
    };

    //gradnet->set_params(good_params);

    gradnet->set_params(randomly_change_params(gradnet->get_params(), 0.1f));

    train(gradnet, classifcation_data.training_data_, 10000, 0.01f, 0.1f);
    test(gradnet, classifcation_data.test_data_);
}

void cifar_10_classification_test()
{
    std::cout << frame_string("cifar_10_classification_test") << std::endl;
    std::cout << "loading cifar-10 ..." << std::flush;
    auto classifcation_data = load_cifar_10_bin("./stuff/cifar-10-batches-bin", true);
    std::cout << " done" << std::endl;

    classifcation_data = normalize_classification_dataset(classifcation_data, false);

    /*
    classifcation_data.training_data_ =
        fplus::sample(
            classifcation_data.training_data_.size() / 10000,
            classifcation_data.training_data_);
    classifcation_data.test_data_ =
        fplus::sample(
            classifcation_data.test_data_.size() / 1000,
            classifcation_data.test_data_);
     */

    using namespace fd;
    /*
    layer_ptrs layers = {
        conv(size3d(3, 32, 32), size2d(3, 3), 8, 1), leaky_relu(size3d(8, 32, 32), 0.01f),
        conv(size3d(8, 32, 32), size2d(3, 3), 8, 1), leaky_relu(size3d(8, 32, 32), 0.01f),
        max_pool(size3d(8, 32, 32), 2),
        conv(size3d(8, 16, 16), size2d(3, 3), 16, 1), leaky_relu(size3d(16, 16, 16), 0.01f),
        conv(size3d(16, 16, 16), size2d(3, 3), 16, 1), leaky_relu(size3d(16, 16, 16), 0.01f),
        max_pool(size3d(16, 16, 16), 2),
        conv(size3d(16, 8, 8), size2d(3, 3), 32, 1), leaky_relu(size3d(32, 8, 8), 0.01f),
        conv(size3d(32, 8, 8), size2d(3, 3), 32, 1), leaky_relu(size3d(32, 8, 8), 0.01f),
        max_pool(size3d(32, 8, 8), 2),
        conv(size3d(32, 4, 4), size2d(3, 3), 64, 1), leaky_relu(size3d(64, 4, 4), 0.01f),
        conv(size3d(64, 4, 4), size2d(3, 3), 64, 1), leaky_relu(size3d(64, 4, 4), 0.01f),
        conv(size3d(64, 4, 4), size2d(1, 1), 64, 1), leaky_relu(size3d(64, 4, 4), 0.01f),
        flatten(size3d(64, 4, 4)),
        fc(size3d(64, 4, 4).volume(), 256),
        fc(256, 256),
        fc(256, 10),
        softmax(size3d(1,1,10))
        };
    */

    pre_layers layers = {
        conv(size2d(1, 1), 8, 1), elu(1),
        conv(size2d(1, 1), 8, 1), elu(1),
        max_pool(2),

        conv(size2d(3, 3), 16, 1), elu(1),
        conv(size2d(3, 3), 16, 1), elu(1),
        max_pool(2),

        conv(size2d(3, 3), 32, 1), elu(1),
        conv(size2d(3, 3), 32, 1), elu(1),
        max_pool(2),

        conv(size2d(3, 3), 64, 1), elu(1),
        conv(size2d(3, 3), 64, 1), elu(1),
        max_pool(2),

        conv(size2d(1, 1), 128, 1), elu(1),
        conv(size2d(1, 1), 128, 1), elu(1),
        max_pool(2),

        conv(size2d(1, 1), 256, 1), elu(1),
        conv(size2d(1, 1), 128, 1), elu(1),
        conv(size2d(1, 1), 64, 1), elu(1),
        conv(size2d(1, 1), 32, 1), elu(1),
        conv(size2d(1, 1), 16, 1), elu(1),

        flatten(),
        fc(16),
        tanh(),
        fc(10),
        tanh(),
        softmax()
        };

    pre_layers layers_tiny = {
        max_pool(16),
        flatten(),
        fc(10),
        sigmoid(),
        softmax()
        };

    pre_layers layers_very_tiny = {
        max_pool(32),
        flatten(),
        fc(10),
        sigmoid(),
        softmax()
        };

    auto tobinet = net(layers)(size3d(3, 32, 32));
    std::cout << "net.param_count() " << tobinet->param_count() << std::endl;
    tobinet->set_params(randomly_change_params(tobinet->get_params(), 0.1f));
    train(tobinet, classifcation_data.training_data_, 100000, 0.001f, 0.1f);
    test(tobinet, classifcation_data.test_data_);
}

fd::float_t relative_error(fd::float_t x, fd::float_t y)
{
    const auto divisor = fplus::max(x, y);
    if (divisor < 0.0001f)
        return 0;
    else
        return fplus::abs_diff(x, y) / divisor;
}

bool gradients_equal(fd::float_t max_diff, const fd::float_vec& xs, const fd::float_vec& ys)
{
    return fplus::all_by(
        fplus::is_less_or_equal_than<fd::float_t>(max_diff),
        fplus::zip_with(relative_error, xs, ys));
}

// http://cs231n.github.io/neural-networks-3/#gradcheck
void gradient_check_backprop_implementation()
{
    std::cout << frame_string("gradient_check_backprop_implementation") << std::endl;
    using namespace fd;
    input_with_output_vec training_data =
    {
        {{size3d(1,1,2), {-1.31,  2.26}}, {size3d(1,3,1), { 0.131, 0.241,0.576}}},
        {{size3d(1,1,2), {-5.12,  6.13}}, {size3d(1,3,1), { 0.214,-0.452,0.157}}},
        {{size3d(1,1,2), { 2.63, -3.85}}, {size3d(1,3,1), {-0.413,-0.003,1.752}}},
        //{{size3d(1,1,1), {2}}, {size3d(1,5,1), {1.5,2.61,3.781,4.2,5.519}}}
        //{{size3d(1,1,1), {1}}, {size3d(1,2,1), {1,3}}},
        //{{size3d(1,1,1), {1}}, {size3d(1,2,1), {7,2}}}
        //{{size3d(1,1,1), {1}}, {size3d(1,3,1), {1,2,3}}}
    };
    auto net_001 = net(
    {
        flatten(),
        fc(2),
        softplus(),
        fc(2),
        identity(),
        fc(2),
        relu(),
        fc(2),
        leaky_relu(0.03),
        fc(2),
        elu(1),
        fc(2),
        erf(),
        fc(2),
        step(),
        fc(2),
        fast_sigmoid(),
        fc(4),
        sigmoid(),
        fc(3),
        tanh(),
        //softmax()
    })(size3d(1, 1, 2));

    const auto show_one_value = fplus::show_float_fill_left<fd::float_t>(' ', 7 + 4, 7);
    const auto show_gradient = [show_one_value](const float_vec& xs) -> std::string
    {
        return fplus::show_cont(fplus::transform(show_one_value, xs));
    };

    for (int i = 0; i < 100; ++i)
    {
        net_001->set_params(randomly_change_params(net_001->get_params(), 0.1f));
        auto gradient = calc_net_gradient(net_001, training_data);
        auto gradient_backprop = calc_net_gradient_backprop(net_001, training_data);
        if (!gradients_equal(0.001, gradient_backprop, gradient))
        {
            std::cout << "todo remove params            " << show_gradient(net_001->get_params()) << std::endl;
            std::cout << "todo remove gradient          " << show_gradient(gradient) << std::endl;
            std::cout << "todo remove gradient_backprop " << show_gradient(gradient_backprop) << std::endl;
            std::cout << "todo remove abs_diff          " << show_gradient(fplus::zip_with(relative_error, gradient, gradient_backprop)) << std::endl;
            assert(false);
        }
    }
    std::cout << frame_string("Backprop implementation seems to be correct.") << std::endl;
}

int main()
{
    gradient_check_backprop_implementation();
    lenna_filter_test();
    xor_as_net_test();
    gradients_classification_test();
    //cifar_10_classification_test();
}
