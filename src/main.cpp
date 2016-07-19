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
    fd::matrix3d output(fd::size3d(1, 1, 10));
    output.set(0, 0, vec[0], 1);
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

void lenna_filter_test()
{
    cv::Mat img_uchar = cv::imread("images/lenna_512x512.png", cv::IMREAD_COLOR);
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
    cv::imwrite("lenna_512x512_filtered1.png", filtered1);

    cv::Mat filtered2 = filter2Ds_via_net(img, {kernel_scharr_x, kernel_scharr_x});
    filtered2 = shrink_via_net(filtered2, 2);
    filtered2 = grow_via_net(filtered2, 2);
    filtered2 = normalize_float_img(filtered2);
    filtered2 = float_img_to_uchar_img(filtered2);
    cv::imwrite("lenna_512x512_filtered2.png", filtered2);
}

void xor_as_net_test()
{
    std::cout << "xor_as_net_test" << std::endl;

    using namespace fd;

    layer_ptrs layers = {
        fc(2, 4),
        tanh(size3d(1, 1, 4)),
        fc(4, 4),
        tanh(size3d(1, 1, 4)),
        fc(4, 1),
        tanh(size3d(1, 1, 1)),
        };

    layer_ptrs layers_min = {
        fc(2, 2),
        tanh(size3d(1, 1, 2)),
        fc(2, 1),
        tanh(size3d(1, 1, 1)),
        };

    auto xor_net = net(layers);
    std::cout << "net.param_count() " << xor_net->param_count() << std::endl;

    input_with_output_vec xor_table =
    {
       {{size3d(1,1,2), {0, 0}}, {size3d(1,1,1), {0}}},
       {{size3d(1,1,2), {0, 1}}, {size3d(1,1,1), {1}}},
       {{size3d(1,1,2), {1, 0}}, {size3d(1,1,1), {1}}},
       {{size3d(1,1,2), {1, 1}}, {size3d(1,1,1), {0}}},
    };

    classification_dataset classifcation_data =
    {
        xor_table,
        xor_table
    };

    xor_net->set_params(randomly_change_params(xor_net->get_params()));
    train(xor_net, classifcation_data.training_data_, 10000, 0.1f, 0.01f);
    test(xor_net, classifcation_data.test_data_);
}

fd::matrix3d load_image_as_matrix3d(const std::string& file_path)
{
    cv::Mat img_uchar = cv::imread(file_path, cv::IMREAD_COLOR);
    cv::Mat img = uchar_img_to_float_img(img_uchar);
    return cv_bgr_img_float_to_matrix3d(img);
}

fd::classification_dataset load_gradient_dataset(const std::string& base_dir)
{
    fd::input_with_output_vec image_list =
    {
       {load_image_as_matrix3d(base_dir + "/x/001.png"), {fd::size3d(1,1,2), {1,0}}},
       {load_image_as_matrix3d(base_dir + "/x/002.png"), {fd::size3d(1,1,2), {1,0}}},
       {load_image_as_matrix3d(base_dir + "/x/003.png"), {fd::size3d(1,1,2), {1,0}}},
       {load_image_as_matrix3d(base_dir + "/y/001.png"), {fd::size3d(1,1,2), {0,1}}},
       {load_image_as_matrix3d(base_dir + "/y/002.png"), {fd::size3d(1,1,2), {0,1}}},
       {load_image_as_matrix3d(base_dir + "/y/003.png"), {fd::size3d(1,1,2), {0,1}}}
    };

    fd::classification_dataset classifcation_data =
    {
        image_list,
        image_list
    };

    return classifcation_data;
}

void gradients_classification_test()
{
    auto classifcation_data = load_gradient_dataset("images/datasets/classification/gradients");

    using namespace fd;

    layer_ptrs layers = {
        conv(size3d(3, 32, 32), size2d(3, 3), 2, 1), tanh(size3d(2, 32, 32)),
        max_pool(size3d(2, 32, 32), 32),
        flatten(size3d(2, 1, 1)),
        softmax(size3d(1, 1, 2))
        };

    auto gradnet = net(layers);
    std::cout << "net.param_count() " << gradnet->param_count() << std::endl;
    gradnet->set_params(randomly_change_params(gradnet->get_params()));
    train(gradnet, classifcation_data.training_data_, 1000, 0.001f, 0.3f);
    test(gradnet, classifcation_data.test_data_);
}

void cifar_10_classification_test()
{
    std::cout << "loading cifar-10 ..." << std::flush;
    auto classifcation_data = load_cifar_10_bin("./stuff/cifar-10-batches-bin", true);
    std::cout << " done" << std::endl;
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

    layer_ptrs layers = {
        conv(size3d(3, 32, 32), size2d(1, 1), 8, 1), tanh(size3d(8, 32, 32)),
        bottleneck_sandwich_dims_individual(
            size3d(8, 32, 32),
            size2d(3, 3),
            tanh(size3d(4, 32, 32)),
            tanh(size3d(8, 32, 32))),
        max_pool(size3d(8, 32, 32), 2),

        conv(size3d(8, 16, 16), size2d(3, 3), 16, 1), tanh(size3d(16, 16, 16)),
        bottleneck_sandwich_dims_individual(
            size3d(16, 16, 16),
            size2d(3, 3),
            tanh(size3d(8, 16, 16)),
            tanh(size3d(16, 16, 16))),
        max_pool(size3d(16, 16, 16), 2),

        conv(size3d(16, 8, 8), size2d(3, 3), 32, 1), tanh(size3d(32, 8, 8)),
        bottleneck_sandwich_dims_individual(
            size3d(32, 8, 8),
            size2d(3, 3),
            tanh(size3d(16, 8, 8)),
            tanh(size3d(32, 8, 8))),
        max_pool(size3d(32, 8, 8), 2),

        conv(size3d(32, 4, 4), size2d(3, 3), 64, 1), tanh(size3d(64, 4, 4)),
        bottleneck_sandwich_dims_individual(
            size3d(64, 4, 4),
            size2d(3, 3),
            tanh(size3d(32, 4, 4)),
            tanh(size3d(64, 4, 4))),
        conv(size3d(64, 4, 4), size2d(1, 1), 32, 1), tanh(size3d(32, 4, 4)),
        bottleneck_sandwich_dims_individual(
            size3d(32, 4, 4),
            size2d(3, 3),
            tanh(size3d(16, 4, 4)),
            tanh(size3d(32, 4, 4))),
        conv(size3d(32, 4, 4), size2d(1, 1), 16, 1), tanh(size3d(16, 4, 4)),

        flatten(size3d(16, 4, 4)),
        fc(size3d(16, 4, 4).volume(), 64),
        tanh(size3d(1, 1, 64)),
        fc(64, 32),
        tanh(size3d(1, 1, 32)),
        fc(32, 10),
        tanh(size3d(1, 1, 10)),
        softmax(size3d(1, 1, 10))
        };

    layer_ptrs layers_tiny = {
        max_pool(size3d(3, 32, 32), 16),
        flatten(size3d(3, 2, 2)),
        fc(size3d(3, 2, 2).volume(), 10),
        tanh(size3d(1, 1, 10)),
        softmax(size3d(1, 1, 10))
        };

    layer_ptrs layers_very_tiny = {
        max_pool(size3d(3, 32, 32), 32),
        flatten(size3d(3, 1, 1)),
        fc(size3d(3, 1, 1).volume(), 10),
        tanh(size3d(1, 1, 10)),
        softmax(size3d(1, 1, 10))
        };

    auto tobinet = net(layers_very_tiny);
    std::cout << "net.param_count() " << tobinet->param_count() << std::endl;
    tobinet->set_params(randomly_change_params(tobinet->get_params()));
    train(tobinet, classifcation_data.training_data_, 100000, 0.001f, 200.6f);
    test(tobinet, classifcation_data.test_data_);
}

int main()
{
    //lenna_filter_test();
    //xor_as_net_test();
    gradients_classification_test();
    //cifar_10_classification_test();
}
