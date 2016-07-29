// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.h>

#include <cassert>
#include <fstream>
#include <iostream>

// http://stackoverflow.com/a/21802936
inline std::vector<unsigned char> read_file(const std::string& filename,
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

inline fd::input_with_output parse_cifar_10_bin_line(
    const std::vector<unsigned char>& vec)
{
    assert(vec.size() == 3073);
    fd::matrix3d output(fd::size3d(1, 10, 1), fd::float_vec(10, 0));
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

inline fd::input_with_output_vec load_cifar_10_bin_file(const std::string& file_path,
		bool mini_version)
{
    std::size_t mini_version_img_count = 3;
	std::size_t max_bytes = mini_version ? 3073 * mini_version_img_count : 0;
    const auto bytes = read_file(file_path, max_bytes);
    const auto lines = fplus::split_every(3073, bytes);
    assert((mini_version && lines.size() == mini_version_img_count) ||
        lines.size() == 10000);
    return fplus::transform(parse_cifar_10_bin_line, lines);
}

inline fd::input_with_output_vec load_cifar_10_bin_training(
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

inline fd::input_with_output_vec load_cifar_10_bin_test(
    const std::string& base_directory,
	bool mini_version)
{
    return load_cifar_10_bin_file(base_directory + "/test_batch.bin", mini_version);
}

inline fd::classification_dataset load_cifar_10_bin(
    const std::string& base_directory,
	bool mini_version_training = false,
    bool mini_version_test = false)
{
    return {
        load_cifar_10_bin_training(base_directory, mini_version_training),
        load_cifar_10_bin_test(base_directory, mini_version_test)};
}
