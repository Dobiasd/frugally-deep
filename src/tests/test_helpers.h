// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <iostream>

#include "opencv_helpers.h"

#include "frugally_deep/frugally_deep.h"

#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <cassert>
#include <fstream>
#include <iostream>

inline std::string frame_string(const std::string& str)
{
    return
        std::string(str.size(), '-') + "\n" +
        str + "\n" +
        std::string(str.size(), '-');
}

inline bool file_exists(const std::string& file_path)
{
    return static_cast<bool>(std::ifstream(file_path));
}

std::vector<std::string> list_JPEGs(const std::string& dir_path)
{
    using namespace boost::filesystem;
    return fplus::transform_and_keep_justs([](const directory_entry& entry) -> fplus::maybe<std::string>
    {
        if (!boost::filesystem::is_regular_file(entry))
            return {};
        const auto file_name_str = entry.path().filename().string();
        if (!fplus::is_suffix_of(std::string(".JPEG"), file_name_str) &&
            !fplus::is_suffix_of(std::string(".jpeg"), file_name_str) &&
            !fplus::is_suffix_of(std::string(".jpg"), file_name_str) &&
            !fplus::is_suffix_of(std::string(".JPG"), file_name_str))
            return {};
		return fplus::replace_elems( '\\', '/', entry.path().string());;
    }, std::vector<directory_entry>(directory_iterator(path(dir_path)), {}));
}

inline fd::matrix3d load_col_image_as_matrix3d(const std::string& file_path)
{
    assert(file_exists(file_path));
    cv::Mat img_uchar = cv::imread(file_path, cv::IMREAD_COLOR);
    cv::Mat img = uchar_img_to_float_img(img_uchar);
    return cv_bgr_img_float_to_matrix3d(img);
}

inline cv::Mat resize_cv_image(int height, int width, const cv::Mat& img)
{
    cv::Mat result;
    cv::resize(img, result, cv::Size(width, height));
    return result;
}

inline fd::matrix3d load_col_image_as_matrix3d(
    std::size_t height, std::size_t width, const std::string& file_path)
{
    assert(file_exists(file_path));
    cv::Mat img_uchar = cv::imread(file_path, cv::IMREAD_COLOR);
    img_uchar = resize_cv_image(
        static_cast<int>(height), static_cast<int>(width), img_uchar);
    cv::Mat img = uchar_img_to_float_img(img_uchar);
    return cv_bgr_img_float_to_matrix3d(img);
}

inline fd::matrix3d load_col_image_as_matrix3d(
    std::size_t y0, std::size_t x0,
    std::size_t height, std::size_t width,
    const std::string& file_path)
{
    assert(file_exists(file_path));
    cv::Mat img_uchar = cv::imread(file_path, cv::IMREAD_COLOR);
    img_uchar = img_uchar(
        cv::Rect(
            cv::Point(static_cast<int>(x0), static_cast<int>(y0)),
            cv::Size(static_cast<int>(width), static_cast<int>(height))));
    cv::Mat img = uchar_img_to_float_img(img_uchar);
    return cv_bgr_img_float_to_matrix3d(img);
}

inline fd::matrix3d load_gray_image_as_matrix3d(const std::string& file_path)
{
    assert(file_exists(file_path));
    cv::Mat img_uchar = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img = uchar_img_to_float_img(img_uchar);
    return cv_gray_img_float_to_matrix3d(img);
}
