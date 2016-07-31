// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "frugally_deep/frugally_deep.h"

#include <opencv2/opencv.hpp>

#include <cassert>

inline fd::matrix3d cv_bgr_img_float_to_matrix3d(const cv::Mat& img)
{
    assert(img.type() == CV_32FC3);
    fd::matrix3d m(fd::size3d(
        3,
        static_cast<std::size_t>(img.cols),
        static_cast<std::size_t>(img.rows)));
    for (std::size_t y = 0; y < m.size().height_; ++y)
    {
        for (std::size_t x = 0; x < m.size().width_; ++x)
        {
            cv::Vec3f col =
                img.at<cv::Vec3f>(static_cast<int>(y), static_cast<int>(x));
            for (std::size_t c = 0; c < 3; ++c)
            {
                m.set(c, y, x,
                    static_cast<float_t>(col[static_cast<int>(c)]));
            }
        }
    }
    return m;
}

inline fd::matrix3d cv_gray_img_float_to_matrix3d(const cv::Mat& img)
{
    assert(img.type() == CV_32FC1);
    fd::matrix3d m(fd::size3d(
        1,
        static_cast<std::size_t>(img.cols),
        static_cast<std::size_t>(img.rows)));
    for (std::size_t y = 0; y < m.size().height_; ++y)
    {
        for (std::size_t x = 0; x < m.size().width_; ++x)
        {
            float col = img.at<float>(
                static_cast<int>(y), static_cast<int>(x));
            m.set(0, y, x,
                static_cast<float_t>(col));
        }
    }
    return m;
}

inline fd::matrix3d cv_float_kernel_to_matrix3d(const cv::Mat& kernel,
    std::size_t depth, std::size_t z)
{
    assert(kernel.type() == CV_32FC1);
    fd::matrix3d m(fd::size3d(
        depth,
        static_cast<std::size_t>(kernel.cols),
        static_cast<std::size_t>(kernel.rows)));
    for (std::size_t y = 0; y < m.size().height_; ++y)
    {
        for (std::size_t x = 0; x < m.size().width_; ++x)
        {
            float val =
                kernel.at<float>(static_cast<int>(y), static_cast<int>(x));
            m.set(z, y, x, val);
        }
    }
    return m;
}

inline cv::Mat matrix3d_to_cv_bgr_img_float(const fd::matrix3d& m)
{
    assert(m.size().depth_ == 3);
    cv::Mat img(
        static_cast<int>(m.size().height_),
        static_cast<int>(m.size().width_),
        CV_32FC3);
    for (std::size_t y = 0; y < m.size().height_; ++y)
    {
        for (std::size_t x = 0; x < m.size().width_; ++x)
        {
            cv::Vec3f col(
                static_cast<float>(m.get(0, y, x)),
                static_cast<float>(m.get(1, y, x)),
                static_cast<float>(m.get(2, y, x)));
            img.at<cv::Vec3f>(static_cast<int>(y), static_cast<int>(x)) = col;
        }
    }
    return img;
}

inline cv::Mat matrix3d_to_cv_gray_img_float(const fd::matrix3d& m)
{
    assert(m.size().depth_ == 1);
    cv::Mat img(
        static_cast<int>(m.size().height_),
        static_cast<int>(m.size().width_),
        CV_32FC1);
    for (std::size_t y = 0; y < m.size().height_; ++y)
    {
        for (std::size_t x = 0; x < m.size().width_; ++x)
        {
            float col(static_cast<float>(m.get(0, y, x)));
            img.at<float>(static_cast<int>(y), static_cast<int>(x)) = col;
        }
    }
    return img;
}

inline fd::layer_ptr cv_kernel_to_layer(const fd::size3d& in_size,
    const cv::Mat& cv_kernel)
{
    std::vector<fd::filter> filters = {
        fd::filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 0), 0),
        fd::filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 1), 0),
        fd::filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 2), 0)
    };
    auto size_without_depth = [](const fd::filter& f) -> fd::size2d
    {
        return fd::size2d(f.get_matrix3d().size().height_, f.get_matrix3d().size().width_);
    };
    fd::layer_ptr kernel_layer = std::make_shared<fd::convolutional_layer>(
        in_size,
        size_without_depth(filters[0]),
        filters.size(),
        1);
    auto all_filter_params = fplus::transform_and_concat(
        [](const fd::filter& f)
        {
            return f.get_params();
        }, filters);
    assert(kernel_layer->param_count() == all_filter_params.size());
    kernel_layer->set_params(all_filter_params);
    assert(kernel_layer->get_params() == all_filter_params);
    return kernel_layer;
}

inline cv::Mat filter2D_via_net(const cv::Mat& img, const cv::Mat& cv_kernel)
{
    fd::matrix3d in_vol = cv_bgr_img_float_to_matrix3d(img);
    std::vector<fd::layer_ptr> layers = {
        cv_kernel_to_layer(in_vol.size(), cv_kernel)
    };
    fd::multi_layer_net net(layers);
    auto out_vol = net.forward_pass(in_vol);
    cv::Mat result = matrix3d_to_cv_bgr_img_float(out_vol);
    return result;
}

inline cv::Mat filter2Ds_via_net(const cv::Mat& img,
    const std::vector<cv::Mat>& cv_kernels)
{
    fd::matrix3d in_vol = cv_bgr_img_float_to_matrix3d(img);
    auto layers = fplus::transform(
        fplus::bind_1st_of_2(cv_kernel_to_layer, in_vol.size()),
        cv_kernels);
    fd::multi_layer_net net(layers);
    auto out_vol = net.forward_pass(in_vol);
    //std::cout << fplus::show_cont(net.get_params()) << std::endl;
    cv::Mat result = matrix3d_to_cv_bgr_img_float(out_vol);
    return result;
}

inline cv::Mat shrink_via_net(const cv::Mat& img, std::size_t scale_factor)
{
    fd::matrix3d in_vol = cv_bgr_img_float_to_matrix3d(img);
    fd::avg_pool_layer net(in_vol.size(), scale_factor);
    auto out_vol = net.forward_pass(in_vol);
    cv::Mat result = matrix3d_to_cv_bgr_img_float(out_vol);
    return result;
}

inline cv::Mat grow_via_net(const cv::Mat& img, std::size_t scale_factor)
{
    fd::matrix3d in_vol = cv_bgr_img_float_to_matrix3d(img);
    fd::unpool_layer net(in_vol.size(), scale_factor);
    auto out_vol = net.forward_pass(in_vol);
    cv::Mat result = matrix3d_to_cv_bgr_img_float(out_vol);
    return result;
}

inline cv::Mat uchar_img_to_float_img(const cv::Mat& uchar_img)
{
    cv::Mat result;
    if (uchar_img.type() == CV_8UC3)
    {
        uchar_img.convertTo(result, CV_32FC3, 1.0f/255.0f);
        return result;
    }
    else if (uchar_img.type() == CV_8UC1)
    {
        uchar_img.convertTo(result, CV_32FC1, 1.0f/255.0f);
        return result;
    }
    assert(false);
    return result;
}

inline cv::Mat float_img_to_uchar_img(const cv::Mat& floatImg)
{
    cv::Mat result;
    if (floatImg.type() == CV_32FC3)
    {
        floatImg.convertTo(result, CV_8UC3, 255.0f);
        return result;
    }
    else if (floatImg.type() == CV_32FC1)
    {
        floatImg.convertTo(result, CV_8UC1, 255.0f);
        return result;
    }
    assert(false);
    return result;
}

inline cv::Mat float_img_bgr_rgb_switch(const cv::Mat& img)
{
    assert(img.type() == CV_32FC3);
    cv::Mat result;
    cv::cvtColor(img, result, CV_RGB2BGR);
    return result;
}

inline cv::Mat normalize_float_img(const cv::Mat& floatImg)
{
    assert(floatImg.type() == CV_32FC3 || floatImg.type() == CV_32FC1);
    cv::Mat result;
    cv::normalize(floatImg, result, 0, 1, cv::NORM_MINMAX);
    return result;
}

inline void save_matrix3d_image(const fd::matrix3d& m, const std::string& path)
{
    if (m.size().depth_ == 3)
    {
        cv::Mat img = float_img_to_uchar_img(matrix3d_to_cv_bgr_img_float(m));
        cv::imwrite(path, img);
    }
    else if (m.size().depth_ == 1)
    {
        cv::Mat img = float_img_to_uchar_img(matrix3d_to_cv_gray_img_float(m));
        cv::imwrite(path, img);
    }
    else
    {
        assert(false); // matrix must have depth 1 or 3
    }
}

inline fd::matrix3d load_matrix3d_image_gray(const std::string& path)
{
    cv::Mat img = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    img = uchar_img_to_float_img(img);
    return cv_gray_img_float_to_matrix3d(img);
}

inline fd::matrix3d load_matrix3d_image_bgr(const std::string& path)
{
    cv::Mat img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    img = uchar_img_to_float_img(img);
    return cv_bgr_img_float_to_matrix3d(img);
}
