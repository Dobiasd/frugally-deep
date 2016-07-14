#include "conv_net.h"
#include "conv_layer.h"
#include "filter.h"
#include "matrix3d.h"

#include <fplus/fplus.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <iostream>
#include <random>


matrix3d cv_bgr_img_float_to_matrix3d(const cv::Mat& img)
{
    assert(img.type() == CV_32FC3);
    matrix3d m(size3d(
        3,
        static_cast<std::size_t>(img.cols),
        static_cast<std::size_t>(img.rows)));
    for (std::size_t y = 0; y < m.size().height(); ++y)
    {
        for (std::size_t x = 0; x < m.size().width(); ++x)
        {
            cv::Vec3f col =
                img.at<cv::Vec3f>(static_cast<int>(y), static_cast<int>(x));
            for (std::size_t c = 0; c < 3; ++c)
            {
                m.set(c, y, x,
                    static_cast<float>(col[static_cast<int>(c)]));
            }
        }
    }
    return m;
}

matrix3d cv_float_kernel_to_matrix3d(const cv::Mat& kernel,
    std::size_t depth, std::size_t z)
{
    assert(kernel.type() == CV_32FC1);
    matrix3d m(size3d(
        depth,
        static_cast<std::size_t>(kernel.cols),
        static_cast<std::size_t>(kernel.rows)));
    for (std::size_t y = 0; y < m.size().height(); ++y)
    {
        for (std::size_t x = 0; x < m.size().width(); ++x)
        {
            float val =
                kernel.at<float>(static_cast<int>(y), static_cast<int>(x));
            m.set(z, y, x, val);
        }
    }
    return m;
}

cv::Mat matrix3d_to_cv_bgr_img_float(const matrix3d& m)
{
    assert(m.size().depth() == 3);
    cv::Mat img(
        static_cast<int>(m.size().height()),
        static_cast<int>(m.size().width()),
        CV_32FC3);
    for (std::size_t y = 0; y < m.size().height(); ++y)
    {
        for (std::size_t x = 0; x < m.size().width(); ++x)
        {
            cv::Vec3f col(m.get(0, y, x), m.get(1, y, x), m.get(2, y, x));
            img.at<cv::Vec3f>(static_cast<int>(y), static_cast<int>(x)) = col;
        }
    }
    return img;
}


filter random_filter(const size3d& size)
{
    /*
    std::random_device rd;
    std::mt19937 mt(rd());
    std::size_t parameter_count = size.area();
    float multiplicator = 1.0f / static_cast<float>(parameter_count);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    */
    // todo befuellen
    return filter(matrix3d(size));
}


layer_ptr cv_kernel_to_layer(const cv::Mat& cv_kernel)
{
    std::vector<filter> filters = {
        filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 0)),
        filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 1)),
        filter(cv_float_kernel_to_matrix3d(cv_kernel, 3, 2))
    };
    return std::make_shared<conv_layer>(filters);
}

cv::Mat filter2DNet(const cv::Mat& img, const cv::Mat& cv_kernel)
{
    matrix3d in_vol = cv_bgr_img_float_to_matrix3d(img);
    std::vector<layer_ptr> layers = {cv_kernel_to_layer(cv_kernel)};
    conv_net net = conv_net(layers);
    auto out_vol = net.forward_pass(in_vol);
    cv::Mat result = matrix3d_to_cv_bgr_img_float(out_vol);
    return result;
}

cv::Mat filter2DsNet(const cv::Mat& img, const std::vector<cv::Mat>& cv_kernels)
{
    matrix3d in_vol = cv_bgr_img_float_to_matrix3d(img);
    auto layers = fplus::transform(cv_kernel_to_layer, cv_kernels);
    conv_net net = conv_net(layers);
    auto out_vol = net.forward_pass(in_vol);
    cv::Mat result = matrix3d_to_cv_bgr_img_float(out_vol);
    return result;
}

cv::Mat uchar_img_to_float_img(const cv::Mat& uchar_img)
{
    assert(uchar_img.type() == CV_8UC3);
    cv::Mat result;
    uchar_img.convertTo(result, CV_32FC3, 1.0f/255.0f);
    return result;
}

cv::Mat floatImgToUCharImg(const cv::Mat& floatImg)
{
    assert(floatImg.type() == CV_32FC3);
    cv::Mat result;
    floatImg.convertTo(result, CV_8UC3, 255.0f);
    return result;
}

cv::Mat normalizeFloatImg(const cv::Mat& floatImg)
{
    assert(floatImg.type() == CV_32FC3);
    cv::Mat result;
    cv::normalize(floatImg, result, 0, 1, cv::NORM_MINMAX);
    return result;
}

int main()
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
    filter2D(img, filtered1, CV_32FC3, kernel_scharr_x);
    filter2D(filtered1, filtered1, -1, kernel_scharr_x);
    filtered1 = normalizeFloatImg(filtered1);
    filtered1 = floatImgToUCharImg(filtered1);
    cv::imwrite("lenna_512x512_filtered1.png", filtered1);

    cv::Mat filtered2 = filter2DsNet(img, {kernel_scharr_x, kernel_scharr_x});
    filtered2 = normalizeFloatImg(filtered2);
    filtered2 = floatImgToUCharImg(filtered2);
    cv::imwrite("lenna_512x512_filtered2.png", filtered2);
}


// todo:
// Conv free func
// transposed convolution layer
// Skip conn aka comp graph
// was ist ein softmax-layer nochmal?
// Affine layer- flow layer?

// zweites video dabei, was die differenzframes drin hat
// anfang vom neuronalen netz koennte der codec sein und nur der FC-Layer waere das eigentliche Video
// oder low-bitrate-video so nachverbessern? https://arxiv.org/pdf/1504.06993.pdf
// bias fuer layer, filter oder neurons?

// image compression
// http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html
// http://cs.stanford.edu/people/karpathy/convnetjs/

// statt affine: http://torch.ch/blog/2015/09/07/spatial_transformers.html

// Frameworks:
// http://caffe.berkeleyvision.org
// http://torch.ch
// http://deeplearning.net/software/theano
// https://github.com/Lasagne/Lasagne
// http://keras.io
// https://www.tensorflow.org

// mal bilder von papers oder pharma klassifizieren lassen

// tiny-cnn: mal autoencoder bauen mit deconv-layer und fit statt train