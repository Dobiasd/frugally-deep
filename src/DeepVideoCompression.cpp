#include <opencv2/opencv.hpp>
#include <fplus/fplus.h>

cv::Mat filter2DNestedForLoops(const cv::Mat& img, const cv::Mat kernel)
{
    cv::Mat dst = img.clone();
    for (int y = 1; y < img.rows - 1; ++y)
    {
        for (int x = 1; x < img.cols - 1; ++x)
        {
            float acc = 0.0f;
            for (int yk = 0; yk < kernel.rows; ++yk)
            {
                for (int xk = 0; xk < kernel.cols; ++xk)
                {
                    acc +=
                        img.at<float>(y + yk - 1, x + xk - 1) *
                        kernel.at<float>(yk, xk);
                }
            }
            dst.at<float>(y, x) = acc;
        }
    }
    return dst;
}

std::vector<float> img_to_vec(const cv::Mat& img)
{
    std::vector<float> img_vec;
    img_vec.reserve(static_cast<std::size_t>(img.rows * img.cols));
    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            img_vec.push_back(img.at<float>(y, x));
        }
    }
    return img_vec;
}

cv::Mat vec_to_img(const std::vector<float>& img_vec, const cv::Size& size)
{
    assert(img_vec.size() == static_cast<std::size_t>(size.width * size.height));
    cv::Mat img(size, CV_32FC1);
    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            img.at<float>(y, x) = img_vec[static_cast<std::size_t>(y*img.cols + x)];
        }
    }
    return img;
}

// N-dimensional array.
// Values of the last dimension are stored adjecently in memory.
class tensor
{
public:
    tensor(const std::vector<std::size_t>& dimensions) :
        dimensions_(dimensions),
        dim_idx_factors_(generate_dim_idx_factors_(dimensions_)),
        values_(fplus::product(dimensions_), 0.0f)
    {
    }
    // Values of the last dimension are adjecent in the vector.
    tensor(const std::vector<std::size_t>& dimensions,
        const std::vector<float>& values) :
        dimensions_(dimensions),
        dim_idx_factors_(generate_dim_idx_factors_(dimensions_)),
        values_(values)
    {
        assert(values.size() == fplus::product(dimensions));
    }
    float get(const std::vector<std::size_t>& pos) const
    {
        return values_[idx(pos)];
    }
    void set(const std::vector<std::size_t>& pos, float value)
    {
        values_[idx(pos)] = value;
    }
    std::vector<std::size_t> get_dimensions() const
    {
        return dimensions_;
    }
    // todo
    // tensor get_sub_tensor(std::size_t pos_first_dim)
// todo
//private:
    static std::vector<std::size_t> generate_dim_idx_factors_
        (const std::vector<std::size_t>& dimensions)
    {
        assert(dimensions.size() > 0);
        return
            fplus::tail(
                fplus::scan_right(
                    std::multiplies<std::size_t>(),
                    std::size_t(1),
                    dimensions
                    ));
    }
    std::size_t idx(const std::vector<std::size_t>& pos) const
    {
        assert(pos.size() == dimensions_.size());
        assert(
            fplus::all(
                fplus::zip_with(
                    fplus::is_less<std::size_t>,
                    pos,
                    dimensions_)));
        return
            fplus::sum(
                fplus::zip_with(
                    std::multiplies<std::size_t>(), pos, dim_idx_factors_));
    }
    std::vector<std::size_t> dimensions_;
    std::vector<std::size_t> dim_idx_factors_;
    std::vector<float> values_;
};

std::string show_tensor(const tensor& t)
{
    // todo: recurse through all dimensions-
    std::string result;
    result += "dimensions_";
    result += "\n";
    result += fplus::show_cont(t.dimensions_) + "\n";
    result += "dim_idx_factors_";
    result += "\n";
    result += fplus::show_cont(t.dim_idx_factors_) + "\n";
    result += "values_";
    result += "\n";
    result += fplus::show_cont(t.values_) + "\n";
    return result;
}

cv::Mat filter2DMatMult(const cv::Mat& img, const cv::Mat kernel)
{
    auto img_vec = img_to_vec(img);
    auto kernel_vec = img_to_vec(kernel);
    return vec_to_img(img_vec, img.size());
}

int main()
{
    cv::Mat img = cv::imread("images/lenna_512x512.png", cv::IMREAD_GRAYSCALE);

    img.convertTo(img, CV_32FC1);

    cv::Mat kernel(cv::Size(3,3), CV_32FC1, cv::Scalar(0));
    kernel.at<float>(cv::Point(0,0)) = 1.0f / 45.0f;
    kernel.at<float>(cv::Point(1,0)) = 2.0f / 45.0f;
    kernel.at<float>(cv::Point(2,0)) = 3.0f / 45.0f;
    kernel.at<float>(cv::Point(0,1)) = 4.0f / 45.0f;
    kernel.at<float>(cv::Point(1,1)) = 5.0f / 45.0f;
    kernel.at<float>(cv::Point(2,1)) = -6.0f / 45.0f;
    kernel.at<float>(cv::Point(0,2)) = -7.0f / 45.0f;
    kernel.at<float>(cv::Point(1,2)) = -8.0f / 45.0f;
    kernel.at<float>(cv::Point(2,2)) = -9.0f / 45.0f;

    cv::Mat filtered1;
    filter2D(img, filtered1, -1, kernel);
    cv::imwrite("lenna_512x512_filtered1.png", filtered1);

    cv::Mat filtered2 = filter2DNestedForLoops(img, kernel);
    cv::imwrite("lenna_512x512_filtered2.png", filtered2);

    cv::Mat filtered3 = filter2DMatMult(img, kernel);
    cv::imwrite("lenna_512x512_filtered3.png", filtered3);

    tensor t1({4,7,12,3});
    t1.set({3,1,9,2}, 42.0f);
    std::cout << t1.get({3,1,9,2}) << std::endl;

    tensor t2({3,3,3});
    t2.set({0,0,0}, 1.0f);
    t2.set({1,1,1}, 5.0f);
    std::cout << show_tensor(t2) << std::endl;
}