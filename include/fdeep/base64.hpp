// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace fdeep { namespace internal
{

// Make sure data outlives instances of this this fascade class.
class json_data_strs_char_prodiver {
public:
    json_data_strs_char_prodiver(const nlohmann::json& data,
        std::string::value_type pad_right_char) :
        data_(data),
        it_data_(std::begin(data_)),
        current_str_(data_to_str(*it_data_)),
        it_str_(std::begin(current_str_)),
        pad_right_char_(pad_right_char)
    {
    }
    static std::string data_to_str(const nlohmann::json& dat)
    {
        std::string result = dat;
        return result;
    }
    std::size_t size() const
    {
        std::size_t sum = 0;
        for (const auto& dat : data_)
        {
            sum += data_to_str(dat).size();
        }
        return sum;
    }
    std::string::value_type next()
    {
        if (it_data_ == std::end(data_))
        {
            return pad_right_char_;
        }
        if (it_str_ == std::end(current_str_))
        {
            ++it_data_;
            current_str_ = data_to_str(*it_data_);
            it_str_ = std::begin(current_str_);
        }
        return *(it_str_++);
    }
private:
    const nlohmann::json& data_;
    nlohmann::json::const_iterator it_data_;
    std::string current_str_;
    std::string::const_iterator it_str_;
    std::string::value_type pad_right_char_;
};

// source: https://stackoverflow.com/a/31322410/1866775
static const std::uint8_t from_base64[] = { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  62, 255,  62, 255,  63,
     52,  53,  54,  55,  56,  57,  58,  59,  60,  61, 255, 255, 255, 255, 255, 255,
    255,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
     15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 255, 255, 255, 255,  63,
    255,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
     41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51, 255, 255, 255, 255, 255};
static const char to_base64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";
inline std::vector<std::uint8_t> Base64_decode(
    json_data_strs_char_prodiver&& encoded_string)
{
    // Make sure string length is a multiple of 4
    auto encoded_size = encoded_string.size();
    while ((encoded_string.size() % 4) != 0)
    {
        ++encoded_size;
    }
    std::vector<std::uint8_t> ret;
    ret.reserve(3 * encoded_size / 4);
    for (size_t i = 0; i < encoded_size; i += 4)
    {
        // Get values for each group of four base 64 characters
        std::uint8_t b4[4];
        const auto c0 = encoded_string.next();
        const auto c1 = encoded_string.next();
        const auto c2 = encoded_string.next();
        const auto c3 = encoded_string.next();
        b4[0] = (c0 <= 'z') ? from_base64[static_cast<std::size_t>(c0)] : 0xff;
        b4[1] = (c1 <= 'z') ? from_base64[static_cast<std::size_t>(c1)] : 0xff;
        b4[2] = (c2 <= 'z') ? from_base64[static_cast<std::size_t>(c2)] : 0xff;
        b4[3] = (c3 <= 'z') ? from_base64[static_cast<std::size_t>(c3)] : 0xff;
        // Transform into a group of three bytes
        std::uint8_t b3[3];
        b3[0] = static_cast<std::uint8_t>(((b4[0] & 0x3f) << 2) + ((b4[1] & 0x30) >> 4));
        b3[1] = static_cast<std::uint8_t>(((b4[1] & 0x0f) << 4) + ((b4[2] & 0x3c) >> 2));
        b3[2] = static_cast<std::uint8_t>(((b4[2] & 0x03) << 6) + ((b4[3] & 0x3f) >> 0));
        // Add the byte to the return value if it isn't part of an '=' character (indicated by 0xff)
        if (b4[1] != 0xff) ret.push_back(b3[0]);
        if (b4[2] != 0xff) ret.push_back(b3[1]);
        if (b4[3] != 0xff) ret.push_back(b3[2]);
    }
    return ret;
}

} } // namespace fdeep, namespace internal
