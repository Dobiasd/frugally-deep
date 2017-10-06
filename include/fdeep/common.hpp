// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

namespace fdeep { namespace internal
{

inline std::runtime_error error(const std::string& error)
{
    return std::runtime_error(error);
}

inline void raise_error(const std::string& msg)
{
    throw error(msg);
}

inline void assertion(bool cond, const std::string& error)
{
    if (!cond)
    {
        raise_error(error);
    }
}

typedef float float_t;

// shares unerlying data on copy and assignment
class shared_float_vec
{
public:
    typedef std::vector<float_t> vec_t;
    typedef std::shared_ptr<vec_t> vec_ptr_t;
    typedef vec_t::iterator iterator;
    typedef vec_t::const_iterator const_iterator;
    typedef vec_t::value_type value_type;

    shared_float_vec(std::size_t size, float_t initial) :
        data_(std::make_shared<std::vector<float_t>>(size, initial))
    {
    }

    shared_float_vec(const shared_float_vec&) = default;
    shared_float_vec& operator = (const shared_float_vec&) = default;

    std::size_t size() const
    {
        return data_->size();
    }
    bool empty() const
    {
        return data_->empty();
    }
    iterator begin()
    {
        return data_->begin();
    }
    iterator end()
    {
        return data_->end();
    }
    const_iterator begin() const
    {
        return data_->begin();
    }
    const_iterator end() const
    {
        return data_->end();
    }
    void push_back(float_t x)
    {
        data_->push_back(x);
    }
    void reserve(std::size_t capacity)
    {
        data_->reserve(capacity);
    }
    float_t& operator[](std::size_t index)
    {
        return (*data_)[index];
    }
    const float_t& operator[](std::size_t index) const
    {
        return (*data_)[index];
    }
private:
    vec_ptr_t data_;
};

struct shared_float_vec_back_insert_iterator : public std::back_insert_iterator<shared_float_vec>
{
    typedef std::back_insert_iterator<shared_float_vec> base_type;
    explicit shared_float_vec_back_insert_iterator(shared_float_vec& v) :
        base_type(v), v_ptr_(&v), pos_(0) {}
    shared_float_vec_back_insert_iterator(const shared_float_vec_back_insert_iterator& other) :
        base_type(*other.v_ptr_), v_ptr_(other.v_ptr_), pos_(other.pos_) {}
    shared_float_vec_back_insert_iterator& operator=(const shared_float_vec_back_insert_iterator& other)
    {
        v_ptr_ = other.v_ptr_;
        pos_ = other.pos_;
        return *this;
    }
    shared_float_vec_back_insert_iterator& operator=(const float_t& x)
    {
        (*v_ptr_)[pos_] = x;
        ++pos_;
        return *this;
    }
    shared_float_vec_back_insert_iterator& operator=(float_t&& x)
    {
        (*v_ptr_)[pos_] = std::move(x);
        ++pos_;
        return *this;
    }
    shared_float_vec_back_insert_iterator& operator*() { return *this; }
    shared_float_vec_back_insert_iterator& operator++() { return *this; }
    shared_float_vec_back_insert_iterator operator++(int) { return *this; }
private:
    shared_float_vec* v_ptr_;
    std::size_t pos_;
};

//typedef shared_float_vec float_vec;
typedef std::vector<float_t> float_vec;

} } // namespace fdeep, namespace internal
