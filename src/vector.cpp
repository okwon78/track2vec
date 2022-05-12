/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#include "vector.h"
#include "matrix.h"

#include <iomanip>
#include <cmath>

namespace track2vec
{

Vector::Vector(int64_t m) : data_(m) {}
Vector::Vector(std::vector<double> &vec)
{
    data_ = std::move(vec);
}

void Vector::addRow(const Matrix &A, int64_t i)
{
    assert(i >= 0);
    assert(i < A.size(0));
    assert(size() == A.size(1));
    A.addRowToVector(*this, i);
}

void Vector::addRow(const Matrix &A, int64_t i, double a)
{
    assert(i >= 0);
    assert(i < A.size(0));
    assert(size() == A.size(1));
    A.addRowToVector(*this, i, a);
}

void Vector::zero()
{
    std::fill(data_.begin(), data_.end(), 0.0);
}

void Vector::mul(double a)
{
    for (int64_t i = 0; i < size(); i++)
    {
        data_[i] *= a;
    }
}

Vector Vector::avg(const Vector &ref)
{
    size_t dim = data_.size();
    Vector vec(dim);
    
    for (int64_t i = 0; i < dim; i++)
    {
        vec[i] = (data_[i] + ref[i]) / 2;
    }
    
    return vec;
}

double Vector::norm() const
{
    double sum = 0;
    for (int64_t i = 0; i < size(); i++)
    {
        sum += data_[i] * data_[i];
    }
    return std::sqrt(sum);
}

std::ostream &operator<<(std::ostream &os, const Vector &v)
{
    os << std::setprecision(5);
    for (int64_t j = 0; j < v.size(); j++)
    {
        os << v[j] << ' ';
    }
    return os;
}

} // namespace track2vec
