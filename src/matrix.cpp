/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#include "matrix.h"

#include <thread>
#include <random>
#include "vector.h"

namespace track2vec
{

Matrix::Matrix(int64_t m, int64_t n)
: m_(m), n_(n), data_(m * n) {}

int64_t Matrix::size(int64_t dim) const
{
    assert(dim == 0 || dim == 1);
    return dim == 0 ? m_ : n_;
}

void Matrix::zero()
{
    std::fill(data_.begin(), data_.end(), 0.0);
}

double &Matrix::at(int64_t i, int64_t j)
{
    return data_[i * n_ + j];
}

void Matrix::randomInit(int64_t seed)
{
    std::minstd_rand rng(seed);
    std::uniform_real_distribution<> uniform(-1, 1);
    for (int64_t i = 0; i < (m_ * n_); i++)
    {
        data_[i] = uniform(rng);
    }
}

void Matrix::addVectorToRow(const Vector &vec, int64_t i)
{
    assert(i >= 0);
    assert(i < m_);
    assert(vec.size() == n_);
    for (int64_t j = 0; j < n_; j++)
    {
        data_[i * n_ + j] += vec[j];
    }
}

void Matrix::addVectorToRow(const Vector &vec, int64_t i, double a)
{
    assert(i >= 0);
    assert(i < m_);
    assert(vec.size() == n_);
    for (int64_t j = 0; j < n_; j++)
    {
        data_[i * n_ + j] += a * vec[j];
    }
}

void Matrix::addRowToVector(Vector &x, int64_t i) const
{
    assert(i >= 0);
    assert(i < this->size(0));
    assert(x.size() == this->size(1));
    for (int64_t j = 0; j < n_; j++)
    {
        x[j] += at(i, j);
    }
}

void Matrix::addRowToVector(Vector &x, int64_t i, double a) const
{
    assert(i >= 0);
    assert(i < this->size(0));
    assert(x.size() == this->size(1));
    for (int64_t j = 0; j < n_; j++)
    {
        x[j] += a * at(i, j);
    }
}

double Matrix::dotRow(const Vector &vec, int64_t i) const
{
    double d = 0.0;
    
    for (int64_t j = 0; j < n_; j++)
    {
        d += at(i, j) * vec[j];
    }
    if (std::isnan(d))
    {
        std::cerr << "EncounteredNaNError: " << vec << std::endl;
        throw EncounteredNaNError();
    }
    return d;
}

} // namespace track2vec
