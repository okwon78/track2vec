/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#pragma once

#include <stdexcept>
#include <iostream>
#include <cassert>
#include <cstdint>
#include <vector>

namespace track2vec
{

class Vector;

class Matrix
{
private:
    int64_t m_;
    int64_t n_;
    std::vector<double> data_;
    
public:
    explicit Matrix(int64_t, int64_t);
    int64_t size(int64_t dim) const;
    void zero();
    double &at(int64_t i, int64_t j);
    
    void addVectorToRow(const Vector &, int64_t, double);
    void addVectorToRow(const Vector &, int64_t);
    
    void addRowToVector(Vector&, int64_t) const;
    void addRowToVector(Vector&, int64_t, double) const;
    
    void randomInit(int64_t);
    
    double dotRow(const Vector&, int64_t) const;
    
    inline const double &at(int64_t i, int64_t j) const
    {
        assert(i * n_ + j < data_.size());
        return data_[i * n_ + j];
    };
    
    inline int64_t rows() const
    {
        return m_;
    }
    inline int64_t cols() const
    {
        return n_;
    }
    
    class EncounteredNaNError : public std::runtime_error
    {
    public:
        EncounteredNaNError() : std::runtime_error("Encountered NaN.") {}
    };
};

} // namespace track2vec
