/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#pragma once

#include <iostream>
#include <cstdint>
#include <vector>

namespace track2vec
{

class Matrix;

class Vector
{
private:
    std::vector<double> data_;
    
public:
    explicit Vector(int64_t);
    Vector(std::vector<double> &);
    Vector(const Vector &) = default;
    Vector(Vector &&) noexcept = default;
    Vector &operator=(const Vector &) = default;
    Vector &operator=(Vector &&) = default;
    
    void addRow(const Matrix &, int64_t);
    void addRow(const Matrix &, int64_t, double);
    
    inline double &operator[](int64_t i)
    {
        return data_[i];
    }
    inline const double &operator[](int64_t i) const
    {
        return data_[i];
    }
    
    inline int64_t size() const
    {
        return data_.size();
    }
    
    inline const std::vector<double> &data() const
    {
        return data_;
    }
    
    void zero();
    void mul(double);
    Vector avg(const Vector &);
    double norm() const;
};

std::ostream &operator<<(std::ostream &, const Vector &);

} // namespace track2vec
