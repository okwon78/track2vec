/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#pragma once

#include <vector>
#include <random>
#include <unordered_map>
#include <set>

#include "matrix.h"
#include "model.h"

namespace track2vec
{

class Loss
{
public:
    Loss(std::shared_ptr<Matrix> &, int64_t);
    void initNegative(std::vector<int64_t> &);
    double forward(int64_t, const std::set<int64_t>&, model::State &, double);
    
private:
    static const int64_t NEGATIVE_TABLE_SIZE = 10000000;
    
    std::shared_ptr<Matrix> output_;
    int64_t neg_;
    std::vector<double> t_sigmoid_;
    std::vector<double> t_log_;
    std::vector<int64_t> negatives_;
    std::uniform_int_distribution<size_t> uniform_;
    
    int64_t getNegative(int64_t, const std::set<int64_t>&, std::minstd_rand&);
    double binaryLogistic(int64_t, model::State &, bool, double);
    double sigmoid(double) const;
    double log(double) const;
};

} // namespace track2vec
