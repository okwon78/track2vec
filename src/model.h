/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#pragma once

#include <memory>
#include <random>
#include <set>

#include "vector.h"

namespace track2vec
{
namespace model
{

class State
{
private:
    double lossValue_;
    int64_t nexamples_;
    
public:
    Vector hidden;
    Vector output;
    Vector grad;
    std::minstd_rand rng;
    
    State(int64_t hiddenSize, int64_t outputSize, int64_t seed);
    double getLoss();
    void incrementNExamples(double loss);
};

} // namespace model

class Matrix;
class Loss;

class Model
{
private:
    std::shared_ptr<Matrix> input_;
    std::shared_ptr<Matrix> output_;
    std::shared_ptr<Loss> loss_;
    
public:
    Model(std::shared_ptr<Matrix>, std::shared_ptr<Matrix>, std::shared_ptr<Loss>);
    void update(int64_t,
                const std::vector<int64_t> &,
                const std::vector<int64_t> &,
                int64_t,
                const std::set<int64_t>&,
                double,
                model::State&);
    
    void computeHidden(int64_t,
                       const std::vector<int64_t>&,
                       const std::vector<int64_t>&,
                       model::State&) const;
    
    void backprop(int64_t,
                  const std::vector<int64_t>&,
                  const std::vector<int64_t> &,
                  const Vector&);
};

} // namespace track2vec
