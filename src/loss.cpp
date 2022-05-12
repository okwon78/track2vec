/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#include "loss.h"
#include "matrix.h"

namespace track2vec
{

constexpr int64_t SIGMOID_TABLE_SIZE = 512;
constexpr int64_t MAX_SIGMOID = 8;
constexpr int64_t LOG_TABLE_SIZE = 512;

Loss::Loss(std::shared_ptr<Matrix> &output, int64_t neg): output_(output), neg_(neg), uniform_()
{
    t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
    for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++)
    {
        double x = double(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
        t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
    }
    
    t_log_.reserve(LOG_TABLE_SIZE + 1);
    for (int i = 0; i < LOG_TABLE_SIZE + 1; i++)
    {
        double x = (double(i) + 1e-5) / LOG_TABLE_SIZE;
        t_log_.push_back(std::log(x));
    }
}

void Loss::initNegative(std::vector<int64_t> &trackCounts)
{
    double z = 0.0;
    for (size_t i = 0; i < trackCounts.size(); i++)
    {
        z += pow(trackCounts[i], 0.5);
    }
    
    for (size_t i = 0; i < trackCounts.size(); i++)
    {
        double c = pow(trackCounts[i], 0.5);
        for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++)
        {
            negatives_.push_back(i);
        }
    }
    
    uniform_ = std::uniform_int_distribution<size_t>(0, negatives_.size() - 1);
}

double Loss::log(double x) const
{
    if (x > 1.0)
    {
        return 0.0;
    }
    int64_t i = int64_t(x * LOG_TABLE_SIZE);
    return t_log_[i];
}

double Loss::binaryLogistic(int64_t outputIdx, model::State &state, bool labelIsPositive, double lr)
{
    double score = sigmoid(output_->dotRow(state.hidden, outputIdx));
    double alpha = lr * (double(labelIsPositive) - score);
    
    state.grad.addRow(*output_, outputIdx, alpha);
    output_->addVectorToRow(state.hidden, outputIdx, alpha);
    
    return labelIsPositive ? -log(score) : -log(1.0 - score);
}

double Loss::sigmoid(double x) const
{
    if (x < -MAX_SIGMOID)
    {
        return 0.0;
    }
    else if (x > MAX_SIGMOID)
    {
        return 1.0;
    }
    else
    {
        int64_t i = (x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2;
        return t_sigmoid_[i];
    }
}

double Loss::forward(int64_t output_idx, const std::set<int64_t>& outputs, model::State &state, double lr)
{
    assert(output_idx >= 0);
    
    double loss = binaryLogistic(output_idx, state, true, lr);
    
    for (int32_t n = 0; n < neg_; n++)
    {
        int64_t negative_idx = getNegative(output_idx, outputs, state.rng);
        loss += binaryLogistic(negative_idx, state, false, lr);
    }
    
    return loss;
}

int64_t Loss::getNegative(int64_t outputIdx, const std::set<int64_t>& outputs, std::minstd_rand &rng)
{
    int32_t negative = outputIdx;
    
    while (0 < outputs.count(negative))
    {
        negative = negatives_[uniform_(rng)];
    }
    
    return negative;
}

} // namespace track2vec
