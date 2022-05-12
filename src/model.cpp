/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#include "model.h"
#include "loss.h"

namespace track2vec
{
namespace model
{

State::State(int64_t hiddenSize, int64_t outputSize, int64_t seed)
: lossValue_(0.0), nexamples_(0), hidden(hiddenSize), output(outputSize), grad(hiddenSize), rng(seed) {}

double State::getLoss()
{
    double loss = lossValue_ / nexamples_;
    
    lossValue_ = 0.0;
    nexamples_ = 0;
    
    return loss;
}

void State::incrementNExamples(double loss)
{
    lossValue_ += loss;
    nexamples_++;
}

} // namespace model
Model::Model(std::shared_ptr<Matrix> input, std::shared_ptr<Matrix> output, std::shared_ptr<Loss> loss)
: input_(input), output_(output), loss_(loss) {}

void Model::computeHidden(int64_t input_idx,
                          const std::vector<int64_t> &artist_indices,
                          const std::vector<int64_t> &genre_indices,
                          model::State &state) const
{
    
    Vector &hidden = state.hidden;
    hidden.zero();
    
    // track embedding
    hidden.addRow(*input_, input_idx);
    
    // artist embedding
    for (const auto &artist_idx : artist_indices)
    {
        hidden.addRow(*input_, artist_idx);
    }
    
    // reco genre embedding
    for (const auto &genre_idx : genre_indices)
    {
        hidden.addRow(*input_, genre_idx);
    }
    
    int64_t total = 1 + artist_indices.size() + genre_indices.size();
    hidden.mul(1.0 / total);
}

void Model::update(int64_t input_idx,
                   const std::vector<int64_t> &artist_indices,
                   const std::vector<int64_t> &genre_indices,
                   int64_t output_idx,
                   const std::set<int64_t>& outputs,
                   double lr,
                   model::State &state)
{
    computeHidden(input_idx, artist_indices, genre_indices, state);
    Vector &grad = state.grad;
    grad.zero();
    
    double lossValue = loss_->forward(output_idx, outputs, state, lr);
    state.incrementNExamples(lossValue);
    
    backprop(input_idx, artist_indices, genre_indices, grad);
}

void Model::backprop(int64_t track_idx,
                     const std::vector<int64_t> &artist_indices,
                     const std::vector<int64_t> &genre_indices,
                     const Vector &grad)
{
    
    input_->addVectorToRow(grad, track_idx);
    
    // update artist embedding
    for (auto artist_idx : artist_indices)
    {
        input_->addVectorToRow(grad, artist_idx);
    }
    // update genre embedding
    for (auto genre_idx : genre_indices)
    {
        input_->addVectorToRow(grad, genre_idx);
    }
}

} // namespace track2vec
