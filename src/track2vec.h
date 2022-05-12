/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#pragma once

#include <time.h>
#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>

#include "args.h"
#include "dictionary.h"
#include "matrix.h"
#include "model.h"
#include "vector.h"

namespace track2vec
{

class Track2Vec
{
private:
    //Custom Obj Instance
    std::shared_ptr<Args> args_;
    std::shared_ptr<Matrix> input_;
    std::shared_ptr<Matrix> output_;
    std::shared_ptr<Dictionary> dict_;
    std::shared_ptr<Model> model_;
    
    //Variable
    std::atomic<int64_t> processedTotalTokenCount_{};
    std::atomic<double> log_loss_{};
    std::chrono::steady_clock::time_point start_;
    
    //Data
    std::unordered_map<int64_t, std::vector<std::string>> data_;
    
    // output file path
    static const std::string model_output_track;
    static const std::string model_input_track;
    static const std::string model_input_artist;
    static const std::string model_input_genre;
    
    static const std::string track_vec;
    static const std::string artist_vec;
    static const std::string genre_vec;
    
    //Misc
    std::exception_ptr trainException_;
    
public:
    using TrainCallback = std::function<void(int64_t)>;
    using LogCallback = std::function<void(double, double, double, double, int64_t)>;
    
    Track2Vec(std::shared_ptr<Args> args);
    void loadData();
    void train(const LogCallback &callback = {});
    void saveModel(const std::string &);
    void saveVectors(const std::string &);
    
private:
    void saveOutputMatrix(const std::string &);
    void saveTrackEmbeddingVectors(const std::string &);
    void saveTrackInputVectors(const std::string &);
    void saveArtistInputVectors(const std::string &);
    void saveGenreInputVectors(const std::string &);
    
    void loadTrackInputVectors(const std::string &);
    void loadArtistInputVectors(const std::string &);
    void loadGenreInputVectors(const std::string &);
    void loadOutputMatrix(const std::string &);
    
    std::shared_ptr<Matrix> createRandomMatrix() const;
    std::shared_ptr<Matrix> createTrainOutputMatrix() const;
    void setInputMatrixFromFile(const std::string &);
    void setOutputMatrixFromFile(const std::string &);
    
    void startThreads(const LogCallback &);
    void trainThread(int64_t);
    void trainThreadInMemory(int64_t);
    bool keepTraining(const int64_t) const;
    void printInfo(double, double, const LogCallback & = {});
    std::tuple<int64_t, double, double> progressInfo(double);
    
    void skipgram(model::State &, double, const std::vector<std::string> &);
    void getTrackEmbeddingVector(Vector&, int64_t,
                                 const std::vector<int64_t>&,
                                 const std::vector<int64_t>&) const;
    
    inline void getOutputVector(Vector& vec, int64_t idx) const
    {
        vec.addRow(*output_, idx);
    }
    inline void getInputVector(Vector& vec, int64_t idx) const
    {
        vec.addRow(*input_, idx);
    }
};

} // namespace track2vec
