/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#include "track2vec.h"

#include <iomanip>
#include <fstream>
#include <thread>
#include <iostream>
#include <set>
#include <nlohmann/json.hpp>

#include "model.h"
#include "utils.h"
#include "loss.h"

namespace track2vec
{

using json = nlohmann::json;

const std::string Track2Vec::model_output_track = "model_output_track.json";
const std::string Track2Vec::model_input_track = "model_input_track.json";
const std::string Track2Vec::model_input_artist = "model_input_artist.json";
const std::string Track2Vec::model_input_genre = "model_input_genre.json";

const std::string Track2Vec::track_vec = "track_vec.json";
const std::string Track2Vec::artist_vec = "artist_vec.json";
const std::string Track2Vec::genre_vec = "genre_vec.json";

Track2Vec::Track2Vec(std::shared_ptr<Args> args) :
args_(args),
processedTotalTokenCount_(0),
log_loss_(-1),
trainException_(nullptr) {}

void Track2Vec::train(const LogCallback &callback)
{
    dict_ = std::make_shared<Dictionary>(args_);
    dict_->loadMeta(args_->metaFileName, args_->input);
    
    input_ = createRandomMatrix();
    output_ = createTrainOutputMatrix();
    
    if (args_->loadPretrained > 0) {
        setInputMatrixFromFile(args_->outputDir);
        setOutputMatrixFromFile(args_->outputDir);
    }
    
    auto loss = std::make_shared<Loss>(output_, args_->neg);
    auto track_cnt = dict_->getTrackCount();
    loss->initNegative(track_cnt);
    model_ = std::make_shared<Model>(input_, output_, loss);
    
    if (args_->memory > 0)
    {
        loadData();
    }
    
    startThreads(callback);
}

void Track2Vec::saveModel(const std::string &outputDir)
{
    if (!input_ || !output_)
    {
        throw std::runtime_error("Model never trained");
    }
    
    std::string output_filename = outputDir + "/" + model_output_track;
    saveOutputMatrix(output_filename);
    
    std::string track_filename = outputDir + "/" + model_input_track;
    saveTrackInputVectors(track_filename);
    
    std::string artist_filename = outputDir + "/" + model_input_artist;
    saveArtistInputVectors(artist_filename);
    
    std::string genre_filename = outputDir + "/" + model_input_genre;
    saveGenreInputVectors(genre_filename);
}

void Track2Vec::saveOutputMatrix(const std::string &filename)
{
    if (!output_)
    {
        throw std::runtime_error("Model never trained");
    }
    
    std::ofstream ofs(filename, std::ofstream::binary);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(filename + " cannot be opened for saving vectors!");
    }
    
    const std::unordered_map<std::string, trackEntry> &tracks = dict_->getTrackEntries();
    
    json j;
    Vector vec(args_->dim);
    for (const auto &pair : tracks)
    {
        const trackEntry &entry = pair.second;
        getOutputVector(vec, entry.idx);
        
        j.clear();
        j["track_id"] = entry.track_id;
        j["vector"] = vec.data();
        ofs << j << std::endl;
    }
    
    ofs.close();
}

void Track2Vec::saveTrackInputVectors(const std::string &filename)
{
    if (!input_)
    {
        throw std::runtime_error("Model never trained");
    }
    
    std::ofstream ofs(filename, std::ofstream::binary);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(filename + " cannot be opened for saving vectors!");
    }
    
    const std::unordered_map<std::string, trackEntry> &tracks = dict_->getTrackEntries();
    
    json j;
    Vector vec(args_->dim);
    for (const auto &pair : tracks)
    {
        const trackEntry &entry = pair.second;
        getInputVector(vec, entry.idx);
        
        j.clear();
        j["track_id"] = entry.track_id;
        j["vector"] = vec.data();
        ofs << j << std::endl;
    }
    
    ofs.close();
}

void Track2Vec::saveVectors(const std::string &outputDir)
{
    saveTrackEmbeddingVectors(outputDir + "/" + track_vec);
    saveArtistInputVectors(outputDir + "/" + artist_vec);
    saveGenreInputVectors(outputDir + "/" + genre_vec);
}

void Track2Vec::saveTrackEmbeddingVectors(const std::string &filename)
{
    if (!input_ || !output_)
    {
        throw std::runtime_error("Model never trained");
    }
    
    std::ofstream ofs(filename, std::ofstream::binary);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(filename + " cannot be opened for saving vectors!");
    }
    
    Vector vec(args_->dim);
    
    const std::unordered_map<std::string, trackEntry> &tracks = dict_->getTrackEntries();
    
    json j;
    for (const auto &pair : tracks)
    {
        const trackEntry &entry = pair.second;
        const std::string &track_id = entry.track_id;
        
        getTrackEmbeddingVector(vec, entry.idx, entry.artist_matrix_indices, entry.genre_matrix_indices);
        
        j.clear();
        j["track_id"] = track_id;
        j["vector"] = vec.data();
        ofs << j << std::endl;
    }
    
    ofs.close();
}
void Track2Vec::getTrackEmbeddingVector(Vector &vec,
                                        int64_t trackIdx,
                                        const std::vector<int64_t> &artistInices,
                                        const std::vector<int64_t> &genreInices) const
{
    Vector in(args_->dim);
    getInputVector(in, trackIdx);
    
    for (int64_t artistIdx : artistInices)
    {
        getInputVector(in, artistIdx);
    }
    
    for (int64_t genreIdx : genreInices)
    {
        getInputVector(in, genreIdx);
    }
    
    size_t z = 1 + artistInices.size() + genreInices.size();
    in.mul(1.0 / z);
    
    Vector out(args_->dim);
    getOutputVector(out, trackIdx);
    
    vec = in.avg(out);
}

void Track2Vec::saveArtistInputVectors(const std::string &filename)
{
    if (!input_)
    {
        throw std::runtime_error("Model never trained");
    }
    
    std::ofstream ofs(filename, std::ofstream::binary);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(filename + " cannot be opened for saving vectors!");
    }
    
    Vector vec(args_->dim);
    const std::unordered_map<std::string, artistEntry> &artists = dict_->getArtistEntries();
    
    json j;
    for (const auto &pair : artists)
    {
        const artistEntry &entry = pair.second;
        getInputVector(vec, entry.idx);
        
        j.clear();
        j["artist_id"] = entry.artist_id;
        j["vector"] = vec.data();
        ofs << j << std::endl;
    }
    
    ofs.close();
}

void Track2Vec::saveGenreInputVectors(const std::string &filename)
{
    if (!input_)
    {
        throw std::runtime_error("Model never trained");
    }
    
    std::ofstream ofs(filename, std::ofstream::binary);
    if (!ofs.is_open())
    {
        throw std::invalid_argument(filename + " cannot be opened for saving vectors!");
    }
    
    Vector vec(args_->dim);
    const std::unordered_map<std::string, genreEntry> &genres = dict_->getGenreEntries();
    
    json j;
    for (const auto &pair : genres)
    {
        const genreEntry &entry = pair.second;
        getInputVector(vec, entry.idx);
        j.clear();
        j["genre_id"] = entry.genre_id;
        j["vector"] = vec.data();
        ofs << j << std::endl;
    }
    
    ofs.close();
}

void Track2Vec::startThreads(const LogCallback &callback)
{
    start_ = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    const int64_t ntokens = dict_->ntokens();
    
    for (int64_t i = 0; i < args_->thread; i++)
    {
        if (args_->memory > 0)
        {
            threads.push_back(std::thread([=]() { trainThreadInMemory(i); }));
        }
        else
        {
            threads.push_back(std::thread([=]() { trainThread(i); }));
        }
    }
    
    if (args_->verbose > 0) {
        std::cerr << "Number of thread: " << args_->thread << std::endl;
    }
    
    while (keepTraining(ntokens))
    {
        std::this_thread::sleep_for(std::chrono::seconds(args_->printInterval));
        
        if (log_loss_ >= 0)
        {
            double progress = double(processedTotalTokenCount_) / (args_->epoch * ntokens);
            printInfo(progress, log_loss_, callback);
        }
    }
    
    for (int64_t i = 0; i < threads.size(); i++)
    {
        threads[i].join();
    }
    if (trainException_)
    {
        std::exception_ptr exception = trainException_;
        trainException_ = nullptr;
        std::rethrow_exception(exception);
    }
    
    printInfo(1.0, log_loss_, callback);
}

void Track2Vec::skipgram(model::State &state, double lr, const std::vector<std::string> &sequence)
{
    std::uniform_int_distribution<> uniform(1, args_->ws);
    
    for (int64_t idx = 0; idx < sequence.size(); idx++)
    {
        const std::string &track_id = sequence[idx];
        int64_t input_idx = dict_->getTrackIdx(track_id);
        
        if (input_idx < 0)
            continue;
            
        const trackEntry &entry = dict_->getTrackEntry(track_id);
        double lr_alpha = entry.lr_alpha * lr;
        
        const std::vector<int64_t> &artist_indices = dict_->getArtistMatrixIndices(track_id);
        const std::vector<int64_t> &genre_indices = dict_->getGenreMatrixIndices(track_id);
        
        int64_t boundary = uniform(state.rng);
        std::set<int64_t> output_set;
        
        for (int64_t c = -boundary; c <= boundary; c++)
        {
            if (c != 0 && idx + c >= 0 && idx + c < sequence.size())
            {
                const std::string& target = sequence[idx + c];
                const int64_t output_idx = dict_->getTrackIdx(target);
                output_set.insert(output_idx);
            }
        }
        
        for (int64_t output_idx: output_set) {
            model_->update(input_idx, artist_indices, genre_indices, output_idx, output_set, lr_alpha, state);
        }
        
    }
}

void Track2Vec::trainThread(int64_t threadId) 
{
    std::ifstream ifs(args_->input);
    if (!ifs.is_open())
    {
        throw std::invalid_argument(args_->input + " cannot be opened for loading data!");
    }
    
    utils::gotoLine(ifs, threadId * args_->threadInterval);
    
    if (args_->verbose > 2)
    {
        std::cerr << ">> trainThread [" << threadId << "] started from poistion [";
        std::cerr << threadId * args_->threadInterval << "]" << std::endl;
    }
    
    model::State state(args_->dim, output_->size(0), threadId + args_->seed);
    
    const int64_t ntokens = dict_->ntokens();
    
    int64_t localTokenCount = 0;
    std::vector<std::string> sequence;
    double lr = args_->lr;
    
    try
    {
        while (keepTraining(ntokens))
        {
            localTokenCount += dict_->getSequence(ifs, sequence, state.rng);
            skipgram(state, lr, sequence);
            
            if (localTokenCount > args_->lrUpdateRate)
            {
                processedTotalTokenCount_ += localTokenCount;
                localTokenCount = 0;
                if (threadId == 0)
                {
                    log_loss_ = state.getLoss();
                }
                
                double progress = double(processedTotalTokenCount_) / (args_->epoch * ntokens);
                double lr = args_->lr * (1.0 - progress);
                lr = lr < 0.001 ? 0.001 : lr;
            }
        }
    }
    catch (Matrix::EncounteredNaNError &)
    {
        trainException_ = std::current_exception();
    }
    
    if (threadId == 0)
        log_loss_ = state.getLoss();
    
    ifs.close();
}

void Track2Vec::trainThreadInMemory(int64_t threadId)
{
    int64_t idx = threadId * data_.size() / args_->thread;
    
    if (args_->verbose > 1)
    {
        std::cerr << ">> trainThreadInMemory [" << threadId << "] started from poistion [";
        std::cerr << threadId * args_->threadInterval << "]" << std::endl;
    }
    
    std::uniform_real_distribution<> uniform(0, 1);
    model::State state(args_->dim, output_->size(0), threadId + args_->seed);
    const int64_t ntokens = dict_->ntokens();
    int64_t localTokenCount = 0;
    double lr = args_->lr;
    
    try
    {
        while (keepTraining(ntokens))
        {
            if (0 == data_.count(idx))
                idx = 0;
            
            const std::vector<std::string> &tracks = data_[idx++];
            std::vector<std::string> sequence;
            localTokenCount += tracks.size();
            
            for (const auto &track : tracks)
            {
                if (false == dict_->discard(track, uniform(state.rng)))
                    sequence.push_back(track);
            }
            
            skipgram(state, lr, sequence);
            
            if (localTokenCount > args_->lrUpdateRate)
            {
                processedTotalTokenCount_ += localTokenCount;
                localTokenCount = 0;
                
                if (threadId == 0)
                    log_loss_ = state.getLoss();
                
                double progress = double(processedTotalTokenCount_) / (args_->epoch * ntokens);
                double lr = args_->lr * (1.0 - progress);
                lr = lr < 0.001 ? 0.001 : lr;
            }
        }
    }
    catch (Matrix::EncounteredNaNError &)
    {
        trainException_ = std::current_exception();
    }
    
    if (threadId == 0)
        log_loss_ = state.getLoss();
}

bool Track2Vec::keepTraining(const int64_t ntokens) const
{
    return processedTotalTokenCount_ < args_->epoch * ntokens && !trainException_;
}

void Track2Vec::printInfo(double progress, double loss, const LogCallback &callback)
{
    double ratio;
    double lr;
    int64_t eta;
    std::tie<double, double, int64_t>(ratio, lr, eta) = progressInfo(progress);
    callback(progress, loss, ratio, lr, eta);
}

std::tuple<int64_t, double, double> Track2Vec::progressInfo(double progress)
{
    int64_t t = int64_t(utils::getDuration(start_, std::chrono::steady_clock::now()));
    double lr = args_->lr * (1.0 - progress);
    double process_ratio = 0;
    
    int64_t eta = 2592000; // Default to one month in seconds (720 * 3600)
    
    if (progress > 0 && t >= 0)
    {
        eta = t * (1 - progress) / progress;
        process_ratio = double(processedTotalTokenCount_) / t / args_->thread;
    }
    
    return std::tuple<double, double, int64_t>(process_ratio, lr, eta);
}

std::shared_ptr<Matrix> Track2Vec::createRandomMatrix() const
{
    int64_t m = dict_->ntracks() + dict_->ngenres() + dict_->nartists();
    std::shared_ptr<Matrix> input = std::make_shared<Matrix>(m, args_->dim);
    input->randomInit(args_->seed);
    
    return input;
}

std::shared_ptr<Matrix> Track2Vec::createTrainOutputMatrix() const
{
    int64_t m = dict_->ntracks();
    std::shared_ptr<Matrix> output = std::make_shared<Matrix>(m, args_->dim);
    output->zero();
    
    return output;
}

void Track2Vec::setInputMatrixFromFile(const std::string &outputDir)
{
    if (!input_)
    {
        throw std::runtime_error("Model never trained");
    }
    
    std::string genre_filename = outputDir + "/" + model_input_genre;
    loadGenreInputVectors(genre_filename);
    
    std::string artist_filename = outputDir + "/" + model_input_artist;
    loadArtistInputVectors(artist_filename);
    
    std::string track_filename = outputDir + "/" + model_input_track;
    loadTrackInputVectors(track_filename);
}

void Track2Vec::setOutputMatrixFromFile(const std::string &outputDir)
{
    if (!output_)
    {
        throw std::runtime_error("input matrix is not available");
    }
    
    std::string output_filename = outputDir + "/" + model_output_track;
    loadOutputMatrix(output_filename);
}

void Track2Vec::loadTrackInputVectors(const std::string &filename)
{
    if (!input_)
    {
        throw std::runtime_error("input matrix is not available");
    }
    
    std::ifstream ifs(filename, std::ofstream::binary);
    if (!ifs.is_open())
    {
        std::cerr << ">> " << filename << " does not exists" << std::endl;
        return;
    }
    
    int64_t track_cnt = 0;
    
    for (std::string line; std::getline(ifs, line);)
    {
        json j = json::parse(line);
        
        std::string track_id = j["track_id"];
        std::vector<double> vec = j["vector"];
        assert(args_->dim == vec.size());
        
        int64_t idx = dict_->getTrackIdx(track_id);
        
        if (0 > idx)
            continue;
        
        auto &trackEntry = dict_->getTrackEntry(track_id);
        trackEntry.lr_alpha = args_->pretrained_lr;
        input_->addVectorToRow(vec, idx);
        
        track_cnt++;
    }
    
    ifs.close();
    
    if( args_->verbose > 0) {
        std::cerr << "Load pretrained input track vector [" << track_cnt << "]: " <<  filename << std::endl;
    }
}

void Track2Vec::loadArtistInputVectors(const std::string &filename)
{
    if (!input_)
    {
        throw std::runtime_error("input matrix is not available");
    }
    
    std::ifstream ifs(filename, std::ofstream::binary);
    if (!ifs.is_open())
    {
        std::cerr << ">> " << filename << " does not exists" << std::endl;
        return;
    }
    
    int artist_cnt = 0;
    
    for (std::string line; std::getline(ifs, line);)
    {
        json j = json::parse(line);
        
        std::string artist_id = j["artist_id"];
        std::vector<double> vec = j["vector"];
        assert(args_->dim == vec.size());
        
        int64_t idx = dict_->getArtistIdx(artist_id);
        if (0 > idx)
            continue;
        
        input_->addVectorToRow(vec, idx);
        artist_cnt++;
    }
    
    ifs.close();
    
    if( args_->verbose > 0) {
        std::cerr << "Load pretrained input artist vector [" << artist_cnt << "]: " <<  filename << std::endl;
    }
}

void Track2Vec::loadGenreInputVectors(const std::string &filename)
{
    if (!input_)
    {
        throw std::runtime_error("input matrix is not available");
    }
    
    std::ifstream ifs(filename, std::ofstream::binary);
    if (!ifs.is_open())
    {
        std::cerr << ">> " << filename << " does not exists" << std::endl;
        return;
    }
    
    int64_t genre_cnt = 0;
    
    for (std::string line; std::getline(ifs, line);)
    {
        json j = json::parse(line);
        
        std::string genre_id = j["genre_id"];
        std::vector<double> vec = j["vector"];
        assert(args_->dim == vec.size());
        
        int64_t idx = dict_->getGenreIdx(genre_id);
        
        if (0 > idx)
            continue;
        
        input_->addVectorToRow(vec, idx);
        genre_cnt++;
    }
    
    ifs.close();
    
    if( args_->verbose > 0) {
        std::cerr << "Load pretrained input genre vector [" << genre_cnt << "]: " <<  filename << std::endl;
    }
}

void Track2Vec::loadOutputMatrix(const std::string &filename)
{
    if (!output_)
    {
        throw std::runtime_error("output matrix is not available");
    }
    
    int64_t output_cnt = 0;
    std::ifstream ifs(filename, std::ofstream::binary);
    if (!ifs.is_open())
    {
        std::cerr << ">> " << filename << " does not exists" << std::endl;
        return;
    }
    
    for (std::string line; std::getline(ifs, line);)
    {
        json j = json::parse(line);
        
        std::string track_id = j["track_id"];
        std::vector<double> vec = j["vector"];
        assert(args_->dim == vec.size());
        
        int64_t idx = dict_->getTrackIdx(track_id);
        if (0 > idx)
            continue;
        
        output_->addVectorToRow(vec, idx);
        output_cnt++;
    }
    ifs.close();
    
    if( args_->verbose > 0) {
        std::cerr << "Load pretrained output vector [" << output_cnt << "]: " <<  filename << std::endl;
    }
}

void Track2Vec::loadData()
{
    std::ifstream ifs(args_->input);
    
    if (!ifs.is_open())
    {
        throw std::invalid_argument(args_->input + " cannot be opened for loading data!");
    }
    
    int64_t idx = 0;
    
    do
    {
        std::vector<std::string> tracks;
        
        if (dict_->getRecord(ifs, tracks))
        {
            data_.emplace(idx++, std::move(tracks));
            
            if (args_->verbose > 2 && idx % 1000 == 0)
            {
                std::cerr << ">> Load [" << idx / 1000 << "K] characters into memory" << std::endl;
            }
        }
        
    } while (false == ifs.eof());
    
    ifs.close();
}

} // namespace track2vec
