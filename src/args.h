/**
 # Copyright (c) 2020-present, Dreamus, Inc.
 # All rights reserved.
 **/

#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace track2vec
{

class Args
{
private:
    std::unordered_set<std::string> manualArgs_;
    void printHelp();
    void printValue();
    
public:
    explicit Args();
    
    void parseArgs(const std::vector<std::string>&);
    std::string input;
    std::string outputDir;
    std::string metaFileName;
    std::string yyyymmddhh;
    std::string s3Log;
    std::string localLog;
    double lr;
    int64_t dim;
    int64_t ntree;
    int64_t ws;
    int64_t epoch;
    int64_t neg;
    int64_t thread;
    int64_t threadInterval;
    int64_t verbose;
    double discard_t;
    int64_t seed;
    int64_t printInterval;
    int64_t lrUpdateRate;
    int64_t logBufferSize;
    double pretrained_lr;
    double es;
    int64_t memory;
    int64_t loadPretrained;
};


} // namespace track2vec
